"""QA-based faithfulness scorer for forecast verbalizations.

Evaluates whether a verbalization correctly conveys the underlying
forecast facts by generating factual questions from ForecastFeatures
and checking whether a QA model can recover the correct answers from
the verbalization text.

This approach is surface-form agnostic: it rewards any phrasing that
correctly conveys the forecast information, unlike NLI-based scoring
which penalises paraphrase distance from a structured numerical premise.

Design
------
For each factual slot in ForecastFeatures / AttributionResult:
  1. A natural-language question is generated (e.g., "What direction is
     the trend?").
  2. An extractive QA model (roberta-base-squad2) reads the verbalization
     and extracts an answer span.
  3. The extracted span is scored against the expected answer using
     **Sentence-BERT cosine similarity** (QAFactEval-style), which
     handles paraphrase naturally — "minimum temperature" ≈ "temp_min",
     "mostly the same" ≈ "flat", "getting wider" ≈ "widening".
  4. Numeric slots use regex extraction first; cosine similarity is used
     as a fallback when no number is found in the extracted span.
  5. Per-slot scores are averaged into a ``coverage_score`` (0–1).

Reference: Fabbri et al. (2022) QAFactEval.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from transformers import pipeline as hf_pipeline

from extension_1.config import QA_MODEL_NAME, SBERT_MODEL_NAME
from extension_1.features.extractor import ForecastFeatures
from extension_1.attribution.types import AttributionResult

logger = logging.getLogger(__name__)


def _numeric_match(extracted: str, expected_value: float, tol: float = 0.15) -> float:
    """1.0 if the extracted span contains the expected number within tolerance."""
    for raw in re.findall(r"[-+]?\d*\.?\d+", extracted):
        try:
            if abs(float(raw) - expected_value) <= tol * (abs(expected_value) + 1e-9):
                return 1.0
        except ValueError:
            continue
    return 0.0



# ─────────────────────── data classes ─────────────────────────────────
@dataclass
class QASlot:
    """One factual question derived from ForecastFeatures."""
    slot_name: str
    question: str
    expected_answer: str
    slot_type: str                      # "categorical" | "numeric" | "boolean" | "entity"
    expected_numeric: float | None = None


@dataclass
class SlotScore:
    """Score for a single QA slot."""
    slot_name: str
    question: str
    expected_answer: str
    extracted_answer: str
    qa_confidence: float                # QA model span-extraction confidence
    score: float                        # semantic match score 0–1
    is_correct: bool


@dataclass
class QAFaithfulnessReport:
    """Aggregated QA-based faithfulness report."""
    coverage_score: float               # mean slot score across all slots
    slot_scores: list[SlotScore]
    correct_slots: int
    total_slots: int
    missing_slots: list[str]            # slot names with score == 0
    is_faithful: bool
    threshold: float = 0.60


# ─────────────── slot builder ──────────────────────────────────────────
def build_qa_slots(
    features: ForecastFeatures,
    attribution: AttributionResult | None = None,
) -> list[QASlot]:
    """Generate factual Q&A pairs from ForecastFeatures and AttributionResult."""
    slots: list[QASlot] = [
        QASlot(
            slot_name="trend_direction",
            question=(
                "What direction is the forecast trend — "
                "is it rising, falling, or flat?"
            ),
            expected_answer=features.trend_direction,
            slot_type="categorical",
        ),
        QASlot(
            slot_name="trend_magnitude",
            question=(
                "How strong or pronounced is the trend — "
                "is it sharp, moderate, or slight?"
            ),
            expected_answer=features.trend_magnitude,
            slot_type="categorical",
        ),
        QASlot(
            slot_name="uncertainty_level",
            question=(
                "How uncertain is the forecast — are prediction "
                "intervals high, moderate, or low?"
            ),
            expected_answer=features.uncertainty_level,
            slot_type="categorical",
        ),
        QASlot(
            slot_name="uncertainty_trend",
            question=(
                "Are the prediction intervals widening, "
                "narrowing, or staying stable over time?"
            ),
            expected_answer=features.uncertainty_trend,
            slot_type="categorical",
        ),
    ]

    if features.downside_risk:
        slots.append(QASlot(
            slot_name="downside_risk",
            question="Is there a significant downside risk or danger of large losses mentioned?",
            expected_answer="true",
            slot_type="boolean",
        ))
    if features.upside_potential:
        slots.append(QASlot(
            slot_name="upside_potential",
            question="Is there meaningful upside potential or opportunity for large gains mentioned?",
            expected_answer="true",
            slot_type="boolean",
        ))
    if features.regime_shift:
        slots.append(QASlot(
            slot_name="regime_shift",
            question="Is a structural break, pattern change, or shift in behaviour mentioned?",
            expected_answer="true",
            slot_type="boolean",
        ))

    if attribution and attribution.attributions:
        top = attribution.attributions[0]
        name_clean = top.name.replace("_", " ")
        slots.append(QASlot(
            slot_name="top_covariate_name",
            question="What is the name of the main factor or driver influencing the forecast?",
            expected_answer=name_clean,
            slot_type="entity",
        ))
        slots.append(QASlot(
            slot_name="top_covariate_direction",
            question=(
                f"Does {name_clean} have a positive or negative "
                f"effect on the forecast?"
            ),
            expected_answer=top.direction,
            slot_type="categorical",
        ))
        slots.append(QASlot(
            slot_name="top_covariate_impact_pct",
            question=(
                f"What percentage of the forecast attribution "
                f"is due to {name_clean}?"
            ),
            expected_answer=f"{top.relative_impact_pct:.1f}",
            slot_type="numeric",
            expected_numeric=top.relative_impact_pct,
        ))

    return slots


# ─────────────── scorer ───────────────────────────────────────────────
class QAFaithfulnessScorer:
    """Score factual coverage of a verbalization using extractive QA.

    For each factual slot derived from the forecast features, a question
    is generated and answered by an extractive QA model reading the
    verbalization text.  The extracted answer is then compared to the
    expected value using semantic matching (synonym sets + token F1).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (must be an extractive QA model).
    device : str
        Torch device string ("cpu" or "cuda").
    correct_threshold : float
        Semantic match score above which a slot is considered correct.
    faithful_threshold : float
        Mean coverage score above which the verbalization is faithful.
    """

    def __init__(
        self,
        model_name: str = QA_MODEL_NAME,
        sbert_model_name: str = SBERT_MODEL_NAME,
        device: str = "cpu",
        correct_threshold: float = 0.60,
        faithful_threshold: float = 0.55,
    ) -> None:
        self.model_name = model_name
        self.sbert_model_name = sbert_model_name
        self.device = device
        self.correct_threshold = correct_threshold
        self.faithful_threshold = faithful_threshold
        self._pipeline: Any = None
        self._sbert: Any = None

    def _load(self) -> Any:
        if self._pipeline is None:
            logger.info("Loading QA model %s …", self.model_name)
            self._pipeline = hf_pipeline(
                "question-answering",
                model=self.model_name,
                device=self.device if self.device != "cpu" else -1,
            )
        return self._pipeline

    def _load_sbert(self) -> Any:
        """Lazy-load the Sentence-BERT model (shared across all slots)."""
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SBERT model %s …", self.sbert_model_name)
            self._sbert = SentenceTransformer(self.sbert_model_name)
        return self._sbert

    def _cosine_score(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between SBERT embeddings of two strings (0–1)."""
        model = self._load_sbert()
        embs = model.encode([text_a, text_b], convert_to_numpy=True, show_progress_bar=False)
        a, b = embs[0], embs[1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))

    def _score_slot(self, slot: QASlot, extracted: str) -> float:
        """Score extracted answer against expected using cosine similarity.

        Numeric slots: regex match first (exact), cosine fallback.
        All other slots: SBERT cosine similarity directly.
        """
        if slot.slot_type == "numeric" and slot.expected_numeric is not None:
            num_score = _numeric_match(extracted, slot.expected_numeric)
            if num_score > 0:
                return num_score
            # Fallback: cosine (handles "about half" ≈ "50")
            return self._cosine_score(extracted, slot.expected_answer) * 0.8
        return self._cosine_score(extracted, slot.expected_answer)

    def score(
        self,
        verbalization_text: str,
        features: ForecastFeatures,
        attribution: AttributionResult | None = None,
    ) -> QAFaithfulnessReport:
        """Compute QA-based faithfulness for a verbalization.

        Parameters
        ----------
        verbalization_text : str
            The full text of the verbalization to evaluate.
        features : ForecastFeatures
            Ground-truth forecast features used to build expected answers.
        attribution : AttributionResult, optional
            Attribution results; adds covariate slots if provided.

        Returns
        -------
        QAFaithfulnessReport
        """
        pipe = self._load()
        slots = build_qa_slots(features, attribution)
        slot_scores: list[SlotScore] = []

        for slot in slots:
            try:
                result = pipe(
                    question=slot.question,
                    context=verbalization_text,
                )
                extracted = result.get("answer", "").strip()
                confidence = float(result.get("score", 0.0))
            except Exception as exc:
                logger.warning("QA failed for slot '%s': %s", slot.slot_name, exc)
                extracted = ""
                confidence = 0.0

            sem_score = self._score_slot(slot, extracted)

            slot_scores.append(SlotScore(
                slot_name=slot.slot_name,
                question=slot.question,
                expected_answer=slot.expected_answer,
                extracted_answer=extracted,
                qa_confidence=confidence,
                score=sem_score,
                is_correct=sem_score >= self.correct_threshold,
            ))
            logger.debug(
                "QA %-28s expected=%-12s extracted=%-25s cosine=%.3f conf=%.3f",
                slot.slot_name, slot.expected_answer, extracted[:25],
                sem_score, confidence,
            )

        if not slot_scores:
            coverage, correct, missing = 0.0, 0, []
        else:
            coverage = float(np.mean([s.score for s in slot_scores]))
            correct  = sum(1 for s in slot_scores if s.is_correct)
            missing  = [s.slot_name for s in slot_scores if s.score == 0.0]

        return QAFaithfulnessReport(
            coverage_score=coverage,
            slot_scores=slot_scores,
            correct_slots=correct,
            total_slots=len(slot_scores),
            missing_slots=missing,
            is_faithful=coverage >= self.faithful_threshold,
            threshold=self.faithful_threshold,
        )
