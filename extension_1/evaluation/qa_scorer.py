"""Claim-based faithfulness scorer for forecast verbalizations.

Evaluates whether a verbalization correctly conveys the underlying
forecast facts by constructing natural-language claims from
ForecastFeatures and measuring how well each claim is reflected in
the verbalization text via Sentence-BERT cosine similarity.

Design (QAFactEval / FactScore inspired)
-----------------------------------------
For each factual slot in ForecastFeatures / AttributionResult:

  1. A natural-language **claim** is constructed from the ground-truth
     feature value, including synonyms to handle paraphrase:
       trend_direction=flat → "The forecast trend is flat, stable,
                               or remaining relatively unchanged"

  2. The verbalization is split into sentences.

  3. The **maximum cosine similarity** between the claim and any sentence
     in the verbalization is computed using Sentence-BERT
     (all-MiniLM-L6-v2, ~80 MB).

  4. Per-slot scores are averaged into a ``coverage_score`` (0–1).

Advantages over extractive QA + matching
-----------------------------------------
- No QA model needed — one SBERT model does everything.
- Works naturally for paraphrase: "mostly the same" ≈ "flat",
  "minimum temperature" ≈ "temp_min", "getting wider" ≈ "widening".
- Sentence-level max avoids dilution from unrelated content.
- Avoids QA extraction failures on abstract categorical questions.

Reference: Fabbri et al. (2022) QAFactEval; Min et al. (2023) FactScore.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from extension_1.config import SBERT_MODEL_NAME
from extension_1.features.extractor import ForecastFeatures
from extension_1.attribution.types import AttributionResult

logger = logging.getLogger(__name__)


# ─────────────────── claim templates ──────────────────────────────────
# Each claim is an enriched natural-language assertion that includes
# synonyms / paraphrases so that SBERT cosine similarity is robust to
# different surface forms in the verbalization.

_TREND_DIR_CLAIMS: dict[str, str] = {
    "rising":  "The forecast trend is rising, increasing, or growing upward over time",
    "falling": "The forecast trend is falling, decreasing, declining, or going down",
    "flat":    "The forecast trend is flat, stable, roughly unchanged, or remaining steady",
}

_TREND_MAG_CLAIMS: dict[str, str] = {
    "sharply":    "The trend is sharp, significant, substantial, or dramatic",
    "moderately": "The trend is moderate, noticeable, or meaningful",
    "slightly":   "The trend is slight, gradual, small, or minimal — only a minor change",
}

_UNCERT_LEVEL_CLAIMS: dict[str, str] = {
    "high":     "The forecast uncertainty is high and prediction intervals are wide or broad",
    "moderate": "The forecast uncertainty is moderate with a reasonable spread of outcomes",
    "low":      "The forecast uncertainty is low with narrow, tight prediction intervals",
}

_UNCERT_TREND_CLAIMS: dict[str, str] = {
    "widening":  (
        "The prediction intervals are widening, expanding, or getting wider "
        "as the forecast extends further ahead"
    ),
    "narrowing": (
        "The prediction intervals are narrowing, converging, tightening, "
        "or getting smaller over the forecast horizon"
    ),
    "stable":    (
        "The prediction intervals are stable, consistent, or holding steady "
        "throughout the forecast period"
    ),
}


def _build_direction_claim(direction: str) -> str:
    return _TREND_DIR_CLAIMS.get(direction, f"The forecast is {direction}")


def _build_magnitude_claim(magnitude: str) -> str:
    return _TREND_MAG_CLAIMS.get(magnitude, f"The trend magnitude is {magnitude}")


def _build_uncert_level_claim(level: str) -> str:
    return _UNCERT_LEVEL_CLAIMS.get(level, f"The uncertainty is {level}")


def _build_uncert_trend_claim(trend: str) -> str:
    return _UNCERT_TREND_CLAIMS.get(trend, f"The prediction intervals are {trend}")


# ─────────────── numeric helper ───────────────────────────────────────

def _numeric_match(text: str, expected_value: float, tol: float = 0.15) -> float:
    """1.0 if text contains the expected number within tolerance."""
    for raw in re.findall(r"[-+]?\d*\.?\d+", text):
        try:
            if abs(float(raw) - expected_value) <= tol * (abs(expected_value) + 1e-9):
                return 1.0
        except ValueError:
            continue
    return 0.0


# ─────────────────────── data classes ─────────────────────────────────
@dataclass
class QASlot:
    """One factual claim derived from ForecastFeatures."""
    slot_name: str
    claim: str                              # natural-language assertion of the fact
    expected_answer: str                    # concise label for logging / traces
    slot_type: str                          # "categorical"|"numeric"|"boolean"|"entity"
    expected_numeric: float | None = None


@dataclass
class SlotScore:
    """Score for a single factual claim."""
    slot_name: str
    claim: str
    expected_answer: str
    best_sentence: str                      # verbalization sentence with highest cosine
    score: float                            # max cosine similarity 0–1
    is_correct: bool


@dataclass
class QAFaithfulnessReport:
    """Aggregated claim-based faithfulness report."""
    coverage_score: float                   # mean slot score across all slots
    slot_scores: list[SlotScore]
    correct_slots: int
    total_slots: int
    missing_slots: list[str]                # slot names with score == 0
    is_faithful: bool
    threshold: float = 0.55


# ─────────────── slot builder ──────────────────────────────────────────
def build_qa_slots(
    features: ForecastFeatures,
    attribution: AttributionResult | None = None,
) -> list[QASlot]:
    """Generate factual claim slots from ForecastFeatures and AttributionResult."""
    slots: list[QASlot] = [
        QASlot(
            slot_name="trend_direction",
            claim=_build_direction_claim(features.trend_direction),
            expected_answer=features.trend_direction,
            slot_type="categorical",
        ),
        QASlot(
            slot_name="trend_magnitude",
            claim=_build_magnitude_claim(features.trend_magnitude),
            expected_answer=features.trend_magnitude,
            slot_type="categorical",
        ),
        QASlot(
            slot_name="uncertainty_level",
            claim=_build_uncert_level_claim(features.uncertainty_level),
            expected_answer=features.uncertainty_level,
            slot_type="categorical",
        ),
        QASlot(
            slot_name="uncertainty_trend",
            claim=_build_uncert_trend_claim(features.uncertainty_trend),
            expected_answer=features.uncertainty_trend,
            slot_type="categorical",
        ),
    ]

    if features.downside_risk:
        slots.append(QASlot(
            slot_name="downside_risk",
            claim=(
                "There is significant downside risk — the forecast could be "
                "much worse than expected or values could fall sharply"
            ),
            expected_answer="true",
            slot_type="boolean",
        ))
    if features.upside_potential:
        slots.append(QASlot(
            slot_name="upside_potential",
            claim=(
                "There is significant upside potential — the forecast could be "
                "much better than expected or values could rise substantially"
            ),
            expected_answer="true",
            slot_type="boolean",
        ))
    if features.regime_shift:
        slots.append(QASlot(
            slot_name="regime_shift",
            claim=(
                "There is a structural break or significant pattern change "
                "detected in the forecast — the behaviour shifts meaningfully"
            ),
            expected_answer="true",
            slot_type="boolean",
        ))

    if attribution and attribution.attributions:
        top = attribution.attributions[0]
        name_clean = top.name.replace("_", " ")
        dir_phrase = (
            "increases or positively drives the forecast upward"
            if top.direction == "positive"
            else "decreases or negatively pulls the forecast downward"
        )
        slots.append(QASlot(
            slot_name="top_covariate_name",
            claim=(
                f"The main driver or most important factor influencing the forecast "
                f"is {name_clean} or {top.name}"
            ),
            expected_answer=name_clean,
            slot_type="entity",
        ))
        slots.append(QASlot(
            slot_name="top_covariate_direction",
            claim=(
                f"{name_clean} has a {top.direction} effect and {dir_phrase}"
            ),
            expected_answer=top.direction,
            slot_type="categorical",
        ))
        slots.append(QASlot(
            slot_name="top_covariate_impact_pct",
            claim=f"{name_clean} contributes {top.relative_impact_pct:.1f}% of the total forecast attribution",
            expected_answer=f"{top.relative_impact_pct:.1f}",
            slot_type="numeric",
            expected_numeric=top.relative_impact_pct,
        ))

    return slots


# ─────────────── scorer ───────────────────────────────────────────────
class QAFaithfulnessScorer:
    """Score factual coverage of a verbalization using claim-based SBERT similarity.

    For each factual slot derived from the forecast features, a natural-language
    claim is constructed.  The verbalization is split into sentences, and the
    maximum cosine similarity between the claim and any sentence is used as the
    slot score.  No extractive QA model is needed.

    Parameters
    ----------
    sbert_model_name : str
        HuggingFace sentence-transformers model ID.
    correct_threshold : float
        Cosine similarity above which a slot is considered correctly covered.
    faithful_threshold : float
        Mean coverage score above which the verbalization is faithful.
    """

    def __init__(
        self,
        sbert_model_name: str = SBERT_MODEL_NAME,
        correct_threshold: float = 0.60,
        faithful_threshold: float = 0.55,
    ) -> None:
        self.sbert_model_name = sbert_model_name
        self.correct_threshold = correct_threshold
        self.faithful_threshold = faithful_threshold
        self._sbert: Any = None

    def _load_sbert(self) -> Any:
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SBERT model %s …", self.sbert_model_name)
            self._sbert = SentenceTransformer(self.sbert_model_name)
        return self._sbert

    def _max_sentence_cosine(
        self, claim: str, sentences: list[str], claim_emb: np.ndarray
    ) -> tuple[float, str]:
        """Max cosine similarity between claim and any sentence; returns (score, sentence)."""
        if not sentences:
            return 0.0, ""

        model = self._load_sbert()
        sent_embs = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        claim_norm = np.linalg.norm(claim_emb)

        best_score, best_sent = 0.0, ""
        for sent, emb in zip(sentences, sent_embs):
            denom = claim_norm * float(np.linalg.norm(emb))
            if denom < 1e-12:
                continue
            cos = float(np.clip(np.dot(claim_emb, emb) / denom, 0.0, 1.0))
            if cos > best_score:
                best_score, best_sent = cos, sent

        return best_score, best_sent

    def _score_slot(
        self,
        slot: QASlot,
        sentences: list[str],
        verbalization_text: str,
        claim_emb: np.ndarray,
    ) -> tuple[float, str]:
        """Score a single slot. Returns (score, best_sentence)."""
        if slot.slot_type == "numeric" and slot.expected_numeric is not None:
            # Regex match on full text first (exact numeric check)
            num_score = _numeric_match(verbalization_text, slot.expected_numeric)
            if num_score > 0:
                # Find the sentence that contains the number for trace display
                for s in sentences:
                    if _numeric_match(s, slot.expected_numeric) > 0:
                        return num_score, s
                return num_score, verbalization_text[:80]
            # Fallback: cosine similarity on the claim
            cos, best = self._max_sentence_cosine(slot.claim, sentences, claim_emb)
            return cos * 0.8, best  # discount — numeric fact wasn't found explicitly

        return self._max_sentence_cosine(slot.claim, sentences, claim_emb)

    def score(
        self,
        verbalization_text: str,
        features: ForecastFeatures,
        attribution: AttributionResult | None = None,
    ) -> QAFaithfulnessReport:
        """Compute claim-based faithfulness for a verbalization.

        Parameters
        ----------
        verbalization_text : str
            Full text of the verbalization to evaluate.
        features : ForecastFeatures
            Ground-truth forecast features used to build claims.
        attribution : AttributionResult, optional
            Attribution results; adds covariate slots if provided.

        Returns
        -------
        QAFaithfulnessReport
        """
        model = self._load_sbert()
        slots = build_qa_slots(features, attribution)

        # Split verbalization into non-empty sentences once
        sentences = [
            s.strip() for s in re.split(r"[.!?]+", verbalization_text) if s.strip()
        ]

        # Encode all claims in one batch for efficiency
        claims = [slot.claim for slot in slots]
        claim_embs = model.encode(claims, convert_to_numpy=True, show_progress_bar=False)

        slot_scores: list[SlotScore] = []

        for slot, claim_emb in zip(slots, claim_embs):
            try:
                slot_score, best_sent = self._score_slot(
                    slot, sentences, verbalization_text, claim_emb
                )
            except Exception as exc:
                logger.warning("Claim scoring failed for slot '%s': %s", slot.slot_name, exc)
                slot_score, best_sent = 0.0, ""

            slot_scores.append(SlotScore(
                slot_name=slot.slot_name,
                claim=slot.claim,
                expected_answer=slot.expected_answer,
                best_sentence=best_sent,
                score=slot_score,
                is_correct=slot_score >= self.correct_threshold,
            ))
            logger.debug(
                "Claim %-28s expected=%-12s cosine=%.3f | %s",
                slot.slot_name, slot.expected_answer, slot_score,
                best_sent[:60],
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
