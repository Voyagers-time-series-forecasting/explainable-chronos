"""
NLI-based consistency scorer for forecast verbalizations.

Uses ``text-classification`` with ``text_pair`` input so the model's native
3-class NLI head (ENTAILMENT / NEUTRAL / CONTRADICTION) is invoked directly.
This matters: ``zero-shot-classification`` reformulates the task as a topic-label
problem, hiding the contradiction class and inflating entailment scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from transformers import pipeline as hf_pipeline  # type: ignore

from extension_1.config import CONSISTENCY_THRESHOLD, NLI_MODEL_NAME
from extension_1.verbalization.types import VerbalizationResult

logger = logging.getLogger(__name__)


@dataclass
class SentenceScore:
    """NLI scores for a single (premise, hypothesis) pair."""

    sentence: str
    premise: str
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float


@dataclass
class ConsistencyReport:
    """Aggregated NLI consistency report for a verbalization."""

    overall_score: float        # mean entailment probability across all sentences
    sentence_scores: List[SentenceScore]
    is_consistent: bool         # True when overall_score >= threshold
    threshold: float
    contradiction_rate: float = 0.0  # fraction of sentences with entailment_prob < 0.30


def render_premise(grounding: Dict[str, Any]) -> str:
    """Convert a grounding dictionary to an unambiguous English premise for NLI."""
    gtype = grounding.get("type", "unknown")

    if gtype == "trend":
        slope = grounding.get("trend_slope", 0)
        norm = grounding.get("normalized_slope", 0)
        direction = grounding.get("trend_direction", "unknown")
        magnitude = grounding.get("trend_magnitude", "unknown")
        horizon = grounding.get("horizon", "?")
        return (
            f"The median forecast slope is {slope:+.4f} per period. "
            f"The normalised slope is {norm:+.4f}. "
            f"The trend is classified as {magnitude} {direction}. "
            f"The forecast horizon is {horizon} periods."
        )

    if gtype == "uncertainty":
        level = grounding.get("uncertainty_level", "unknown")
        trend = grounding.get("uncertainty_trend", "unknown")
        width = grounding.get("mean_interval_width", 0)
        rel = grounding.get("relative_uncertainty", 0)
        w_slope = grounding.get("interval_width_slope", 0)
        return (
            f"The mean prediction interval width is {width:.2f}. "
            f"The relative uncertainty is {rel:.4f}. "
            f"The interval width slope is {w_slope:+.4f} per period. "
            f"Uncertainty is classified as {level} and {trend}."
        )

    if gtype == "risk":
        parts: list[str] = []
        if grounding.get("downside_risk"):
            parts.append("The lower bound (P10) falls more than 20% below current levels.")
        if grounding.get("upside_potential"):
            parts.append("The upper bound (P90) exceeds 20% above current levels.")
        return " ".join(parts) if parts else "No significant tail risks are detected."

    if gtype == "regime_shift":
        pval = grounding.get("regime_shift_pvalue", 1.0)
        return (
            f"A Welch t-test comparing the first and second halves of the "
            f"median forecast yields a p-value of {pval:.4f}, indicating "
            f"a statistically significant mean shift at the midpoint."
        )

    if gtype == "combined":
        return " ".join(render_premise(g) for g in grounding.get("groundings", []))

    if gtype.startswith("rst_"):
        relation = grounding.get("relation", "unknown")
        nucleus = grounding.get("nucleus", "")
        satellite = grounding.get("satellite", "")
        return (
            f"The following is a {relation} relation: "
            f"Nucleus: {nucleus}. Satellite: {satellite}."
        )

    if gtype == "attribution":
        name = grounding.get("covariate_name", "unknown")
        impact = grounding.get("relative_impact_pct", 0)
        importance_score = grounding.get("importance_score", 0)
        direction = grounding.get("direction", None)
        dir_str = f" It has a {direction} effect on the forecast." if direction else ""
        return (
            f"The covariate '{name}' contributes {impact:.1f}% to the total forecast attribution "
            f"(importance score: {importance_score:.4f}).{dir_str}"
        )

    if gtype == "trajectory":
        start = grounding.get("start_value", "?")
        end = grounding.get("end_value", "?")
        pct = grounding.get("pct_change", 0)
        direction = grounding.get("end_direction", "unknown")
        tps = grounding.get("turning_points", [])
        tp_parts: list[str] = []
        for tp in tps[:3]:
            step, val, kind = tp if len(tp) == 3 else (*tp, "peak")
            verb = "peaks" if kind == "peak" else "troughs"
            tp_parts.append(f"the series {verb} near {val:.2f} around step {step}")
        tp_str = (
            f" {'; '.join(tp_parts)}, before" if tp_parts else ""
        )
        return (
            f"The median forecast starts near {start:.2f},{tp_str} "
            f"settling near {end:.2f} at the horizon "
            f"({pct:.1f}% {direction} the starting level)."
        )

    if gtype == "temporal_focus":
        covariates = grounding.get("covariates", [])
        if covariates:
            parts: list[str] = []
            for c in covariates:
                name = c.get("covariate_name", "?")
                step = c.get("peak_step", None)
                pos  = c.get("position_label", None)
                if step is not None and pos is not None:
                    parts.append(f"{name} (focused in {pos}, peak at step {step})")
                elif step is not None:
                    parts.append(f"{name} (peak at step {step})")
                else:
                    parts.append(name)
            return f"The model's temporal attention focused on: {', '.join(parts)}."
        return "Temporal attention data is available."

    # Fallback: serialise any scalar values present in the grounding dict
    parts = [f"{k}: {v}" for k, v in grounding.items() if isinstance(v, (str, int, float, bool))]
    return " ".join(parts) if parts else "Grounding information is available."


class NLIConsistencyScorer:
    """Score factual consistency of a verbalization using NLI entailment.

    Lazy-loads the NLI model on first call.
    """

    # Label sets for BART-large-MNLI (and most MNLI-trained models).
    # The model returns labels in arbitrary order; we look them up by name.
    _ENTAILMENT_LABELS = {"entailment", "ENTAILMENT", "LABEL_2"}
    _NEUTRAL_LABELS    = {"neutral",    "NEUTRAL",    "LABEL_1"}
    _CONTRA_LABELS     = {"contradiction", "CONTRADICTION", "LABEL_0"}

    # Sentences with entailment_prob below this are flagged as near-contradictions.
    _CONTRADICTION_THRESHOLD = 0.30

    def __init__(
        self,
        model_name: str = NLI_MODEL_NAME,
        device: str = "cpu",
        threshold: float = CONSISTENCY_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self._pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is None:
            logger.info("Loading NLI model %s …", self.model_name)
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device if self.device != "cpu" else -1,
                top_k=None,  # return all three class scores
            )
        return self._pipeline

    def _parse_nli_result(self, raw: List[Dict[str, Any]]) -> tuple[float, float, float]:
        """Extract (entailment, neutral, contradiction) from raw pipeline output.

        ``text-classification`` with ``top_k=None`` returns one dict per class;
        labels vary by model (e.g. LABEL_0/1/2 vs entailment/neutral/contradiction).
        """
        probs: dict[str, float] = {item["label"]: item["score"] for item in raw}
        ent = next((v for k, v in probs.items() if k in self._ENTAILMENT_LABELS), 0.0)
        neu = next((v for k, v in probs.items() if k in self._NEUTRAL_LABELS),    0.0)
        con = next((v for k, v in probs.items() if k in self._CONTRA_LABELS),     0.0)
        return float(ent), float(neu), float(con)

    def score(self, verbalization: VerbalizationResult) -> ConsistencyReport:
        """Compute NLI entailment for each (grounding premise, sentence) pair."""
        pipe = self._load()
        sentence_scores: List[SentenceScore] = []

        for idx, sentence in enumerate(verbalization.sentences):
            grounding = verbalization.grounding.get(f"sentence_{idx}", {})
            premise = render_premise(grounding)

            if not premise.strip() or not sentence.strip():
                logger.debug("Skipping NLI for sentence %d — empty premise or sentence.", idx)
                sentence_scores.append(
                    SentenceScore(
                        sentence=sentence,
                        premise=premise,
                        entailment_prob=0.5,
                        neutral_prob=0.3,
                        contradiction_prob=0.2,
                    )
                )
                continue

            # Feed premise + sentence as a text_pair so the 3-class NLI head is used directly.
            raw = pipe({"text": premise, "text_pair": sentence})
            ent, neu, con = self._parse_nli_result(raw)

            sentence_scores.append(
                SentenceScore(
                    sentence=sentence,
                    premise=premise,
                    entailment_prob=ent,
                    neutral_prob=neu,
                    contradiction_prob=con,
                )
            )
            logger.debug(
                "NLI sentence %d: ent=%.3f neu=%.3f con=%.3f | %s",
                idx, ent, neu, con, sentence[:60],
            )

        if not sentence_scores:
            overall = 0.0
            contradiction_rate = 0.0
        else:
            overall = float(np.mean([s.entailment_prob for s in sentence_scores]))
            contradiction_rate = float(
                np.mean([s.entailment_prob < self._CONTRADICTION_THRESHOLD for s in sentence_scores])
            )

        return ConsistencyReport(
            overall_score=overall,
            sentence_scores=sentence_scores,
            is_consistent=overall >= self.threshold,
            threshold=self.threshold,
            contradiction_rate=contradiction_rate,
        )


class SemanticSimilarityScorer:
    """Cosine similarity between LLM output and template reference via SBERT.

    A score near 1.0 means the LLM text is semantically close to the template draft;
    a low score flags drift. Falls back to Jaccard token-overlap if
    ``sentence-transformers`` is not installed.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info("SemanticSimilarityScorer: loaded %s", self.model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed — "
                    "falling back to Jaccard token-overlap similarity."
                )
                self._model = "fallback"
        return self._model

    def score(self, llm_text: str, template_text: str) -> float:
        """Return cosine similarity in [0, 1] between *llm_text* and *template_text*."""
        if not llm_text.strip() or not template_text.strip():
            return 0.0

        model = self._load()
        if model == "fallback":
            return self._jaccard(llm_text, template_text)

        embeddings = model.encode(
            [llm_text, template_text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        a, b = embeddings[0], embeddings[1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """Token-level Jaccard similarity as a no-model fallback."""
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)
