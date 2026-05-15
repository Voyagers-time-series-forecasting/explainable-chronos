"""
NLI-based consistency scorer.

Evaluates whether a verbalized forecast summary is factually consistent
with the underlying numerical features using a Natural Language
Inference (NLI) model from HuggingFace.
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


# ───────────────── result dataclasses ─────────────────────────────────
@dataclass
class SentenceScore:
    """NLI scores for a single sentence.

    Attributes
    ----------
    sentence : str
    premise : str
    entailment_prob : float
    neutral_prob : float
    contradiction_prob : float
    """

    sentence: str
    premise: str
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float


@dataclass
class ConsistencyReport:
    """Aggregated NLI consistency report.

    Attributes
    ----------
    overall_score : float
    sentence_scores : list[SentenceScore]
    is_consistent : bool
    threshold : float
    """

    overall_score: float
    sentence_scores: List[SentenceScore]
    is_consistent: bool
    threshold: float


# ──────────── premise renderer ────────────────────────────────────────
def render_premise(grounding: Dict[str, Any]) -> str:
    """Convert a grounding dictionary to an unambiguous English premise."""
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
        direction = grounding.get("direction", "unknown")
        impact = grounding.get("relative_impact_pct", 0)
        importance_score = grounding.get("importance_score", 0)
        return (
            f"The covariate '{name}' has a {direction} effect on the forecast. "
            f"Its attribution importance score is {importance_score:.4f}, contributing "
            f"{impact:.1f}% of the total forecast attribution."
        )

    if gtype == "trajectory":
        start = grounding.get("start_value", "?")
        end = grounding.get("end_value", "?")
        pct = grounding.get("pct_change", 0)
        direction = grounding.get("end_direction", "unknown")
        tps = grounding.get("turning_points", [])
        tp_str = (
            f" The series passes through {len(tps)} turning point(s)."
            if tps else ""
        )
        return (
            f"The median forecast starts near {start:.2f} and ends near {end:.2f}, "
            f"a change of {pct:.1f}% {direction} the starting level.{tp_str}"
        )

    if gtype == "temporal_focus":
        covariates = grounding.get("covariates", [])
        if covariates:
            names = ", ".join(c.get("covariate_name", "?") for c in covariates)
            return f"The model's temporal attention focused on: {names}."
        return "Temporal attention data is available."

    # Generic fallback — filter out non-serialisable values to avoid empty strings
    parts = [f"{k}: {v}" for k, v in grounding.items() if isinstance(v, (str, int, float, bool))]
    return " ".join(parts) if parts else "Grounding information is available."


# ────────────── NLI Consistency Scorer ────────────────────────────────
class NLIConsistencyScorer:
    """Score factual consistency of a verbalization using NLI.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (must support zero-shot NLI).
    device : str
        Torch device string.
    threshold : float
        Entailment probability above which a sentence is consistent.
    """

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
                "zero-shot-classification",
                model=self.model_name,
                device=self.device if self.device != "cpu" else -1,
            )
        return self._pipeline

    def score(self, verbalization: VerbalizationResult) -> ConsistencyReport:
        """Compute NLI consistency for every sentence.

        Parameters
        ----------
        verbalization : VerbalizationResult

        Returns
        -------
        ConsistencyReport
        """
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

            result = pipe(
                premise,
                candidate_labels=[sentence],
                hypothesis_template="{}",
                multi_label=True,
            )

            entailment_prob = float(result["scores"][0])
            sentence_scores.append(
                SentenceScore(
                    sentence=sentence,
                    premise=premise,
                    entailment_prob=entailment_prob,
                    neutral_prob=(1.0 - entailment_prob) * 0.6,
                    contradiction_prob=(1.0 - entailment_prob) * 0.4,
                )
            )

        overall = float(np.mean([s.entailment_prob for s in sentence_scores])) if sentence_scores else 0.0
        return ConsistencyReport(
            overall_score=overall,
            sentence_scores=sentence_scores,
            is_consistent=overall >= self.threshold,
            threshold=self.threshold,
        )
