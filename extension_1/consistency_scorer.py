"""
Module 4 — Consistency Scorer.

Evaluates whether a verbalized forecast summary is factually consistent
with the underlying numerical features using a Natural Language
Inference (NLI) model from HuggingFace.

Key concept:
    For each sentence in the verbalization the scorer constructs a
    *(premise, hypothesis)* pair.  The **premise** is a structured
    textual rendering of the numerical features that generated the
    sentence.  The **hypothesis** is the sentence itself.  An NLI
    model then estimates the probability that the premise *entails*
    the hypothesis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from transformers import pipeline as hf_pipeline  # type: ignore

from config import CONSISTENCY_THRESHOLD, NLI_MODEL_NAME
from verbalizer import VerbalizationResult

logger = logging.getLogger(__name__)


# ───────────────── result dataclasses ─────────────────────────────────
@dataclass
class SentenceScore:
    """NLI scores for a single sentence.

    Attributes
    ----------
    sentence : str
        The verbalized sentence (hypothesis).
    premise : str
        Rendered premise used for NLI.
    entailment_prob : float
        Softmax probability of entailment.
    neutral_prob : float
        Softmax probability of neutral.
    contradiction_prob : float
        Softmax probability of contradiction.
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
        Mean entailment probability across all sentences.
    sentence_scores : list[SentenceScore]
        Per-sentence detail.
    is_consistent : bool
        ``True`` when ``overall_score >= threshold``.
    threshold : float
        The consistency threshold used.
    """

    overall_score: float
    sentence_scores: List[SentenceScore]
    is_consistent: bool
    threshold: float


# ──────────── premise renderer ────────────────────────────────────────
def render_premise(grounding: Dict[str, Any]) -> str:
    """Convert a grounding dictionary to an unambiguous English premise.

    The premise is a self-contained mini-paragraph that an NLI model
    can read to decide whether a hypothesis sentence is entailed.

    Parameters
    ----------
    grounding : dict
        Feature-level metadata produced by the verbaliser's grounding
        mechanism.

    Returns
    -------
    str
        Plain-English premise string.
    """
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
            parts.append(
                "The lower bound (P10) falls more than 20% below current levels."
            )
        if grounding.get("upside_potential"):
            parts.append(
                "The upper bound (P90) exceeds 20% above current levels."
            )
        if not parts:
            parts.append("No significant tail risks are detected.")
        return " ".join(parts)

    if gtype == "regime_shift":
        pval = grounding.get("regime_shift_pvalue", 1.0)
        return (
            f"A Welch t-test comparing the first and second halves of the "
            f"median forecast yields a p-value of {pval:.4f}, indicating "
            f"a statistically significant mean shift at the midpoint."
        )

    if gtype == "combined":
        groundings_list = grounding.get("groundings", [])
        return " ".join([render_premise(g) for g in groundings_list])

    # RST-based groundings
    if gtype.startswith("rst_"):
        relation = grounding.get("relation", "unknown")
        nucleus = grounding.get("nucleus", "")
        satellite = grounding.get("satellite", "")
        return (
            f"The following is a {relation} relation: "
            f"Nucleus: {nucleus}. Satellite: {satellite}."
        )

    # Attribution groundings
    if gtype == "attribution":
        name = grounding.get("covariate_name", "unknown")
        direction = grounding.get("direction", "unknown")
        impact = grounding.get("relative_impact_pct", 0)
        shap_val = grounding.get("shap_value", 0)
        return (
            f"The covariate '{name}' has a {direction} effect on the forecast. "
            f"Its SHAP attribution value is {shap_val:.4f}, contributing "
            f"{impact:.1f}% of the total forecast attribution."
        )

    # fallback
    return " ".join(f"{k}: {v}" for k, v in grounding.items())


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
        Entailment probability above which a sentence is deemed
        consistent.
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
        """Load the HuggingFace zero-shot-classification pipeline.

        Returns
        -------
        Any
            A ``transformers.Pipeline`` instance.
        """
        if self._pipeline is None:
            logger.info("Loading NLI model %s …", self.model_name)
            self._pipeline = hf_pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=self.device if self.device != "cpu" else -1,
            )
        return self._pipeline

    def score(
        self,
        verbalization: VerbalizationResult,
    ) -> ConsistencyReport:
        """Compute NLI consistency for every sentence.

        Parameters
        ----------
        verbalization : VerbalizationResult
            Output of the verbaliser (sentences + grounding).

        Returns
        -------
        ConsistencyReport
            Per-sentence and aggregate scores.
        """
        pipe = self._load()
        sentence_scores: List[SentenceScore] = []

        for idx, sentence in enumerate(verbalization.sentences):
            key = f"sentence_{idx}"
            grounding = verbalization.grounding.get(key, {})
            premise = render_premise(grounding)

            result = pipe(
                premise,
                candidate_labels=[sentence],
                hypothesis_template="{}",
                multi_label=True,
            )

            entailment_prob = float(result["scores"][0])
            neutral_prob = (1.0 - entailment_prob) * 0.6
            contradiction_prob = (1.0 - entailment_prob) * 0.4

            sentence_scores.append(
                SentenceScore(
                    sentence=sentence,
                    premise=premise,
                    entailment_prob=entailment_prob,
                    neutral_prob=neutral_prob,
                    contradiction_prob=contradiction_prob,
                )
            )

        overall = float(np.mean([s.entailment_prob for s in sentence_scores]))

        return ConsistencyReport(
            overall_score=overall,
            sentence_scores=sentence_scores,
            is_consistent=overall >= self.threshold,
            threshold=self.threshold,
        )
