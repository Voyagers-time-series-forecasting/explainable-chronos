"""Template-based forecast verbalizer."""

from __future__ import annotations

import logging
import random
from enum import Enum
from typing import Any

from extension_1.config import RANDOM_SEED
from extension_1.verbalization.types import VerbalizationResult
from extension_1.verbalization.trajectory import verbalize_temporal_focus, verbalize_trajectory
from extension_1.attribution.types import AttributionResult
from extension_1.features.extractor import ForecastFeatures

logger = logging.getLogger(__name__)


class RSTRelation(Enum):
    CAUSE = "cause"
    CONTRAST = "contrast"
    CONCESSION = "concession"
    ELABORATION = "elaboration"
    SEQUENCE = "sequence"


class DiscoursePlanner:
    """Selects RST relations based on feature + attribution combinations."""

    def plan(
        self,
        features: ForecastFeatures,
        attribution: AttributionResult | None = None,
    ) -> list[tuple[RSTRelation, dict[str, Any]]]:
        """Return (relation, template_kwargs) pairs to render as RST sentences."""
        relations: list[tuple[RSTRelation, dict[str, Any]]] = []

        if features.trend_direction == "rising" and features.downside_risk:
            relations.append((
                RSTRelation.CONCESSION,
                {
                    "nucleus": f"values are {features.trend_magnitude} expected to increase",
                    "satellite": "a downside scenario remains possible",
                },
            ))

        if features.trend_direction != "flat" and features.uncertainty_level == "high":
            relations.append((
                RSTRelation.CONTRAST,
                {
                    "nucleus": f"the trend is {features.trend_magnitude} {features.trend_direction}",
                    "satellite": "wide prediction intervals suggest caution",
                },
            ))

        if features.regime_shift:
            relations.append((
                RSTRelation.ELABORATION,
                {
                    "nucleus": "a structural break is detected near the midpoint",
                    "satellite": (
                        f"indicating a potential change in the underlying "
                        f"pattern (p={features.regime_shift_pvalue:.4f})"
                    ),
                },
            ))

        if attribution and attribution.attributions:
            top = attribution.attributions[0]
            if top.relative_impact_pct > 20:
                relations.append((
                    RSTRelation.CAUSE,
                    {
                        "nucleus": f"the forecast trend is {features.trend_direction}",
                        "satellite": (
                            f"primarily driven by {top.name} "
                            f"({top.relative_impact_pct:.1f}% contribution)"
                        ),
                    },
                ))

        return relations


# Phrase-variant pools for lexical diversity across windows
_DIRECTION_PHRASES: dict[str, list[str]] = {
    "rising": [
        "expected to increase",
        "projected to grow",
        "showing upward momentum",
        "on an upward trajectory",
    ],
    "falling": [
        "expected to decrease",
        "projected to decline",
        "showing downward momentum",
        "on a downward trajectory",
    ],
    "flat": [
        "expected to remain relatively stable",
        "projected to stay roughly flat",
        "showing little directional movement",
        "likely to hold near current levels",
    ],
}

_MAGNITUDE_PHRASES: dict[str, list[str]] = {
    "sharply": ["sharply", "significantly", "substantially"],
    "moderately": ["moderately", "noticeably", "meaningfully"],
    "slightly": ["slightly", "marginally", "modestly"],
}

_UNCERTAINTY_LEVEL_PHRASES: dict[str, list[str]] = {
    "high": ["wide", "broad", "substantially spread out"],
    "moderate": ["moderately wide", "of medium breadth", "at a reasonable spread"],
    "low": ["narrow", "tight", "closely clustered"],
}

_UNCERTAINTY_TREND_PHRASES: dict[str, list[str]] = {
    "widening": [
        "widening over the forecast horizon",
        "spreading further as time progresses",
        "expanding as the forecast extends",
    ],
    "narrowing": [
        "narrowing over the forecast horizon",
        "converging as time progresses",
        "tightening as the forecast extends",
    ],
    "stable": [
        "remaining stable throughout",
        "holding steady across the horizon",
        "consistent over the forecast period",
    ],
}

_UNCERTAINTY_INTERPRETATION: dict[str, dict[str, str]] = {
    "high": {
        "widening": "rapidly growing uncertainty that warrants caution",
        "narrowing": "uncertainty that, while currently high, shows signs of convergence",
        "stable": "persistently high uncertainty throughout the forecast",
    },
    "moderate": {
        "widening": "gradually increasing uncertainty",
        "narrowing": "uncertainty that is moderating over time",
        "stable": "a consistent level of moderate uncertainty",
    },
    "low": {
        "widening": "low but gradually expanding uncertainty",
        "narrowing": "diminishing uncertainty and increasing forecast confidence",
        "stable": "confidence that remains strong throughout the forecast",
    },
}

_RST_CONNECTORS: dict[RSTRelation, list[str]] = {
    RSTRelation.CONCESSION: [
        "Although {satellite}, {nucleus}.",
        "{nucleus}, although {satellite}.",
        "Even though {satellite}, {nucleus}.",
    ],
    RSTRelation.CONTRAST: [
        "While {nucleus}, {satellite}.",
        "{nucleus}; however, {satellite}.",
        "Despite the fact that {nucleus}, {satellite}.",
    ],
    RSTRelation.CAUSE: [
        "This is {satellite}.",
        "The forecast is {satellite}.",
        "{nucleus}, {satellite}.",
    ],
    RSTRelation.ELABORATION: [
        "Furthermore, {nucleus}, {satellite}.",
        "Additionally, {nucleus}, {satellite}.",
        "Notably, {nucleus}, {satellite}.",
    ],
    RSTRelation.SEQUENCE: ["First, {nucleus}. Then, {satellite}."],
}


class TemplateVerbalizer:
    """RST-based sentence planner for forecast summaries.

    Produces a deterministic draft that the LLMVerbalizer uses as input.
    Each sentence is paired with a grounding dict for NLI consistency scoring.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed
        self._planner = DiscoursePlanner()

    def verbalize(
        self,
        features: ForecastFeatures,
        attribution: AttributionResult | None = None,
    ) -> VerbalizationResult:
        """Convert features + attribution into a natural-language summary."""
        rng = random.Random(self.seed)
        sentences: list[str] = []
        grounding: dict[str, Any] = {}
        rst_used: list[str] = []

        # Sentence 1: Trend
        direction_phrase = rng.choice(_DIRECTION_PHRASES[features.trend_direction])
        if features.trend_direction == "flat":
            trend_sentence = (
                f"The forecast indicates that values are {direction_phrase} "
                f"over the next {features.horizon} periods."
            )
        else:
            magnitude_phrase = rng.choice(_MAGNITUDE_PHRASES[features.trend_magnitude])
            trend_sentence = (
                f"The forecast indicates a {magnitude_phrase} trend, "
                f"with values {direction_phrase} over the next {features.horizon} periods."
            )
        sentences.append(trend_sentence)
        grounding["sentence_0"] = {
            "type": "trend",
            "trend_direction": features.trend_direction,
            "trend_magnitude": features.trend_magnitude,
            "trend_slope": features.trend_slope,
            "normalized_slope": features.normalized_slope,
            "horizon": features.horizon,
        }

        # Sentence 2: Trajectory (concrete P50 values + turning points)
        if features.trajectory:
            traj_sentence, traj_grounding = verbalize_trajectory(features.trajectory)
            sentences.append(traj_sentence)
            grounding[f"sentence_{len(sentences) - 1}"] = traj_grounding

        # Sentence 3: Uncertainty
        sentences.append(
            f"Prediction intervals are "
            f"{rng.choice(_UNCERTAINTY_LEVEL_PHRASES[features.uncertainty_level])} and "
            f"{rng.choice(_UNCERTAINTY_TREND_PHRASES[features.uncertainty_trend])}, "
            f"suggesting {_UNCERTAINTY_INTERPRETATION[features.uncertainty_level][features.uncertainty_trend]}."
        )
        grounding[f"sentence_{len(sentences) - 1}"] = {
            "type": "uncertainty",
            "uncertainty_level": features.uncertainty_level,
            "uncertainty_trend": features.uncertainty_trend,
            "mean_interval_width": features.mean_interval_width,
            "relative_uncertainty": features.relative_uncertainty,
            "interval_width_slope": features.interval_width_slope,
        }

        # Sentence 4: Tail-risk flags (conditional)
        risk_parts: list[str] = []
        if features.downside_risk:
            risk_parts.append(
                "a risk of significant downside, with lower bounds exceeding 20% below current levels"
            )
        if features.upside_potential:
            risk_parts.append(
                "potential for significant upside, with upper bounds exceeding 20% above current levels"
            )
        if risk_parts:
            sentences.append("Notably, there is " + " and ".join(risk_parts) + ".")
            grounding[f"sentence_{len(sentences) - 1}"] = {
                "type": "risk",
                "downside_risk": features.downside_risk,
                "upside_potential": features.upside_potential,
                "regime_shift": features.regime_shift,
            }

        # RST-driven discourse sentences
        for relation, kwargs in self._planner.plan(features, attribution):
            rst_sentence = rng.choice(_RST_CONNECTORS[relation]).format(**kwargs)
            sentences.append(rst_sentence[0].upper() + rst_sentence[1:])
            rst_used.append(relation.value)
            grounding[f"sentence_{len(sentences) - 1}"] = {
                "type": f"rst_{relation.value}",
                "relation": relation.value,
                **kwargs,
            }

        # Attribution sentences (one per top-k covariate)
        if attribution and attribution.attributions:
            for attr in attribution.attributions[: attribution.top_k]:
                direction_word = getattr(attr, "direction", "positive")
                sentences.append(
                    f"{attr.name.replace('_', ' ').title()} has a {direction_word} effect "
                    f"on the forecast, contributing {attr.relative_impact_pct:.1f}% of the "
                    f"total attribution."
                )
                grounding[f"sentence_{len(sentences) - 1}"] = {
                    "type": "attribution",
                    "covariate_name": attr.name,
                    "importance_score": attr.importance_score,
                    "relative_impact_pct": attr.relative_impact_pct,
                    "direction": direction_word,
                }

        # Temporal focus sentence (only when attention saliency is available)
        if attribution and attribution.temporal:
            history_length = len(attribution.temporal[0].saliency)
            tpf_sentence, tpf_grounding = verbalize_temporal_focus(
                attribution.temporal, history_length
            )
            if tpf_sentence:
                sentences.append(tpf_sentence)
                grounding[f"sentence_{len(sentences) - 1}"] = tpf_grounding

        return VerbalizationResult(
            summary=" ".join(sentences),
            sentences=sentences,
            grounding=grounding,
            rst_relations=rst_used,
        )
