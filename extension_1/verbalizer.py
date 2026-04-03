"""
Module 3 — Verbaliser.

Converts ``ForecastFeatures`` (and optionally ``AttributionResult``)
into a natural-language summary that describes the predicted trend,
uncertainty, and covariate attributions.

Two approaches:

* **Approach A** — ``TemplateVerbalizer``: RST-based sentence planner
  with phrase-variant dictionaries and discourse-relation triggers.
* **Approach B** — ``LLMVerbalizer``: takes the template
  output and asks an LLM to rewrite it for fluency while preserving
  factual claims.  Includes structured grounding triples.

* **DiscoursePlanner**: selects which RST relations to activate based
  on combinations of features and attributions.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from config import RANDOM_SEED
from covariate_attribution import AttributionResult
from feature_extractor import ForecastFeatures

from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


# ───────────────── result dataclass ───────────────────────────────────
@dataclass
class VerbalizationResult:
    """Output of the verbalisation step.

    Attributes
    ----------
    summary : str
        Full paragraph combining all sentences.
    sentences : list[str]
        Individual sentences that compose the summary.
    grounding : dict[str, Any]
        Maps each sentence (by index label) to the numerical features
        that generated it — critical for NLI consistency checking.
    rst_relations : list[str]
        RST relations triggered during verbalization.
    """

    summary: str
    sentences: List[str]
    grounding: Dict[str, Any]
    rst_relations: List[str] = field(default_factory=list)


# ───────────────── RST discourse relations ────────────────────────────
class RSTRelation(Enum):
    """Rhetorical Structure Theory discourse relations."""

    CAUSE = "cause"
    CONTRAST = "contrast"
    CONCESSION = "concession"
    ELABORATION = "elaboration"
    SEQUENCE = "sequence"


class DiscoursePlanner:
    """Selects RST relations based on feature + attribution combinations.

    Each ``plan()`` call inspects the features and optional attribution
    and returns a list of ``(RSTRelation, template_kwargs)`` tuples that
    the verbalizer will render.
    """

    def plan(
        self,
        features: ForecastFeatures,
        attribution: Optional[AttributionResult] = None,
    ) -> List[Tuple[RSTRelation, Dict[str, Any]]]:
        """Determine which RST relations to activate.

        Returns
        -------
        list[tuple[RSTRelation, dict]]
            Each entry is a relation plus the kwargs for the template.
        """
        relations: List[Tuple[RSTRelation, Dict[str, Any]]] = []

        # CONCESSION: P50 rising but P10 below baseline
        if features.trend_direction == "rising" and features.downside_risk:
            relations.append((
                RSTRelation.CONCESSION,
                {
                    "nucleus": (
                        f"values are {features.trend_magnitude} "
                        f"expected to increase"
                    ),
                    "satellite": "a downside scenario remains possible",
                },
            ))

        # CONTRAST: strong trend but high uncertainty
        if features.trend_direction != "flat" and features.uncertainty_level == "high":
            relations.append((
                RSTRelation.CONTRAST,
                {
                    "nucleus": (
                        f"the trend is {features.trend_magnitude} "
                        f"{features.trend_direction}"
                    ),
                    "satellite": "wide prediction intervals suggest caution",
                },
            ))

        # ELABORATION: regime shift detected
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

        # CAUSE: top SHAP covariate with high positive attribution
        if attribution and attribution.attributions:
            top = attribution.attributions[0]
            if top.relative_impact_pct > 20:
                relations.append((
                    RSTRelation.CAUSE,
                    {
                        "nucleus": f"the forecast trend is {features.trend_direction}",
                        "satellite": (
                            f"primarily driven by {top.name} "
                            f"({top.direction} impact, "
                            f"{top.relative_impact_pct:.1f}% contribution)"
                        ),
                    },
                ))

            # CONTRAST: opposing attributions
            if len(attribution.attributions) >= 2:
                a, b = attribution.attributions[0], attribution.attributions[1]
                if a.direction != b.direction and b.relative_impact_pct > 15:
                    relations.append((
                        RSTRelation.CONTRAST,
                        {
                            "nucleus": (
                                f"{a.name} has a {a.direction} effect "
                                f"({a.relative_impact_pct:.1f}%)"
                            ),
                            "satellite": (
                                f"{b.name} pushes in the {b.direction} direction "
                                f"({b.relative_impact_pct:.1f}%)"
                            ),
                        },
                    ))

        return relations


# ────────── phrase-variant dictionaries ───────────────────────────────
_DIRECTION_PHRASES: Dict[str, List[str]] = {
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

_MAGNITUDE_PHRASES: Dict[str, List[str]] = {
    "sharply": ["sharply", "significantly", "substantially"],
    "moderately": ["moderately", "noticeably", "meaningfully"],
    "slightly": ["slightly", "marginally", "modestly"],
}

_UNCERTAINTY_LEVEL_PHRASES: Dict[str, List[str]] = {
    "high": ["wide", "broad", "substantially spread out"],
    "moderate": ["moderately wide", "of medium breadth", "at a reasonable spread"],
    "low": ["narrow", "tight", "closely clustered"],
}

_UNCERTAINTY_TREND_PHRASES: Dict[str, List[str]] = {
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

_UNCERTAINTY_INTERPRETATION: Dict[str, Dict[str, str]] = {
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

_RST_CONNECTORS: Dict[RSTRelation, List[str]] = {
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
    RSTRelation.SEQUENCE: [
        "First, {nucleus}. Then, {satellite}.",
    ],
}


# ─────────────── Approach A: RST-template-based ───────────────────────
class TemplateVerbalizer:
    """RST-based sentence planner for forecast summaries.

    Parameters
    ----------
    seed : int
        Random seed for reproducible phrase-variant selection.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed
        self._planner = DiscoursePlanner()

    def verbalize(
        self,
        features: ForecastFeatures,
        attribution: Optional[AttributionResult] = None,
    ) -> VerbalizationResult:
        """Convert features + attribution into a natural-language summary.

        Parameters
        ----------
        features : ForecastFeatures
            Features extracted from a quantile forecast.
        attribution : AttributionResult, optional
            SHAP covariate attributions.

        Returns
        -------
        VerbalizationResult
            Summary, individual sentences, grounding, and RST relations.
        """
        rng = random.Random(self.seed)
        sentences: List[str] = []
        grounding: Dict[str, Any] = {}
        rst_used: List[str] = []

        # ── Sentence 1: Trend ─────────────────────────────────────
        direction_phrase = rng.choice(
            _DIRECTION_PHRASES[features.trend_direction]
        )

        if features.trend_direction == "flat":
            trend_sentence = (
                f"The forecast indicates that values are "
                f"{direction_phrase} over the next "
                f"{features.horizon} periods."
            )
        else:
            magnitude_phrase = rng.choice(
                _MAGNITUDE_PHRASES[features.trend_magnitude]
            )
            trend_sentence = (
                f"The forecast indicates a {magnitude_phrase} "
                f"trend, with values {direction_phrase} over "
                f"the next {features.horizon} periods."
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

        # ── Sentence 2: Uncertainty ───────────────────────────────
        width_phrase = rng.choice(
            _UNCERTAINTY_LEVEL_PHRASES[features.uncertainty_level]
        )
        trend_phrase = rng.choice(
            _UNCERTAINTY_TREND_PHRASES[features.uncertainty_trend]
        )
        interp = _UNCERTAINTY_INTERPRETATION[features.uncertainty_level][
            features.uncertainty_trend
        ]

        uncertainty_sentence = (
            f"Prediction intervals are {width_phrase} and "
            f"{trend_phrase}, suggesting {interp}."
        )
        sentences.append(uncertainty_sentence)
        grounding["sentence_1"] = {
            "type": "uncertainty",
            "uncertainty_level": features.uncertainty_level,
            "uncertainty_trend": features.uncertainty_trend,
            "mean_interval_width": features.mean_interval_width,
            "relative_uncertainty": features.relative_uncertainty,
            "interval_width_slope": features.interval_width_slope,
        }

        # ── Sentence 3: Risk flags (conditional) ──────────────────
        risk_parts: List[str] = []
        risk_grounding: Dict[str, Any] = {
            "type": "risk",
            "downside_risk": features.downside_risk,
            "upside_potential": features.upside_potential,
            "regime_shift": features.regime_shift,
        }

        if features.downside_risk:
            risk_parts.append(
                "a risk of significant downside, with lower bounds "
                "exceeding 20% below current levels"
            )
        if features.upside_potential:
            risk_parts.append(
                "potential for significant upside, with upper bounds "
                "exceeding 20% above current levels"
            )

        if risk_parts:
            risk_sentence = "Notably, there is " + " and ".join(risk_parts) + "."
            sentences.append(risk_sentence)
            grounding[f"sentence_{len(sentences) - 1}"] = risk_grounding

        # ── RST-driven sentences ──────────────────────────────────
        rst_relations = self._planner.plan(features, attribution)
        for relation, kwargs in rst_relations:
            template = rng.choice(_RST_CONNECTORS[relation])
            rst_sentence = template.format(**kwargs)
            # Capitalize first letter
            rst_sentence = rst_sentence[0].upper() + rst_sentence[1:]
            sentences.append(rst_sentence)
            rst_used.append(relation.value)
            grounding[f"sentence_{len(sentences) - 1}"] = {
                "type": f"rst_{relation.value}",
                "relation": relation.value,
                **kwargs,
            }

        # ── Attribution sentences (if present) ────────────────────
        if attribution and attribution.attributions:
            top_attribs = attribution.attributions[: attribution.top_k]
            for attr in top_attribs:
                attr_sentence = (
                    f"{attr.name.replace('_', ' ').title()} has a "
                    f"{attr.direction} effect on the forecast, "
                    f"contributing {attr.relative_impact_pct:.1f}% "
                    f"of the total attribution."
                )
                sentences.append(attr_sentence)
                grounding[f"sentence_{len(sentences) - 1}"] = {
                    "type": "attribution",
                    "covariate_name": attr.name,
                    "shap_value": attr.shap_value,
                    "direction": attr.direction,
                    "relative_impact_pct": attr.relative_impact_pct,
                }

        summary = " ".join(sentences)
        return VerbalizationResult(
            summary=summary,
            sentences=sentences,
            grounding=grounding,
            rst_relations=rst_used,
        )


# ─────────────── Approach B: LLM-refined ──────────────────
class LLMVerbalizer:
    """LLM-refined verbalisation with grounding triples.

    Takes the template-based output and constructs a prompt asking an
    LLM to rewrite for fluency while preserving factual claims.
    Includes structured grounding triples and instructions to flag
    ungroundable sentences.

    Parameters
    ----------
    template_verbalizer : TemplateVerbalizer
        The primary verbaliser whose output will be refined.
    """

    def __init__(
        self,
        template_verbalizer: Optional[TemplateVerbalizer] = None,
        model_id: str = "google/gemma-4-E2B-it"
    ) -> None:
        self.template_verbalizer = template_verbalizer or TemplateVerbalizer()
        self.model_id = model_id
        self._processor = None
        self._model = None

    def _load_model(self) -> None:
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading LLM %s on %s ...", self.model_id, device)
            
            self._processor = AutoTokenizer.from_pretrained(
                self.model_id,
                extra_special_tokens={}
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id
            ).to(device)

    def verbalize(
        self,
        features: ForecastFeatures,
        attribution: Optional[AttributionResult] = None,
    ) -> VerbalizationResult:
        """Verbalize with LLM using draft template result."""
        self._load_model()
        
        # 1. Get draft
        template_result = self.template_verbalizer.verbalize(features, attribution)
        prompt = self.build_refinement_prompt(features, template_result, attribution)
        
        # 2. Process
        messages = [
            {
                "role": "system",
                "content": "You are an expert Data Scientist and Analyst who translates time-series forecasting metrics into clear, professional, executive-level summaries."
            },
            {"role": "user", "content": prompt},
        ]
        text = self._processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            enable_thinking=False
        )
        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        # 3. Generate
        outputs = self._model.generate(**inputs, max_new_tokens=1024)
        response = self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # 4. Parse
        if hasattr(self._processor, "parse_response"):
            try:
                parsed = self._processor.parse_response(response)
                if parsed:
                    if isinstance(parsed, dict):
                        response = parsed.get("content", str(parsed))
                    elif isinstance(parsed, str):
                        response = parsed
                    else:
                        response = str(parsed)
            except Exception:
                pass
                
        response = response.replace("<eos>", "").replace("<bos>", "").strip()
        
        # 5. Extract sentences and map to combined grounding
        sentences = [s.strip() + "." for s in response.split(".") if s.strip()]
        grounding = {}
        for i, _ in enumerate(sentences):
            grounding[f"sentence_{i}"] = {
                "type": "combined",
                "groundings": list(template_result.grounding.values())
            }
            
        return VerbalizationResult(
            summary=response,
            sentences=sentences,
            grounding=grounding,
            rst_relations=template_result.rst_relations
        )

    def build_grounding_triples(
        self,
        features: ForecastFeatures,
        attribution: Optional[AttributionResult] = None,
    ) -> List[Tuple[str, str, str]]:
        """Build structured (subject, predicate, object) triples.

        Parameters
        ----------
        features : ForecastFeatures
            Extracted features.
        attribution : AttributionResult, optional
            SHAP attributions.

        Returns
        -------
        list[tuple[str, str, str]]
            Grounding triples.
        """
        triples: List[Tuple[str, str, str]] = [
            ("P50_trend", "is", features.trend_direction),
            ("trend_magnitude", "is", features.trend_magnitude),
            ("P50_slope", "equals", f"{features.trend_slope:+.4f}"),
            ("prediction_interval", "is", features.uncertainty_level),
            ("interval_trend", "is", features.uncertainty_trend),
            ("interval_asymmetry", "is", features.asymmetry_label),
            ("downside_risk", "is", str(features.downside_risk).lower()),
            ("upside_potential", "is", str(features.upside_potential).lower()),
            ("regime_shift", "is", str(features.regime_shift).lower()),
        ]

        if attribution:
            for attr in attribution.attributions[: attribution.top_k]:
                triples.append((
                    f"{attr.name}_covariate",
                    f"{'increases' if attr.direction == 'positive' else 'decreases'}_forecast_by",
                    f"{attr.relative_impact_pct:.1f}%",
                ))

        return triples

    def build_refinement_prompt(
        self,
        features: ForecastFeatures,
        template_result: Optional[VerbalizationResult] = None,
        attribution: Optional[AttributionResult] = None,
    ) -> str:
        """Construct an LLM prompt for fluency refinement.

        Parameters
        ----------
        features : ForecastFeatures
            Raw extracted features for grounding.
        template_result : VerbalizationResult, optional
            If *None*, generates one via the template verbaliser first.
        attribution : AttributionResult, optional
            Covariate attributions for grounding triples.

        Returns
        -------
        str
            A ready-to-send prompt string.
        """
        if template_result is None:
            template_result = self.template_verbalizer.verbalize(
                features, attribution=attribution,
            )

        triples = self.build_grounding_triples(features, attribution)
        triples_str = "\n".join(
            f"  ({s}, {p}, {o})" for s, p, o in triples
        )

        features_str = "\n".join(
            f"  - {k}: {v}" for k, v in features.to_dict().items()
            if k != "threshold_breaches"
        )

        prompt = (
            "Please rewrite the following Draft Summary of a time-series forecast to sound more natural, fluent, and professional. "
            "The Draft Summary was currently generated from a template and might feel slightly rigid.\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "1. You MUST preserve all factual claims exactly as they appear in the draft (e.g., trend direction, uncertainty levels, exact percentages).\n"
            "2. Do NOT add any new information, assumptions, or external context that is not present in the Numerical Features.\n"
            "3. Ensure the tone is objective, analytical, and concise.\n"
            "4. Your final output should be ONLY the completely rewritten summary paragraph with no additional intro/outro text.\n\n"
            "### Numerical Features (For Context)\n"
            f"{features_str}\n\n"
            "### Structured Grounding Facts (Must be preserved)\n"
            f"{triples_str}\n\n"
            "### Draft Summary (To Rewrite)\n"
            f"{template_result.summary}\n"
        )
        return prompt
