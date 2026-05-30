"""LLM-refined forecast verbalizer.

Uses a causal LM (Qwen family) to rewrite template-generated summaries into
fluent professional prose while preserving every numerical fact.

Model selection (automatic):
  - CUDA available  → ``Qwen/Qwen2.5-7B-Instruct`` loaded in fp16
  - CPU only        → ``Qwen/Qwen1.5-1.8B-Chat``

Override by passing ``model_id`` explicitly.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extension_1.config import select_llm_model
from extension_1.verbalization.types import VerbalizationResult
from extension_1.attribution.types import AttributionResult
from extension_1.features.extractor import ForecastFeatures
from extension_1.verbalization.template import TemplateVerbalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a quantitative analyst writing informative, professional forecast summaries. "
    "You receive structured numerical facts about a time-series forecast and write a "
    "clear, explanatory summary directly from those facts.\n\n"
    "ENTAILMENT RULE (critical):\n"
    "Every sentence you write will be verified by an NLI model that checks whether "
    "the numerical facts ENTAIL your sentence. To maximise entailment:\n"
    "  - Each sentence must be fully and directly supported by the stated facts.\n"
    "  - Use the exact direction words (rising/falling/flat), magnitude words "
    "(sharply/moderately/slightly), and numerical values given in the facts.\n"
    "  - Do NOT add causal interpretations, domain knowledge, or any claim "
    "not directly stated in the facts. Such additions cannot be entailed and "
    "will be flagged as contradictions.\n"
    "  - Do NOT hedge with 'may', 'might', 'could' unless the fact itself states uncertainty.\n\n"
    "ATTRIBUTION RULE:\n"
    "When covariate facts are provided, mention ALL of them — not just the top one. "
    "For each covariate state its direction (positive/negative) and exact impact percentage. "
    "Explain the relative distribution: which dominates, which are secondary, "
    "and whether they all push in the same direction or oppose each other. "
    "This is the most important part of the explanation for the user.\n\n"
    "CONSTRAINTS:\n"
    "  - 4 to 6 sentences. Prioritise being informative over being brief.\n"
    "  - Never change the sign or magnitude of a covariate impact percentage."
)

# ---------------------------------------------------------------------------
# Few-shot examples: FACTS → SUMMARY (no draft)
# ---------------------------------------------------------------------------

_FEW_SHOT_BLOCK = """\
## Example 1  [lead with covariate breakdown, then trend, then uncertainty, then risk]
FACTS:
  trend_direction: rising | trend_magnitude: sharply | trend_slope: +0.1842 | horizon: 96
  uncertainty_level: moderate | uncertainty_trend: stable
  downside_risk: false | upside_potential: true
  covariate: temperature | direction: positive | impact_pct: 45.2
  covariate: humidity | direction: positive | impact_pct: 31.7
  covariate: wind_speed | direction: negative | impact_pct: 23.1
  attribution_summary: temperature (45.2% pos), humidity (31.7% pos), wind_speed (23.1% neg)
SUMMARY: The covariate attribution is dominated by temperature (45.2% positive influence) and \
humidity (31.7% positive), both pushing the forecast upward, while wind_speed partially offsets \
these with a negative contribution of 23.1%. \
Over the next 96 periods the median forecast rises sharply, with a slope of +0.1842 per period. \
Prediction intervals are moderately wide and remain stable throughout the horizon. \
The P90 quantile signals meaningful upside potential.

## Example 2  [lead with uncertainty/risk, then trend, then full attribution breakdown]
FACTS:
  trend_direction: falling | trend_magnitude: moderately | trend_slope: -0.0934 | horizon: 96
  uncertainty_level: high | uncertainty_trend: widening
  downside_risk: true | upside_potential: false
  regime_shift: true | regime_shift_pvalue: 0.0031
  covariate: pressure | direction: negative | impact_pct: 52.4
  covariate: cloud_cover | direction: negative | impact_pct: 30.1
  covariate: dew_point | direction: positive | impact_pct: 17.5
  attribution_summary: pressure (52.4% neg), cloud_cover (30.1% neg), dew_point (17.5% pos)
SUMMARY: Prediction intervals are wide and expanding over the 96-period horizon, reflecting \
rapidly growing uncertainty. \
The P10 quantile flags significant downside risk, with lower bounds exceeding 20% below current levels. \
The median forecast falls moderately over the 96 periods (slope: −0.0934 per period), and a \
statistically significant structural break is detected near the midpoint (p = 0.0031). \
On the attribution side, pressure is the dominant negative driver at 52.4%, reinforced by \
cloud_cover (30.1% negative); dew_point partially counteracts these with a positive contribution of 17.5%.

## Example 3  [compact trend+uncertainty, then detailed attribution]
FACTS:
  trend_direction: flat | trend_magnitude: slightly | trend_slope: +0.0021 | horizon: 96
  uncertainty_level: low | uncertainty_trend: narrowing
  downside_risk: false | upside_potential: false
  covariate: wind | direction: negative | impact_pct: 38.1
  covariate: temp_min | direction: positive | impact_pct: 35.6
  covariate: precipitation | direction: positive | impact_pct: 26.3
  attribution_summary: wind (38.1% neg), temp_min (35.6% pos), precipitation (26.3% pos)
SUMMARY: Over the next 96 periods, values are projected to remain essentially flat \
(slope: +0.0021), with narrow and converging prediction intervals indicating high forecast confidence. \
The attribution is closely distributed across three covariates: wind exerts the largest negative \
influence at 38.1%, while temp_min (35.6%) and precipitation (26.3%) both push positively, \
almost balancing wind's effect.
"""


class LLMVerbalizer:
    """LLM-refined verbalization with few-shot prompting.

    Takes the template-based output and uses an LLM to rewrite it for
    fluency while strictly preserving factual claims. Two or three
    worked examples are included in every prompt.

    Parameters
    ----------
    template_verbalizer : TemplateVerbalizer, optional
    model_id : str, optional
        HuggingFace model ID. Defaults to the hardware-appropriate model
        selected by :func:`~extension_1.config.select_llm_model`.
    """

    def __init__(
        self,
        template_verbalizer: TemplateVerbalizer | None = None,
        model_id: str | None = None,
    ) -> None:
        self.template_verbalizer = template_verbalizer or TemplateVerbalizer()
        self.model_id = model_id or select_llm_model()
        self._processor: Any = None
        self._model: Any = None
        self._lock = threading.Lock()

    def _load_model(self) -> None:
        """Load model on first call (thread-safe lazy initialisation)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.info("Loading LLM %s on %s (dtype=%s) …", self.model_id, device, dtype)
            self._processor = AutoTokenizer.from_pretrained(
                self.model_id, extra_special_tokens={}
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(device)
            self._model.eval()

    def verbalize(
        self,
        features: ForecastFeatures,
        attribution: AttributionResult | None = None,
    ) -> VerbalizationResult:
        """Generate a summary directly from numerical features (no template draft).

        The template verbalizer is still used internally to obtain a grounding
        dictionary for the NLI scorer, but its prose is NOT fed to the LLM.
        """
        self._load_model()

        # Grounding metadata for NLI — derived from template, not shown to LLM.
        template_result = self.template_verbalizer.verbalize(features, attribution)

        prompt = self.build_refinement_prompt(features, attribution)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,          # greedy — maximises factual fidelity
                repetition_penalty=1.15,
            )

        response = self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        response = response.replace("<eos>", "").replace("<bos>", "").strip()

        # Strip "SUMMARY:" prefix the model may echo back.
        for prefix in ("SUMMARY:", "REWRITE:"):
            if response.upper().startswith(prefix):
                response = response[len(prefix):].strip()
                break

        sentences = [s.strip() + "." for s in re.split(r'\.(?:\s+|$)', response) if s.strip()]
        grounding = {
            f"sentence_{i}": {
                "type": "combined",
                "groundings": list(template_result.grounding.values()),
            }
            for i in range(len(sentences))
        }

        return VerbalizationResult(
            summary=response,
            sentences=sentences,
            grounding=grounding,
            rst_relations=list(getattr(template_result, "rst_relations", [])),
            draft_summary=template_result.summary,   # kept for traceability
            prompt=text,
        )

    def build_grounding_triples(
        self,
        features: ForecastFeatures,
        attribution: AttributionResult | None = None,
    ) -> list[tuple[str, str, str]]:
        """Build structured (subject, predicate, object) triples for context."""
        triples: list[tuple[str, str, str]] = [
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
                verb = "increases" if attr.direction == "positive" else "decreases"
                triples.append((
                    f"{attr.name}_covariate",
                    f"{verb}_forecast_by",
                    f"{attr.relative_impact_pct:.1f}%",
                ))
        return triples

    def build_refinement_prompt(
        self,
        features: ForecastFeatures,
        attribution: AttributionResult | None = None,
    ) -> str:
        """Construct the few-shot FACTS→SUMMARY prompt (no template draft).

        The prompt gives the LLM only the raw numerical facts and grounding
        triples. The template prose is intentionally withheld so the LLM
        generates an independent verbalization.
        """
        # ── Compact fact block (one line per feature group) ──────────────
        f = features
        fact_lines = [
            f"  trend_direction: {f.trend_direction} | "
            f"trend_magnitude: {f.trend_magnitude} | "
            f"trend_slope: {f.trend_slope:+.4f} | "
            f"horizon: {f.horizon}",

            f"  uncertainty_level: {f.uncertainty_level} | "
            f"uncertainty_trend: {f.uncertainty_trend} | "
            f"mean_interval_width: {f.mean_interval_width:.4f}",

            f"  downside_risk: {str(f.downside_risk).lower()} | "
            f"upside_potential: {str(f.upside_potential).lower()}",

            f"  regime_shift: {str(f.regime_shift).lower()}"
            + (f" | pvalue: {f.regime_shift_pvalue:.4f}" if f.regime_shift else ""),
        ]

        if attribution and attribution.attributions:
            attrs = attribution.attributions[: attribution.top_k]
            for attr in attrs:
                fact_lines.append(
                    f"  covariate: {attr.name} | "
                    f"direction: {attr.direction} | "
                    f"impact_pct: {attr.relative_impact_pct:.1f}"
                )
            # Pre-formatted attribution summary so the model sees the full distribution at a glance
            summary_parts = ", ".join(
                f"{a.name} ({a.relative_impact_pct:.1f}% {a.direction})"
                for a in attrs
            )
            fact_lines.append(f"  attribution_summary: {summary_parts}")

        facts_str = "\n".join(fact_lines)
        triples_str = "; ".join(
            f"{s} {p} {o}"
            for s, p, o in self.build_grounding_triples(features, attribution)
        )

        return (
            "Write a forecast summary from the structured facts below. "
            "Each sentence must be directly and fully entailed by the stated facts — "
            "use the exact direction, magnitude, and numerical values given.\n\n"
            + _FEW_SHOT_BLOCK
            + "\n---\n\n"
            "## Now write a summary for:\n"
            f"FACTS:\n{facts_str}\n"
            f"GROUNDING TRIPLES: {triples_str}\n"
            "SUMMARY:"
        )
