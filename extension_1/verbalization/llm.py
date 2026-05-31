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
import random
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
    "You are a data analyst explaining a forecast to someone with no statistics background — "
    "think of a business manager, a shop owner, or a curious non-expert. "
    "Your job is to turn technical forecast results into a clear, helpful explanation that anyone can understand.\n\n"
    "HOW TO WRITE:\n"
    "  - Use plain, everyday language. Avoid jargon like 'quantile', 'P10/P90', 'slope', or 'attribution'.\n"
    "  - When you mention a number, briefly say what it means in plain terms.\n"
    "    For example: instead of 'slope +0.18', say 'values are expected to grow a little each day'.\n"
    "  - If there is a key driver (covariate), explain its role simply, like: "
    "'The main thing pushing this up is temperature, which accounts for about 45% of the expected change'.\n"
    "  - If there is uncertainty, explain it simply: "
    "'We are reasonably confident about the direction, though the exact numbers could be higher or lower'.\n"
    "  - Keep sentences short and varied in structure — avoid repeating the same pattern.\n\n"
    "ACCURACY RULES:\n"
    "  - Never reverse the trend direction or change the sign of any factor's influence.\n"
    "  - Never invent numbers, causes, or outcomes not stated in the facts.\n"
    "  - Do not speculate about why the trend is happening — only describe what the model shows.\n\n"
    "LENGTH: Write 3 to 5 short, clear sentences."
)

# ---------------------------------------------------------------------------
# Style hints — injected randomly to vary discourse structure across calls
# ---------------------------------------------------------------------------

_STYLE_HINTS: list[str] = [
    "Start with the overall outlook (is it good news or bad news?), then add detail.",
    "Lead with the most important finding, then explain what it means for the reader.",
    "Begin by describing what is expected to happen, then mention how confident we are.",
    "If there is a key driver, mention it first and explain its role, then describe the trend.",
    "Open with the confidence level of the forecast, then describe what is expected.",
    "Tell a short story: what is happening, why (if a driver is given), and what to watch out for.",
]

# ---------------------------------------------------------------------------
# Few-shot examples: FACTS → SUMMARY (non-technical style)
# ---------------------------------------------------------------------------

_FEW_SHOT_BLOCK = """\
## Example 1
FACTS:
  trend_direction: rising | trend_magnitude: sharply | trend_slope: +0.1842 | horizon: 96
  uncertainty_level: moderate | uncertainty_trend: stable
  downside_risk: false | upside_potential: true
  covariate: temperature | direction: positive | impact_pct: 45.2
SUMMARY: Temperature is the biggest factor here, responsible for roughly 45% of the expected change — \
when temperatures rise, this forecast rises with it. \
Overall, values are expected to climb noticeably over the coming 96 periods. \
The forecast is reasonably confident: our range of possible outcomes is moderate and stays steady throughout. \
There is a real chance things could end up even higher than expected, so it is worth keeping an eye on the upside.

## Example 2
FACTS:
  trend_direction: falling | trend_magnitude: moderately | trend_slope: -0.0934 | horizon: 96
  uncertainty_level: high | uncertainty_trend: widening
  downside_risk: true | upside_potential: false
  regime_shift: true | regime_shift_pvalue: 0.0031
SUMMARY: The forecast points to a gradual decline over the next 96 periods, though the picture is uncertain — \
the range of possible outcomes is wide and gets wider the further out we look. \
There is a notable risk of things turning out worse than the central estimate, so caution is advised. \
The model also picked up a meaningful shift in behaviour around the middle of the horizon, \
suggesting the pattern may have changed, which adds to the uncertainty.

## Example 3
FACTS:
  trend_direction: flat | trend_magnitude: slightly | trend_slope: +0.0021 | horizon: 96
  uncertainty_level: low | uncertainty_trend: narrowing
  downside_risk: false | upside_potential: false
  covariate: wind | direction: negative | impact_pct: 38.1
SUMMARY: The forecast is fairly stable — values are expected to stay roughly the same over the next 96 periods. \
We are quite confident in this outlook: the range of possible outcomes is narrow and getting narrower over time. \
Wind is the main factor pulling things down, contributing about 38% of the total influence on this forecast.
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
                max_new_tokens=300,
                do_sample=True,
                temperature=0.75,
                top_p=0.92,
                repetition_penalty=1.1,
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
            for attr in attribution.attributions[: attribution.top_k]:
                fact_lines.append(
                    f"  covariate: {attr.name} | "
                    f"direction: {attr.direction} | "
                    f"impact_pct: {attr.relative_impact_pct:.1f}"
                )

        facts_str = "\n".join(fact_lines)
        triples_str = "; ".join(
            f"{s} {p} {o}"
            for s, p, o in self.build_grounding_triples(features, attribution)
        )

        style_hint = random.choice(_STYLE_HINTS)
        return (
            "Explain the following forecast results to a non-technical reader.\n"
            f"Style guidance: {style_hint}\n\n"
            + _FEW_SHOT_BLOCK
            + "\n---\n\n"
            "## Now write an explanation for:\n"
            f"FACTS:\n{facts_str}\n"
            f"KEY DRIVERS: {triples_str}\n"
            "EXPLANATION:"
        )
