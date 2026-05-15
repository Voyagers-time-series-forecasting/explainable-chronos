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
    "You are a quantitative analyst writing concise, professional forecast summaries. "
    "You receive a draft description of a time-series forecast and rewrite it for clarity "
    "and fluency while preserving every numerical value and factual claim exactly as stated. "
    "Do not add interpretations, causal claims, or information not present in the draft."
)

# ---------------------------------------------------------------------------
# Few-shot examples (cover three representative scenarios)
# ---------------------------------------------------------------------------

_FEW_SHOT_BLOCK = """\
## Example 1
DRAFT: The P50 forecast is rising sharply. The prediction interval is moderate and stable. \
Upside potential is flagged. Temperature has a positive effect on the forecast, contributing \
45.2% of the total attribution.
REWRITE: The median forecast shows a sharp upward trajectory over the horizon. The prediction \
interval is moderate and stable, reflecting consistent forecast confidence. The P90 quantile \
signals meaningful upside potential. Temperature is the dominant positive driver, accounting \
for 45.2% of the total covariate attribution.

## Example 2
DRAFT: The P50 forecast is falling moderately. The prediction interval is high and widening. \
Downside risk is flagged. A regime shift was detected.
REWRITE: The median forecast shows a moderate downward trend. The prediction interval is wide \
and expanding, indicating growing uncertainty over the horizon. The P10 quantile flags downside \
risk, with values potentially falling below 80% of the last observed level. A statistically \
significant regime shift was detected at the forecast midpoint, suggesting a structural change \
in the series dynamics.

## Example 3
DRAFT: The P50 forecast is flat. The prediction interval is low and stable. Wind has a negative \
effect on the forecast, contributing 38.1% of the total attribution.
REWRITE: The median forecast remains flat over the forecast horizon. The prediction interval is \
narrow and stable, indicating high forecast confidence. Wind is the dominant negative driver, \
accounting for 38.1% of the total covariate attribution.
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
        """Verbalize with LLM refinement over a template draft."""
        self._load_model()

        template_result = self.template_verbalizer.verbalize(features, attribution)
        prompt = self.build_refinement_prompt(features, template_result, attribution)

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
                do_sample=False,          # greedy — maximises factual accuracy
                repetition_penalty=1.15,  # discourage verbatim repetition of template
            )

        response = self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        response = response.replace("<eos>", "").replace("<bos>", "").strip()

        # Strip "REWRITE:" prefix the model may echo back
        if response.upper().startswith("REWRITE:"):
            response = response[len("REWRITE:"):].strip()

        sentences = [s.strip() + "." for s in response.split(".") if s.strip()]
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
        template_result: VerbalizationResult | None = None,
        attribution: AttributionResult | None = None,
    ) -> str:
        """Construct the few-shot refinement prompt."""
        if template_result is None:
            template_result = self.template_verbalizer.verbalize(features, attribution)

        facts = "\n".join(
            f"  {k}: {v}"
            for k, v in features.to_dict().items()
            if k != "threshold_breaches"
        )
        triples_str = "; ".join(
            f"{s} {p} {o}"
            for s, p, o in self.build_grounding_triples(features, attribution)
        )

        return (
            "Rewrite the DRAFT below into fluent professional prose. "
            "Preserve every number and factual claim exactly. "
            "Do not add new facts or causal interpretations.\n\n"
            + _FEW_SHOT_BLOCK
            + "\n---\n\n"
            "## Now rewrite the following\n"
            f"NUMERICAL FACTS: {facts}\n"
            f"GROUNDING: {triples_str}\n\n"
            f"DRAFT: {template_result.summary}\n"
            "REWRITE:"
        )
