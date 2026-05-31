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
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extension_1.config import select_llm_model
from extension_1.verbalization.types import VerbalizationResult
from extension_1.attribution.types import AttributionResult
from extension_1.features.extractor import ForecastFeatures
from extension_1.verbalization.trajectory import verbalize_temporal_focus, verbalize_trajectory
from extension_1.verbalization.template import TemplateVerbalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_ANALYST = (
    "You are a quantitative analyst writing concise, professional forecast summaries. "
    "You receive a draft description of a time-series forecast and rewrite it for clarity "
    "and fluency while preserving every numerical value and factual claim exactly as stated.\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. Keep your rewrite strictly under 4-5 sentences. Be extremely concise.\n"
    "2. DO NOT add interpretations, causal claims, or hallucinate relationships.\n"
    "3. NEVER change the direction (positive/negative) or percentages of covariate impacts."
)

_SYSTEM_PROMPT_EXECUTIVE = (
    "You are a communication specialist writing plain-language forecast summaries for "
    "non-technical decision-makers who have no statistics background.\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. Avoid jargon: replace 'P10/P50/P90' with 'lower/central/upper estimate', "
    "'prediction interval' with 'forecast range', 'regime shift' with 'notable change in pattern'.\n"
    "2. Keep every numerical value exactly as given — do not round or approximate percentages.\n"
    "3. Maximum 3-4 sentences. Focus on what matters for a business decision.\n"
    "4. Do NOT add causal claims or speculation not present in the draft."
)

# ---------------------------------------------------------------------------
# Few-shot examples (cover three representative scenarios)
# ---------------------------------------------------------------------------

_FEW_SHOT_ANALYST = """\
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

_FEW_SHOT_EXECUTIVE = """\
## Example 1
DRAFT: The P50 forecast is rising sharply. The prediction interval is moderate and stable. \
Upside potential is flagged. Temperature has a positive effect on the forecast, contributing \
45.2% of the total attribution.
REWRITE: Values are expected to rise strongly over the coming period, with a real chance of an \
even higher outcome. The forecast is moderately confident. Temperature is the main factor \
driving this increase, responsible for 45.2% of what shapes this forecast.

## Example 2
DRAFT: The P50 forecast is falling moderately. The prediction interval is high and widening. \
Downside risk is flagged. A regime shift was detected.
REWRITE: Values are expected to gradually decline, and there is considerable uncertainty — the \
actual outcome could be significantly worse than the central estimate. The model also detected \
a notable change in behaviour midway through the forecast period, which adds further caution.

## Example 3
DRAFT: The P50 forecast is flat. The prediction interval is low and stable. Wind has a negative \
effect on the forecast, contributing 38.1% of the total attribution.
REWRITE: Little change is expected over the coming period, and the model is quite confident in \
this stability. Wind is the main factor holding values back, accounting for 38.1% of the total \
driver influence.
"""

# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = _SYSTEM_PROMPT_ANALYST
_FEW_SHOT_BLOCK = _FEW_SHOT_ANALYST


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
        persona: Literal["analyst", "executive"] = "analyst",
    ) -> None:
        self.template_verbalizer = template_verbalizer or TemplateVerbalizer()
        self.model_id = model_id or select_llm_model()
        self.persona = persona
        self._processor: Any = None
        self._model: Any = None
        self._lock = threading.Lock()

    def share_model_from(self, other: "LLMVerbalizer") -> None:
        """Reuse an already-loaded model and tokenizer from *other* (avoids double loading)."""
        self._model = other._model
        self._processor = other._processor

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

        sys_prompt = _SYSTEM_PROMPT_EXECUTIVE if self.persona == "executive" else _SYSTEM_PROMPT_ANALYST
        messages = [
            {"role": "system", "content": sys_prompt},
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

        # Split on periods followed by space/end of string to avoid splitting decimals!
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
            draft_summary=template_result.summary,
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
                direction = getattr(attr, "direction", "positive")
                triples.append((
                    f"{attr.name}_covariate",
                    f"has_{direction}_effect_contributing",
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
            if k not in ("threshold_breaches", "trajectory", "trajectory_sentence")
        )
        triples_str = "; ".join(
            f"{s} {p} {o}"
            for s, p, o in self.build_grounding_triples(features, attribution)
        )

        traj_line = ""
        if features.trajectory:
            traj_sentence, _ = verbalize_trajectory(features.trajectory)
            traj_line = f"TRAJECTORY: {traj_sentence}\n"

        temporal_line = ""
        if attribution and attribution.temporal:
            history_length = len(attribution.temporal[0].saliency)
            tpf_sentence, _ = verbalize_temporal_focus(attribution.temporal, history_length)
            if tpf_sentence:
                temporal_line = f"TEMPORAL_FOCUS: {tpf_sentence}\n"

        few_shot = _FEW_SHOT_EXECUTIVE if self.persona == "executive" else _FEW_SHOT_ANALYST
        persona_constraint = (
            "CONSTRAINT: Use plain, non-technical language suitable for a business audience. "
            "Keep every number exactly as stated. Maximum 3-4 sentences.\n"
            if self.persona == "executive"
            else "CONSTRAINT: Be extremely concise. Do not add any extra commentary.\n"
        )
        return (
            "Rewrite the DRAFT below into fluent professional prose. "
            "Preserve every number and factual claim exactly. "
            "Do not add new facts or causal interpretations.\n\n"
            + few_shot
            + "\n---\n\n"
            "## Now rewrite the following\n"
            + persona_constraint
            + f"NUMERICAL FACTS: {facts}\n"
            f"GROUNDING: {triples_str}\n"
            f"{traj_line}"
            f"{temporal_line}"
            f"\nDRAFT: {template_result.summary}\n"
            "REWRITE:"
        )
