"""Lightweight fusion-based forecast verbalizer.

Uses a small instruction-tuned seq2seq model (FLAN-T5) to fuse the
template-generated sentences into fluent prose. The LLM verbalizer rewrites
the draft freely from a natural-language prompt, which gives it room to
paraphrase — and occasionally drift away from — the underlying facts. This
verbalizer instead performs sentence fusion only: it is shown the
already-grounded template sentences verbatim and asked to smooth the
transitions between them with discourse connectives, without rephrasing the
content inside each sentence. Because the task is this narrow (combine,
don't invent), a model an order of magnitude smaller than the LLM
verbalizer reaches comparable fluency with markedly fewer factual slips.

Model selection:
  - default → ``google/flan-t5-base`` (~250M params, runs comfortably on CPU)

Override by passing ``model_id`` explicitly.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from extension_1.config import FUSION_MODEL_NAME
from extension_1.verbalization.types import VerbalizationResult
from extension_1.attribution.types import AttributionResult
from extension_1.features.extractor import ForecastFeatures
from extension_1.verbalization.template import TemplateVerbalizer

logger = logging.getLogger(__name__)


_FUSION_INSTRUCTION = (
    "Combine the numbered sentences below into one fluent paragraph. "
    "Join them with natural connectives such as 'while', 'although', 'and', "
    "or 'meanwhile'. Do not add, remove, or change any number, name, or "
    "fact, and do not introduce any new claim. Keep the meaning of every "
    "sentence intact."
)

_FEW_SHOT_FUSION = """\
Sentences:
1. The forecast indicates a moderate trend, with values projected to grow over the next 14 periods.
2. Prediction intervals are narrow and stable, suggesting confidence that remains strong throughout the forecast.
3. Wind has a negative effect on the forecast, contributing 38.1% of the total attribution.
Paragraph: The forecast indicates a moderate upward trend over the next 14 periods, while prediction intervals stay narrow and stable, reflecting strong confidence throughout. Wind is the dominant negative driver, accounting for 38.1% of the total attribution.

Sentences:
1. The forecast indicates a sharp trend, with values expected to decrease over the next 96 periods.
2. Prediction intervals are wide and widening over the forecast horizon, suggesting rapidly growing uncertainty that warrants caution.
3. Notably, there is a risk of significant downside, with lower bounds exceeding 20% below current levels.
Paragraph: The forecast indicates a sharp downward trend over the next 96 periods, and prediction intervals are wide and widening, reflecting rapidly growing uncertainty that warrants caution. Notably, there is a risk of significant downside, with lower bounds exceeding 20% below current levels.

Sentences:
1. The forecast indicates that values are projected to stay roughly flat over the next 30 periods.
2. A structural break is detected near the midpoint, indicating a potential change in the underlying pattern (p=0.0123).
Paragraph: The forecast indicates that values are projected to stay roughly flat over the next 30 periods. Notably, a structural break is detected near the midpoint, indicating a potential change in the underlying pattern (p=0.0123).
"""


class FusionVerbalizer:
    """Sentence-fusion verbalizer built on a small FLAN-T5 model.

    Reuses the deterministic, fully-grounded ``TemplateVerbalizer`` draft as
    input and asks a small seq2seq model only to fuse the sentences into
    fluent prose, never to regenerate the underlying facts. This keeps the
    factual-consistency guarantees of the template close to intact while
    reading far more naturally, at a fraction of the parameter count of the
    LLM verbalizer.
    """

    def __init__(
        self,
        template_verbalizer: TemplateVerbalizer | None = None,
        model_id: str | None = None,
    ) -> None:
        self.template_verbalizer = template_verbalizer or TemplateVerbalizer()
        self.model_id = model_id or FUSION_MODEL_NAME
        self._tokenizer: Any = None
        self._model: Any = None
        self._lock = threading.Lock()

    def share_model_from(self, other: "FusionVerbalizer") -> None:
        """Reuse an already-loaded model and tokenizer to avoid loading twice."""
        self._model = other._model
        self._tokenizer = other._tokenizer

    def _load_model(self) -> None:
        """Lazy, thread-safe model load."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.info("Loading fusion model %s on %s (dtype=%s) …", self.model_id, device, dtype)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(device)
            self._model.eval()

    def verbalize(
        self,
        features: ForecastFeatures,
        attribution: AttributionResult | None = None,
    ) -> VerbalizationResult:
        """Verbalize by fusing the template draft's sentences into fluent prose."""
        self._load_model()

        template_result = self.template_verbalizer.verbalize(features, attribution)
        prompt = self.build_fusion_prompt(template_result)

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._model.device)

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,  # greedy/beam search — maximises factual accuracy
            )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Split on periods followed by whitespace/end; avoids splitting decimal numbers.
        sentences = [s.strip() + "." for s in re.split(r"\.(?:\s+|$)", response) if s.strip()]
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
            prompt=prompt,
        )

    def build_fusion_prompt(self, template_result: VerbalizationResult) -> str:
        """Construct the few-shot fusion prompt from the template's sentences."""
        numbered = "\n".join(
            f"{i + 1}. {sentence}" for i, sentence in enumerate(template_result.sentences)
        )
        return (
            _FUSION_INSTRUCTION
            + "\n\n"
            + _FEW_SHOT_FUSION
            + "\n---\n\n"
            "Sentences:\n"
            f"{numbered}\n"
            "Paragraph:"
        )
