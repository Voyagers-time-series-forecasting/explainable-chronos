"""LLM-as-judge for comparing forecast explanations from different pipeline configs.

Uses the same hardware-adaptive model selection as the LLM verbalizer:
  - CUDA available  → ``Qwen/Qwen2.5-7B-Instruct`` (fp16)
  - CPU only        → ``Qwen/Qwen1.5-1.8B-Chat``

Prompt technique: one worked few-shot example followed by a chain-of-thought
scaffold (Analyse A → Analyse B → Compare) so the model reasons before scoring.
JSON is extracted from the response with a regex fallback to handle partial outputs.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extension_1.config import select_llm_model
from extension_1.pipeline import PipelineResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an expert evaluator of time-series forecast explanations. "
    "You reason carefully before scoring and always output valid JSON. "
    "Your goal is to identify which explanation is more accurate, fluent, "
    "and clear about what drove the forecast."
)

# ---------------------------------------------------------------------------
# Few-shot example (one well-reasoned comparison)
# ---------------------------------------------------------------------------

_JUDGE_FEW_SHOT = """\
## Evaluation Example
FORECAST FACTS:
- Trend: rising (sharply), slope=+0.1842
- Uncertainty: moderate (stable)
- Interval asymmetry: right_skewed
- Downside risk: False, Upside potential: True
- Regime shift: False

EXPLANATION A [template]:
The P50 forecast is rising sharply. The prediction interval is moderate and stable. \
Upside potential is flagged.
NLI Consistency: 0.812

EXPLANATION B [llm]:
The median forecast shows a sharp upward trajectory with a slope of +0.1842 per step. \
The prediction interval is moderate and stable, reflecting consistent forecast confidence. \
The P90 quantile signals meaningful upside potential, with values potentially exceeding \
120% of the last observed level.
NLI Consistency: 0.851

Step 1 — Analyse Explanation A: States the core facts (rising trend, moderate interval, \
upside potential) but uses raw template phrasing ("is rising sharply", "is flagged") with \
no quantitative context.
Step 2 — Analyse Explanation B: Includes the specific slope value (+0.1842), explains what \
upside potential means numerically (>120% of last observed), and uses professional prose.
Step 3 — Compare: Both are factually accurate, but B provides more quantitative context \
and is clearly more fluent and informative.
OUTPUT: {"scores_a": {"factual_accuracy": 4, "fluency": 2, "attribution_clarity": 3}, \
"scores_b": {"factual_accuracy": 5, "fluency": 5, "attribution_clarity": 4}, \
"winner": "B", "reasoning": "B includes the specific slope value and quantitative upside \
context, making it more informative and professional than A."}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CriterionScore:
    """Per-explanation scores on three dimensions (1–5 each)."""

    factual_accuracy: int   # Does the text match the numerical features?
    fluency: int            # Is it natural and professional prose?
    attribution_clarity: int  # Does it explain what drove the forecast?

    @property
    def total(self) -> int:
        return self.factual_accuracy + self.fluency + self.attribution_clarity

    def to_dict(self) -> dict[str, int]:
        return {
            "factual_accuracy": self.factual_accuracy,
            "fluency": self.fluency,
            "attribution_clarity": self.attribution_clarity,
            "total": self.total,
        }


@dataclass
class JudgeVerdict:
    """Result of a pairwise explanation comparison."""

    config_a: str
    config_b: str
    scores_a: CriterionScore
    scores_b: CriterionScore
    winner: str  # "A", "B", or "tie"
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_a": self.config_a,
            "config_b": self.config_b,
            "scores_a": self.scores_a.to_dict(),
            "scores_b": self.scores_b.to_dict(),
            "winner": self.winner,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Compare forecast explanations using a local causal LLM.

    Parameters
    ----------
    model_id : str, optional
        HuggingFace model ID. Defaults to the hardware-appropriate model
        from :func:`~extension_1.config.select_llm_model`.
    model : Any, optional
        Pre-loaded model instance (reuse from ``LLMVerbalizer`` to avoid
        loading the same weights twice).
    tokenizer : Any, optional
        Pre-loaded tokenizer instance.
    """

    def __init__(
        self,
        model_id: str | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_id = model_id or select_llm_model()
        self._model = model
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    @classmethod
    def from_verbalizer(cls, verbalizer: Any) -> LLMJudge:
        """Construct a judge reusing an already-loaded ``LLMVerbalizer`` model.

        Avoids loading the LLM weights a second time when the judge and
        verbalizer share the same model family.
        """
        verbalizer._load_model()
        return cls(
            model_id=verbalizer.model_id,
            model=verbalizer._model,
            tokenizer=verbalizer._processor,
        )

    def _load_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.info("Loading judge LLM %s on %s (dtype=%s) …", self.model_id, device, dtype)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, extra_special_tokens={}
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=dtype
            ).to(device)
            self._model.eval()

    def _build_prompt(
        self,
        result_a: PipelineResult,
        config_a: str,
        result_b: PipelineResult,
        config_b: str,
    ) -> str:
        f = result_a.features
        facts = (
            f"- Trend: {f.trend_direction} ({f.trend_magnitude}), slope={f.trend_slope:+.4f}\n"
            f"- Uncertainty: {f.uncertainty_level} ({f.uncertainty_trend})\n"
            f"- Interval asymmetry: {f.asymmetry_label}\n"
            f"- Downside risk: {f.downside_risk}, Upside potential: {f.upside_potential}\n"
            f"- Regime shift detected: {f.regime_shift}"
        )
        nli_a = result_a.consistency_report.overall_score
        nli_b = result_b.consistency_report.overall_score

        return (
            "Evaluate two forecast explanations. "
            "Follow the three-step reasoning scaffold, then output the JSON scores.\n\n"
            + _JUDGE_FEW_SHOT
            + "\n---\n\n"
            "## Now evaluate:\n"
            f"FORECAST FACTS:\n{facts}\n\n"
            f"EXPLANATION A [{config_a}]:\n{result_a.verbalization.summary}\n"
            f"NLI Consistency: {nli_a:.3f}\n\n"
            f"EXPLANATION B [{config_b}]:\n{result_b.verbalization.summary}\n"
            f"NLI Consistency: {nli_b:.3f}\n\n"
            "Step 1 — Analyse Explanation A:\n"
            "Step 2 — Analyse Explanation B:\n"
            "Step 3 — Compare and decide:\n"
            "OUTPUT: "
            '{"scores_a": {"factual_accuracy": N, "fluency": N, "attribution_clarity": N}, '
            '"scores_b": {"factual_accuracy": N, "fluency": N, "attribution_clarity": N}, '
            '"winner": "A" or "B" or "tie", "reasoning": "one sentence"}'
        )

    def _parse_verdict(
        self, response: str, config_a: str, config_b: str
    ) -> JudgeVerdict:
        """Extract JSON from the model response; fall back to a neutral tie on failure."""
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                sa = data["scores_a"]
                sb = data["scores_b"]
                winner = str(data.get("winner", "tie")).strip('"').strip("'")
                if winner not in ("A", "B", "tie"):
                    winner = "tie"
                return JudgeVerdict(
                    config_a=config_a,
                    config_b=config_b,
                    scores_a=CriterionScore(
                        factual_accuracy=int(sa["factual_accuracy"]),
                        fluency=int(sa["fluency"]),
                        attribution_clarity=int(sa["attribution_clarity"]),
                    ),
                    scores_b=CriterionScore(
                        factual_accuracy=int(sb["factual_accuracy"]),
                        fluency=int(sb["fluency"]),
                        attribution_clarity=int(sb["attribution_clarity"]),
                    ),
                    winner=winner,
                    reasoning=str(data.get("reasoning", "")),
                )
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse judge JSON (%s); defaulting to tie.", exc)

        fallback = CriterionScore(factual_accuracy=3, fluency=3, attribution_clarity=3)
        return JudgeVerdict(
            config_a=config_a,
            config_b=config_b,
            scores_a=fallback,
            scores_b=fallback,
            winner="tie",
            reasoning="parse_error",
        )

    def compare(
        self,
        result_a: PipelineResult,
        config_a: str,
        result_b: PipelineResult,
        config_b: str,
    ) -> JudgeVerdict:
        """Compare two pipeline results for the same time-series window."""
        self._load_model()

        prompt = self._build_prompt(result_a, config_a, result_b, config_b)
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(text=text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,        # chain-of-thought needs room to reason
                do_sample=True,
                temperature=0.2,           # low temperature → consistent scoring
                repetition_penalty=1.1,
            )

        response = self._tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        response = response.replace("<eos>", "").replace("<bos>", "").strip()
        logger.debug("Judge raw response: %s", response)
        return self._parse_verdict(response, config_a, config_b)

    def compare_all(
        self, results: dict[str, PipelineResult]
    ) -> list[JudgeVerdict]:
        """Run all N*(N-1)/2 pairwise comparisons across the given configs."""
        keys = list(results.keys())
        verdicts: list[JudgeVerdict] = []
        for key_a, key_b in combinations(keys, 2):
            verdict = self.compare(results[key_a], key_a, results[key_b], key_b)
            verdicts.append(verdict)
            logger.info(
                "Judge: %s vs %s → winner=%s (%s)",
                key_a, key_b, verdict.winner, verdict.reasoning,
            )
        return verdicts
