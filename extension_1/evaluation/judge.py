"""LLM-as-judge for comparing forecast explanations from different pipeline configs.

The evaluation question is explicitly framed around **human preference**:
"Would a practitioner (analyst, data scientist, business user) prefer this
explanation over the alternative?"

Each explanation receives a single ``human_preference`` score (1–5):
  5 — strongly preferred: clear, useful, trustworthy
  4 — preferred
  3 — acceptable / neutral
  2 — some reservations
  1 — would not find it useful

The winner is derived from those scores (higher wins; tie if |score_a − score_b| ≤ 1).

Uses the same hardware-adaptive model selection as the LLM verbalizer.
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
    "You are simulating the preference of an expert practitioner — an analyst, "
    "data scientist, or business user — reading a time-series forecast explanation. "
    "Your task: judge which explanation a real human expert would prefer to read and act on. "
    "Prefer explanations that are clear, concise, informative, and trustworthy. "
    "Penalise verbosity, vagueness, hallucinated facts, and missing key information. "
    "You reason step-by-step before scoring and always output valid JSON."
)

# ---------------------------------------------------------------------------
# Few-shot example
# ---------------------------------------------------------------------------

_JUDGE_FEW_SHOT = """\
## Evaluation Example
FORECAST FACTS:
- Trend: rising (sharply), slope=+0.1842
- Uncertainty: moderate (stable)
- Downside risk: False  |  Upside potential: True
- Top covariate: temperature (positive, 45.2% contribution)

EXPLANATION A [template]:
The forecast indicates a sharply rising trend over the next 96 periods. \
Prediction intervals are moderately wide and stable. \
Temperature has a positive effect on the forecast, contributing 45.2% of the total attribution.
NLI: 0.812  |  Fact recall: 0.67  |  Completeness: 0.80

EXPLANATION B [llm]:
The median forecast shows a sharp upward trajectory over the next 96 periods. \
The prediction interval is moderate and stable, reflecting consistent forecast confidence. \
Temperature is the dominant positive driver, accounting for 45.2% of the total attribution. \
The P90 quantile signals meaningful upside potential.
NLI: 0.851  |  Fact recall: 1.00  |  Completeness: 1.00

Step 1 — Analyse A: Covers the main facts in plain template language. Acceptable, \
but the phrasing is mechanical and the upside potential flag is missing.
Step 2 — Analyse B: Same facts, more natural prose, includes upside potential and \
explicitly ties the quantile to the flag — a practitioner immediately knows what to watch.
Step 3 — Which would a human prefer? B is more complete and reads as a professional \
summary. A practitioner would prefer B.
OUTPUT: {"score_a": 3, "score_b": 5, "winner": "B", \
"reasoning": "B is more complete and professional; A omits the upside potential flag."}
"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CriterionScore:
    """Human-preference score for one explanation (1–5)."""

    human_preference: int

    @property
    def total(self) -> int:
        return self.human_preference

    def to_dict(self) -> dict[str, int]:
        return {"human_preference": self.human_preference}


@dataclass
class JudgeVerdict:
    """Result of a pairwise explanation comparison."""

    config_a: str
    config_b: str
    scores_a: CriterionScore
    scores_b: CriterionScore
    winner: str   # "A", "B", or "tie"
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
        """Construct a judge reusing an already-loaded ``LLMVerbalizer`` model."""
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
            logger.info("Loading judge LLM %s on %s …", self.model_id, device)
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
        from extension_1.evaluation.factuality import (
            compute_fact_recall,
            compute_feature_completeness,
        )

        f = result_a.features
        attr_line = ""
        if result_a.attribution and result_a.attribution.attributions:
            top = result_a.attribution.attributions[0]
            attr_line = f"\n- Top covariate: {top.name} ({top.relative_impact_pct:.1f}% contribution)"

        facts = (
            f"- Trend: {f.trend_direction} ({f.trend_magnitude}), slope={f.trend_slope:+.4f}\n"
            f"- Uncertainty: {f.uncertainty_level} ({f.uncertainty_trend})\n"
            f"- Downside risk: {f.downside_risk}  |  Upside potential: {f.upside_potential}"
            f"{attr_line}"
        )

        def _metrics(res: PipelineResult) -> str:
            nli = res.consistency_report.overall_score
            fr  = compute_fact_recall(res.features, res.attribution, res.verbalization.summary)
            fc  = compute_feature_completeness(res.features, res.attribution, res.verbalization.summary)
            return f"NLI: {nli:.3f}  |  Fact recall: {fr:.2f}  |  Completeness: {fc:.2f}"

        return (
            "Evaluate the two explanations below. "
            "For each, give a human_preference score 1–5 "
            "(5 = a practitioner would strongly prefer this). "
            "Derive the winner from whichever score is higher "
            "(tie if |score_a − score_b| ≤ 1).\n\n"
            + _JUDGE_FEW_SHOT
            + "\n---\n\n"
            "## Now evaluate:\n"
            f"FORECAST FACTS:\n{facts}\n\n"
            f"EXPLANATION A [{config_a}]:\n{result_a.verbalization.summary}\n"
            f"{_metrics(result_a)}\n\n"
            f"EXPLANATION B [{config_b}]:\n{result_b.verbalization.summary}\n"
            f"{_metrics(result_b)}\n\n"
            "Step 1 — Analyse A:\n"
            "Step 2 — Analyse B:\n"
            "Step 3 — Which would a human prefer?\n"
            'OUTPUT: {"score_a": N, "score_b": N, "winner": "A" or "B" or "tie", "reasoning": "one sentence"}'
        )

    def _parse_verdict(
        self, response: str, config_a: str, config_b: str
    ) -> JudgeVerdict:
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                sa = int(data.get("score_a", 3))
                sb = int(data.get("score_b", 3))
                winner = str(data.get("winner", "tie")).strip('"').strip("'")
                if winner not in ("A", "B", "tie"):
                    # Derive from scores when the model ignores the field
                    if abs(sa - sb) <= 1:
                        winner = "tie"
                    else:
                        winner = "A" if sa > sb else "B"
                return JudgeVerdict(
                    config_a=config_a,
                    config_b=config_b,
                    scores_a=CriterionScore(human_preference=max(1, min(5, sa))),
                    scores_b=CriterionScore(human_preference=max(1, min(5, sb))),
                    winner=winner,
                    reasoning=str(data.get("reasoning", "")),
                )
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse judge JSON (%s); defaulting to tie.", exc)

        fallback = CriterionScore(human_preference=3)
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
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
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
                "Judge: %s (pref=%d) vs %s (pref=%d) -> winner=%s",
                key_a, verdict.scores_a.human_preference,
                key_b, verdict.scores_b.human_preference,
                verdict.winner,
            )
        return verdicts
