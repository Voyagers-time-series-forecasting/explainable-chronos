"""Configuration for Extension 1 — Post-Hoc Forecast Narration."""

from __future__ import annotations

from dataclasses import dataclass, field


# ──────────────────────────────── seeds ────────────────────────────────
RANDOM_SEED: int = 42

# ──────────────────────────── forecast defaults ───────────────────────
DEFAULT_HORIZON: int = 14
HISTORY_TAIL_LENGTH: int = 5

# ──────────────── numerical stability ─────────────────────────────────
EPSILON: float = 1e-9  # guard against division by zero

# ──────────────── quantile levels ─────────────────────────────────────
QUANTILE_LOW: float = 0.10   # P10
QUANTILE_MID: float = 0.50   # P50
QUANTILE_HIGH: float = 0.90  # P90

# ──────────────────────── trend classification ────────────────────────
SHARP_THRESHOLD: float = 0.05
MODERATE_THRESHOLD: float = 0.02

# ──────────────────── uncertainty classification ──────────────────────
HIGH_UNCERTAINTY_THRESHOLD: float = 0.30
LOW_UNCERTAINTY_THRESHOLD: float = 0.10

# ──────────── uncertainty width-trend classification ──────────────────
WIDENING_THRESHOLD: float = 0.01
NARROWING_THRESHOLD: float = -0.01

# ─────────────────────── tail-risk thresholds ─────────────────────────
DOWNSIDE_RISK_FACTOR: float = 0.80   # min(P10) < last_obs * factor
UPSIDE_POTENTIAL_FACTOR: float = 1.20  # max(P90) > last_obs * factor

# ──────────────────── regime-shift detection ──────────────────────────
REGIME_SHIFT_PVALUE: float = 0.05

# ──────────────────── interval asymmetry ──────────────────────────────
ASYMMETRY_THRESHOLD: float = 0.10  # |asym| < this → "symmetric"

# ────────────────────── NLI consistency scorer ────────────────────────
NLI_MODEL_NAME: str = "facebook/bart-large-mnli"
CONSISTENCY_THRESHOLD: float = 0.70

# ────────────────────── QA faithfulness scorer ──────────────────────────
# Extractive QA model used to check factual slot coverage.
# roberta-base-squad2 is small (~500 MB), fast on CPU, and accurate.
QA_MODEL_NAME: str = "deepset/roberta-base-squad2"
QA_CORRECT_THRESHOLD: float = 0.60   # per-slot cosine similarity threshold
QA_FAITHFUL_THRESHOLD: float = 0.55  # mean coverage threshold

# Sentence-BERT model for semantic answer matching (QAFactEval-style).
# all-MiniLM-L6-v2 is ~80 MB, fast on CPU, strong on semantic similarity.
SBERT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# NLI score decomposition weights (entailment + neutral + contradiction = 1)
NLI_NEUTRAL_WEIGHT: float = 0.60
NLI_CONTRADICTION_WEIGHT: float = 0.40

# ────────────────────── attribution ───────────────────────────────────
ATTRIBUTION_TOP_K: int = 5

# ─────────────────────── LLM model selection ──────────────────────────
# CPU-only (Colab free tier, local dev): compact 1.8B model
LLM_MODEL_CPU: str = "Qwen/Qwen1.5-1.8B-Chat"
# CUDA GPU available: use the 7B Instruct model in fp16 (~14 GB VRAM)
LLM_MODEL_CUDA: str = "Qwen/Qwen2.5-7B-Instruct"


def select_llm_model() -> str:
    """Return the best available LLM model ID based on hardware.

    Colab T4 / better: ``Qwen2.5-7B-Instruct`` loaded in fp16.
    CPU-only: ``Qwen1.5-1.8B-Chat`` (compact, fits in RAM).
    """
    import torch  # local import — keeps config importable without torch
    return LLM_MODEL_CUDA if torch.cuda.is_available() else LLM_MODEL_CPU


@dataclass
class PipelineConfig:
    """Aggregated runtime configuration passed through the pipeline.

    All fields default to the module-level constants above so callers
    only need to override what they change.
    """

    seed: int = RANDOM_SEED
    horizon: int = DEFAULT_HORIZON
    sharp_threshold: float = SHARP_THRESHOLD
    moderate_threshold: float = MODERATE_THRESHOLD
    high_uncertainty: float = HIGH_UNCERTAINTY_THRESHOLD
    low_uncertainty: float = LOW_UNCERTAINTY_THRESHOLD
    widening_threshold: float = WIDENING_THRESHOLD
    narrowing_threshold: float = NARROWING_THRESHOLD
    downside_factor: float = DOWNSIDE_RISK_FACTOR
    upside_factor: float = UPSIDE_POTENTIAL_FACTOR
    regime_pvalue: float = REGIME_SHIFT_PVALUE
    asymmetry_threshold: float = ASYMMETRY_THRESHOLD
    critical_thresholds: dict[str, float] = field(default_factory=dict)
    nli_model: str = NLI_MODEL_NAME
    consistency_threshold: float = CONSISTENCY_THRESHOLD
    attribution_top_k: int = ATTRIBUTION_TOP_K
