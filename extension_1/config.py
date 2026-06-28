"""Configuration for Extension 1 — Post-Hoc Forecast Narration."""

from __future__ import annotations

from dataclasses import dataclass, field


RANDOM_SEED: int = 42

# Forecast defaults
DEFAULT_HORIZON: int = 14
HISTORY_TAIL_LENGTH: int = 5

EPSILON: float = 1e-9  # guard against division by zero

# Quantile levels
QUANTILE_LOW: float = 0.10   # P10
QUANTILE_MID: float = 0.50   # P50
QUANTILE_HIGH: float = 0.90  # P90

# Trend classification thresholds (normalised slope)
SHARP_THRESHOLD: float = 0.05
MODERATE_THRESHOLD: float = 0.02

# Uncertainty classification (relative interval width)
HIGH_UNCERTAINTY_THRESHOLD: float = 0.30
LOW_UNCERTAINTY_THRESHOLD: float = 0.10

# Uncertainty width-trend classification (interval width slope per step)
WIDENING_THRESHOLD: float = 0.01
NARROWING_THRESHOLD: float = -0.01

# Tail-risk flags
DOWNSIDE_RISK_FACTOR: float = 0.80   # min(P10) < last_obs * factor
UPSIDE_POTENTIAL_FACTOR: float = 1.20  # max(P90) > last_obs * factor

# Regime-shift detection (Welch t-test p-value threshold)
REGIME_SHIFT_PVALUE: float = 0.05

# Interval asymmetry: |asym| < this → classified as "symmetric"
ASYMMETRY_THRESHOLD: float = 0.10

# NLI consistency scorer
NLI_MODEL_NAME: str = "facebook/bart-large-mnli"
CONSISTENCY_THRESHOLD: float = 0.70

# Attribution
ATTRIBUTION_TOP_K: int = 5

# LLM model selection
LLM_MODEL_CPU: str = "Qwen/Qwen1.5-1.8B-Chat"
LLM_MODEL_CUDA: str = "Qwen/Qwen2.5-7B-Instruct"
FUSION_MODEL_NAME: str = "google/flan-t5-base"


def select_llm_model() -> str:
    """Return the appropriate LLM model ID based on available hardware."""
    import torch  # local import keeps config importable without torch
    return LLM_MODEL_CUDA if torch.cuda.is_available() else LLM_MODEL_CPU


@dataclass
class PipelineConfig:
    """Aggregated runtime configuration passed through the pipeline.

    All fields default to the module-level constants so callers only need
    to override what they change.
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
