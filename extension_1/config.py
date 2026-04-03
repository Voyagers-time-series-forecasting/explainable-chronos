"""
Configuration for Extension 1 — Post-Hoc Forecast Narration with Covariate Attribution.

Centralises all thresholds, model names, seeds, and tuneable parameters
so that every module imports from a single source of truth.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ──────────────────────────────── seeds ────────────────────────────────
RANDOM_SEED: int = 42

# ──────────────────────────── forecast defaults ───────────────────────
DEFAULT_HORIZON: int = 14
HISTORY_TAIL_LENGTH: int = 5

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

# ────────────────────── evaluation settings ───────────────────────────
EVAL_NUM_SCENARIOS: int = 50

# ────────────────────── SHAP / surrogate ──────────────────────────────
SHAP_TOP_K: int = 5  # number of top attributions to surface in narration
SURROGATE_N_ESTIMATORS: int = 100


@dataclass
class PipelineConfig:
    """Aggregated runtime configuration passed through the pipeline.

    Parameters
    ----------
    seed : int
        Global random seed.
    horizon : int
        Number of forecast steps.
    sharp_threshold : float
        Normalised-slope threshold for "sharply" classification.
    moderate_threshold : float
        Normalised-slope threshold for "moderately" classification.
    high_uncertainty : float
        Relative-uncertainty threshold for "high".
    low_uncertainty : float
        Relative-uncertainty threshold for "low".
    widening_threshold : float
        Slope threshold for interval-width trend → "widening".
    narrowing_threshold : float
        Slope threshold for interval-width trend → "narrowing".
    downside_factor : float
        Factor applied to last observed value for downside risk flag.
    upside_factor : float
        Factor applied to last observed value for upside potential flag.
    regime_pvalue : float
        Welch t-test p-value threshold for regime shift.
    asymmetry_threshold : float
        Threshold for interval asymmetry label ("symmetric" vs skewed).
    critical_thresholds : dict
        Domain-specific critical thresholds (e.g. ``{"max_energy_kw": 500.0}``).
    nli_model : str
        HuggingFace model identifier for NLI scoring.
    consistency_threshold : float
        Minimum mean entailment score to declare consistency.
    shap_top_k : int
        Number of top SHAP attributions to show in narration.
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
    critical_thresholds: Dict[str, float] = field(default_factory=dict)
    nli_model: str = NLI_MODEL_NAME
    consistency_threshold: float = CONSISTENCY_THRESHOLD
    shap_top_k: int = SHAP_TOP_K
