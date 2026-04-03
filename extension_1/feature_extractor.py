"""
Module 2 — Feature Extractor.

Derives interpretable numerical features from raw quantile forecasts
(P10 / P50 / P90).  All computations use only **numpy** and **scipy** —
no ML, no black boxes.

The resulting ``ForecastFeatures`` dataclass is the semantic bridge
consumed by the verbaliser.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats  # type: ignore

from config import PipelineConfig

logger = logging.getLogger(__name__)


# ───────────────────── dataclass ──────────────────────────────────────
@dataclass
class ForecastFeatures:
    """Interpretable features extracted from a quantile forecast.

    Attributes
    ----------
    trend_direction : str
        ``"rising"`` | ``"falling"`` | ``"flat"``.
    trend_magnitude : str
        ``"sharply"`` | ``"moderately"`` | ``"slightly"``.
    trend_slope : float
        Raw linear slope of P50 (per step).
    normalized_slope : float
        Slope divided by ``mean(|P50|)``.
    uncertainty_level : str
        ``"high"`` | ``"moderate"`` | ``"low"``.
    uncertainty_trend : str
        ``"widening"`` | ``"narrowing"`` | ``"stable"``.
    mean_interval_width : float
        ``mean(P90 - P10)``.
    relative_uncertainty : float
        ``mean_interval_width / mean(|P50|)``.
    interval_width_slope : float
        Linear slope of ``(P90 - P10)`` over forecast horizon.
    interval_asymmetry : float
        ``mean((P90-P50) - (P50-P10)) / mean(P90-P10)``.
    asymmetry_label : str
        ``"symmetric"`` | ``"right_skewed"`` | ``"left_skewed"``.
    downside_risk : bool
        ``min(P10) < last_observed * downside_factor``.
    upside_potential : bool
        ``max(P90) > last_observed * upside_factor``.
    regime_shift : bool
        Welch t-test detects a significant mean shift at the midpoint.
    regime_shift_pvalue : float
        p-value from Welch's t-test (1.0 when not applicable).
    threshold_breaches : list[dict]
        Each entry: ``{"name", "threshold", "quantile", "value", "step"}``.
    horizon : int
        Number of forecast steps.
    last_observed : float
        Last value of the historical tail.
    """

    trend_direction: str
    trend_magnitude: str
    trend_slope: float
    normalized_slope: float
    uncertainty_level: str
    uncertainty_trend: str
    mean_interval_width: float
    relative_uncertainty: float
    interval_width_slope: float
    interval_asymmetry: float
    asymmetry_label: str
    downside_risk: bool
    upside_potential: bool
    regime_shift: bool
    regime_shift_pvalue: float
    threshold_breaches: List[Dict[str, Any]]
    horizon: int
    last_observed: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialise features to a plain dictionary."""
        return asdict(self)


# ───────────────── extraction function ────────────────────────────────
def extract_features(
    forecast: Dict[str, Any],
    config: Optional[PipelineConfig] = None,
) -> ForecastFeatures:
    """Compute interpretable features from a quantile forecast dict.

    Parameters
    ----------
    forecast : dict
        Must contain keys ``p10``, ``p50``, ``p90``, ``history_tail``,
        and ``timestamps``.
    config : PipelineConfig, optional
        Threshold configuration; uses defaults when *None*.

    Returns
    -------
    ForecastFeatures
        Populated feature dataclass.
    """
    cfg = config or PipelineConfig()

    p10 = np.asarray(forecast["p10"], dtype=np.float64)
    p50 = np.asarray(forecast["p50"], dtype=np.float64)
    p90 = np.asarray(forecast["p90"], dtype=np.float64)
    history_tail = np.asarray(forecast["history_tail"], dtype=np.float64)
    horizon = len(p50)
    last_observed = float(history_tail[-1])

    # ── 1. Trend direction & magnitude ────────────────────────────
    steps = np.arange(horizon, dtype=np.float64)
    slope, _ = np.polyfit(steps, p50, deg=1)
    mean_abs_p50 = float(np.mean(np.abs(p50)))
    norm_slope = slope / mean_abs_p50 if mean_abs_p50 > 1e-9 else 0.0

    abs_norm = abs(norm_slope)
    if abs_norm > cfg.sharp_threshold:
        magnitude = "sharply"
    elif abs_norm > cfg.moderate_threshold:
        magnitude = "moderately"
    else:
        magnitude = "slightly"

    if abs_norm <= cfg.moderate_threshold:
        direction = "flat"
    elif norm_slope > 0:
        direction = "rising"
    else:
        direction = "falling"

    # ── 2. Uncertainty profile ────────────────────────────────────
    widths = p90 - p10
    mean_width = float(np.mean(widths))
    rel_unc = mean_width / mean_abs_p50 if mean_abs_p50 > 1e-9 else 0.0

    if rel_unc > cfg.high_uncertainty:
        unc_level = "high"
    elif rel_unc > cfg.low_uncertainty:
        unc_level = "moderate"
    else:
        unc_level = "low"

    width_slope, _ = np.polyfit(steps, widths, deg=1)
    if width_slope > cfg.widening_threshold:
        unc_trend = "widening"
    elif width_slope < cfg.narrowing_threshold:
        unc_trend = "narrowing"
    else:
        unc_trend = "stable"

    # ── 3. Interval asymmetry ─────────────────────────────────────
    upper = p90 - p50
    lower = p50 - p10
    mean_total = float(np.mean(widths))
    if mean_total > 1e-9:
        asymmetry = float(np.mean(upper - lower)) / mean_total
    else:
        asymmetry = 0.0

    if abs(asymmetry) < cfg.asymmetry_threshold:
        asym_label = "symmetric"
    elif asymmetry > 0:
        asym_label = "right_skewed"
    else:
        asym_label = "left_skewed"

    # ── 4. Tail risk flags ────────────────────────────────────────
    downside = bool(np.min(p10) < last_observed * cfg.downside_factor)
    upside = bool(np.max(p90) > last_observed * cfg.upside_factor)

    # ── 5. Regime-shift detection ─────────────────────────────────
    mid = horizon // 2
    if mid >= 2:
        first_half = p50[:mid]
        second_half = p50[mid:]
        t_stat, p_val = stats.ttest_ind(first_half, second_half, equal_var=False)
        regime = bool(p_val < cfg.regime_pvalue)
        regime_pval = float(p_val)
    else:
        regime = False
        regime_pval = 1.0

    # ── 6. Domain-specific threshold breaches ─────────────────────
    breaches: List[Dict[str, Any]] = []
    quantile_map = {"p10": p10, "p50": p50, "p90": p90}
    for name, threshold in cfg.critical_thresholds.items():
        for q_name, q_arr in quantile_map.items():
            for step_idx, val in enumerate(q_arr):
                if name.startswith("max") and val > threshold:
                    breaches.append({
                        "name": name,
                        "threshold": threshold,
                        "quantile": q_name,
                        "value": float(val),
                        "step": step_idx,
                    })
                elif name.startswith("min") and val < threshold:
                    breaches.append({
                        "name": name,
                        "threshold": threshold,
                        "quantile": q_name,
                        "value": float(val),
                        "step": step_idx,
                    })

    features = ForecastFeatures(
        trend_direction=direction,
        trend_magnitude=magnitude,
        trend_slope=float(slope),
        normalized_slope=float(norm_slope),
        uncertainty_level=unc_level,
        uncertainty_trend=unc_trend,
        mean_interval_width=mean_width,
        relative_uncertainty=rel_unc,
        interval_width_slope=float(width_slope),
        interval_asymmetry=asymmetry,
        asymmetry_label=asym_label,
        downside_risk=downside,
        upside_potential=upside,
        regime_shift=regime,
        regime_shift_pvalue=regime_pval,
        threshold_breaches=breaches,
        horizon=horizon,
        last_observed=last_observed,
    )
    logger.debug("Extracted features: %s", features)
    return features
