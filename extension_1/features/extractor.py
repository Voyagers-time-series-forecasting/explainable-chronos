"""
Feature Extractor — derives interpretable numerical features from raw
quantile forecasts (P10 / P50 / P90) using only numpy and scipy.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats  # type: ignore

from extension_1.config import (
    EPSILON,
    PipelineConfig,
)

logger = logging.getLogger(__name__)


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
    threshold_breaches: list[dict[str, Any]]
    horizon: int
    last_observed: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise features to a plain dictionary."""
        return asdict(self)


def extract_features(
    forecast: dict[str, Any],
    config: PipelineConfig | None = None,
) -> ForecastFeatures:
    """Compute interpretable features from a quantile forecast dict.

    Parameters
    ----------
    forecast : dict
        Must contain keys ``p10``, ``p50``, ``p90``, ``history_tail``.
    config : PipelineConfig, optional

    Returns
    -------
    ForecastFeatures
    """
    cfg = config or PipelineConfig()

    p10 = np.asarray(forecast["p10"], dtype=np.float64)
    p50 = np.asarray(forecast["p50"], dtype=np.float64)
    p90 = np.asarray(forecast["p90"], dtype=np.float64)
    history_tail = np.asarray(forecast["history_tail"], dtype=np.float64)
    horizon = len(p50)
    last_observed = float(history_tail[-1])

    # ── 1. Trend ──────────────────────────────────────────────────
    steps = np.arange(horizon, dtype=np.float64)
    slope, _ = np.polyfit(steps, p50, deg=1)
    mean_abs_p50 = float(np.mean(np.abs(p50)))
    norm_slope = slope / mean_abs_p50 if mean_abs_p50 > EPSILON else 0.0
    abs_norm = abs(norm_slope)

    magnitude = (
        "sharply" if abs_norm > cfg.sharp_threshold
        else "moderately" if abs_norm > cfg.moderate_threshold
        else "slightly"
    )
    direction = (
        "flat" if abs_norm <= cfg.moderate_threshold
        else "rising" if norm_slope > 0
        else "falling"
    )

    # ── 2. Uncertainty ────────────────────────────────────────────
    widths = p90 - p10
    mean_width = float(np.mean(widths))
    rel_unc = mean_width / mean_abs_p50 if mean_abs_p50 > EPSILON else 0.0

    unc_level = (
        "high" if rel_unc > cfg.high_uncertainty
        else "moderate" if rel_unc > cfg.low_uncertainty
        else "low"
    )

    width_slope, _ = np.polyfit(steps, widths, deg=1)
    unc_trend = (
        "widening" if width_slope > cfg.widening_threshold
        else "narrowing" if width_slope < cfg.narrowing_threshold
        else "stable"
    )

    # ── 3. Interval asymmetry ─────────────────────────────────────
    mean_total = float(np.mean(widths))
    asymmetry = (
        float(np.mean((p90 - p50) - (p50 - p10))) / mean_total
        if mean_total > EPSILON
        else 0.0
    )
    asym_label = (
        "symmetric" if abs(asymmetry) < cfg.asymmetry_threshold
        else "right_skewed" if asymmetry > 0
        else "left_skewed"
    )

    # ── 4. Tail risk flags ────────────────────────────────────────
    downside = bool(np.min(p10) < last_observed * cfg.downside_factor)
    upside = bool(np.max(p90) > last_observed * cfg.upside_factor)

    # ── 5. Regime-shift detection ─────────────────────────────────
    mid = horizon // 2
    if mid >= 2:
        _, p_val = stats.ttest_ind(p50[:mid], p50[mid:], equal_var=False)
        regime = bool(p_val < cfg.regime_pvalue)
        regime_pval = float(p_val)
    else:
        regime, regime_pval = False, 1.0

    # ── 6. Domain-specific threshold breaches ─────────────────────
    breaches: list[dict[str, Any]] = []
    quantile_map = {"p10": p10, "p50": p50, "p90": p90}
    for name, threshold in cfg.critical_thresholds.items():
        for q_name, q_arr in quantile_map.items():
            for step_idx, val in enumerate(q_arr):
                if (name.startswith("max") and val > threshold) or (
                    name.startswith("min") and val < threshold
                ):
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
