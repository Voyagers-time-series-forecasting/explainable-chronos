"""
Shared data generation utilities for generating synthetic time series
and covariates across extensions.

Series are diversified across:
- History length: 30, 50, 100, 200 points
- Forecast horizon: 7, 14, 30 steps
- Scale: low (~10), medium (~500), high (~10000)
- Type: trending, volatile, flat, seasonal, noisy
"""

from typing import List, Tuple
import numpy as np


# ── diversity pools ────────────────────────────────────────────────────
_HISTORY_LENGTHS = [30, 50, 100, 200]
_HORIZONS        = [7, 14, 30]
_BASELINES       = [10.0, 100.0, 500.0, 5000.0]
_LABELS          = ["trending", "volatile", "flat", "seasonal", "noisy"]


def _make_series(
    label: str,
    history_length: int,
    horizon: int,
    baseline: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single (history, future) pair for the given label.

    Parameters
    ----------
    label : str
        One of trending / volatile / flat / seasonal / noisy.
    history_length : int
        Number of historical time steps.
    horizon : int
        Number of future time steps to generate as ground truth.
    baseline : float
        Base level of the series (scale).
    rng : np.random.Generator
        Per-scenario random generator for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        history array of shape (history_length,) and
        future array of shape (horizon,).
    """
    total = history_length + horizon
    noise_scale = baseline * 0.02  # 2% of baseline as base noise

    if label == "trending":
        drift = rng.choice([-1, 1]) * rng.uniform(0.5, 2.0) * (baseline / 500.0)
        increments = drift + rng.normal(0, noise_scale, size=total)
        full = baseline + np.cumsum(increments)

    elif label == "volatile":
        increments = rng.normal(0, noise_scale * 5, size=total)
        full = baseline + np.cumsum(increments)

    elif label == "flat":
        increments = rng.normal(0, noise_scale * 0.5, size=total)
        full = baseline + np.cumsum(increments)

    elif label == "seasonal":
        # Weekly seasonality (period=7) plus small trend
        t = np.arange(total)
        period = 7
        amplitude = baseline * 0.15
        seasonality = amplitude * np.sin(2 * np.pi * t / period)
        drift = rng.uniform(-0.3, 0.3) * (baseline / 500.0)
        noise = rng.normal(0, noise_scale, size=total)
        full = baseline + seasonality + drift * t + noise

    else:  # noisy
        # High noise, no clear structure
        increments = rng.normal(0, noise_scale * 8, size=total)
        full = baseline + np.cumsum(increments)

    full = np.clip(full, 0, None)
    history = full[:history_length]
    future  = full[history_length: history_length + horizon]
    return history, future


def generate_scenarios(
    n: int = 50,
    seed: int = 42,
) -> List[Tuple[str, np.ndarray, np.ndarray, int]]:
    """Create *n* diverse synthetic (label, history, future, seed) tuples.

    Each scenario is independently randomised across:
    - label       : trending / volatile / flat / seasonal / noisy
    - history_len : 30 / 50 / 100 / 200
    - horizon     : 7 / 14 / 30
    - baseline    : 10 / 100 / 500 / 5000

    Parameters
    ----------
    n : int
        Number of scenarios to generate.
    seed : int
        Master random seed for reproducibility.

    Returns
    -------
    list of (label, history, future, per_seed)
    """
    rng = np.random.default_rng(seed)
    scenarios: List[Tuple[str, np.ndarray, np.ndarray, int]] = []

    for i in range(n):
        # Each scenario gets its own seed for full reproducibility
        per_seed = int(rng.integers(0, 2**31))
        sub_rng  = np.random.default_rng(per_seed)

        # Sample all parameters independently
        label          = _LABELS[i % len(_LABELS)]
        history_length = int(sub_rng.choice(_HISTORY_LENGTHS))
        horizon        = int(sub_rng.choice(_HORIZONS))
        baseline       = float(sub_rng.choice(_BASELINES))

        history, future = _make_series(
            label, history_length, horizon, baseline, sub_rng,
        )
        scenarios.append((label, history, future, per_seed))

    return scenarios


def generate_synthetic_covariates(
    history: np.ndarray,
    n_covariates: int = 10,
    seed: int = 42,
):
    """Generate 10 synthetic covariates aligned with the history length.

    Covariates are scaled relative to the history's mean so they work
    correctly regardless of the series baseline (low/medium/high scale).

    Parameters
    ----------
    history : np.ndarray
        Historical time series of any length and scale.
    n_covariates : int
        Number of covariates to generate (currently fixed at 10).
    seed : int
        Random seed.

    Returns
    -------
    CovariateSet
    """
    from extension_1.covariate_attribution import CovariateSet

    rng = np.random.default_rng(seed)
    T   = len(history)

    # Scale-aware noise: proportional to the mean of the series
    mean_level = float(np.mean(np.abs(history))) + 1e-9
    values = np.zeros((T, 10))

    # 1. marketing_spend — strongly correlated
    values[:, 0] = history * 10.0 + rng.normal(mean_level, mean_level * 0.1, size=T)
    # 2. website_traffic — strongly correlated
    values[:, 1] = history * 20.0 + rng.normal(0, mean_level * 0.4, size=T)
    # 3. previous_day_sales — lagged target (1 step)
    values[1:, 2] = history[:-1]
    values[0,  2] = history[0]
    # 4. competitor_promotion_index — anti-correlated (0–100)
    values[:, 3] = np.clip(
        100 - (history / (mean_level + 1e-9)) * 50 + rng.normal(0, 10, size=T),
        0, 100,
    )
    # 5. price_discount_percentage — correlated (0–50%)
    values[:, 4] = np.clip(
        (history / (mean_level + 1e-9)) * 10 + rng.normal(0, 2, size=T),
        0, 50,
    )
    # 6. holiday_proximity — cyclical, period scales with T
    period = max(7, T // 4)
    values[:, 5] = (
        np.sin(np.linspace(0, 4 * np.pi, T)) + 1
    ) * 4 + rng.normal(1, 0.5, size=T)
    # 7. shipping_delay_hours — anti-correlated (0–72 h)
    values[:, 6] = np.clip(
        72 - (history / (mean_level + 1e-9)) * 36 + rng.normal(0, 5, size=T),
        0, 72,
    )
    # 8. social_media_mentions — correlated
    values[:, 7] = history * 2.5 + rng.normal(mean_level * 0.1, mean_level * 0.04, size=T)
    # 9. weather_temperature — pure noise (30–90 °F), scale-independent
    values[:, 8] = rng.uniform(30, 90, size=T)
    # 10. random_sensor_noise — pure noise control variable
    values[:, 9] = rng.normal(0, 1, size=T)

    names = [
        "marketing_spend",
        "website_traffic",
        "previous_day_sales",
        "competitor_promotion_index",
        "price_discount_percentage",
        "holiday_proximity",
        "shipping_delay_hours",
        "social_media_mentions",
        "weather_temperature",
        "random_sensor_noise",
    ]
    descriptions = {
        "marketing_spend":            "Daily marketing spend in USD",
        "website_traffic":            "Total unique daily website visitors",
        "previous_day_sales":         "Unit sales from the previous day",
        "competitor_promotion_index": "Intensity of competitor discounts (0–100)",
        "price_discount_percentage":  "Average discount applied to products (%)",
        "holiday_proximity":          "Proximity to major shopping holidays (0–10 scale)",
        "shipping_delay_hours":       "Average network shipping delay in hours",
        "social_media_mentions":      "Total brand mentions across social platforms",
        "weather_temperature":        "Average national daily temperature (°F)",
        "random_sensor_noise":        "Unrelated control metric — pure noise",
    }

    return CovariateSet(
        names=names,
        values=values,
        descriptions=descriptions,
    )


def generate_demo_time_series(seed: int = 42, length: int = 30) -> np.ndarray:
    """Generate a single dummy time series for demo use."""
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0, 0.5, size=length))