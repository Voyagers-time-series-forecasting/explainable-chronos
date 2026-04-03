"""
Shared data generation utilities for generating synthetic time series
and covariates across extensions.
"""

from typing import List, Tuple
import numpy as np


def generate_scenarios(
    n: int = 50,
    seed: int = 42,
) -> List[Tuple[str, np.ndarray, np.ndarray, int]]:
    """Create *n* synthetic (label, history, future, seed) tuples.
    Models daily e-commerce unit sales.
    """
    rng = np.random.default_rng(seed)
    labels = ["trending", "volatile", "flat"]
    scenarios: List[Tuple[str, np.ndarray, np.ndarray, int]] = []

    for i in range(n):
        label = labels[i % len(labels)]
        per_seed = int(rng.integers(0, 2**31))
        sub_rng = np.random.default_rng(per_seed)

        horizon = 14  # 14 days
        # E-commerce baseline ~500 sales/day
        if label == "trending":
            drift = sub_rng.choice([-1, 1]) * sub_rng.uniform(2.0, 5.0)
            full = 500.0 + np.cumsum(
                drift + sub_rng.normal(0, 5.0, size=50 + horizon)
            )
        elif label == "volatile":
            full = 500.0 + np.cumsum(
                sub_rng.normal(0, 15.0, size=50 + horizon)
            )
        else:  # flat
            full = 500.0 + np.cumsum(
                sub_rng.normal(0, 2.0, size=50 + horizon)
            )

        history = np.clip(full[:50], 0, None)
        future = np.clip(full[50: 50 + horizon], 0, None)
        scenarios.append((label, history, future, per_seed))

    return scenarios


def generate_synthetic_covariates(
    history: np.ndarray,
    n_covariates: int = 10,
    seed: int = 42,
):
    """Generate 10 synthetic covariates for the E-commerce domain."""
    from extension_1.covariate_attribution import CovariateSet

    rng = np.random.default_rng(seed)
    T = len(history)
    values = np.zeros((T, 10))

    # 1. marketing_spend (strongly correlated, e.g. $10 spent per ~1 sale)
    values[:, 0] = history * 10.0 + rng.normal(100, 50, size=T)
    # 2. website_traffic (strongly correlated, e.g. 20 visits per 1 sale)
    values[:, 1] = history * 20.0 + rng.normal(0, 200, size=T)
    # 3. previous_day_sales (lagged)
    values[1:, 2] = history[:-1]
    values[0, 2] = history[0]
    # 4. competitor_promotion_index (anti-correlated 0 to 100)
    values[:, 3] = np.clip(100 - (history / 500.0) * 50 + rng.normal(0, 10, size=T), 0, 100)
    # 5. price_discount_percentage (correlated 0 to 50%)
    values[:, 4] = np.clip((history / 500.0) * 10 + rng.normal(0, 2, size=T), 0, 50)
    # 6. holiday_proximity (correlated cyclical 1-10)
    values[:, 5] = (np.sin(np.linspace(0, 4*np.pi, T)) + 1) * 4 + rng.normal(1, 0.5, size=T)
    # 7. shipping_delay_hours (anti-correlated 0 to 72)
    values[:, 6] = np.clip(72 - (history / 500.0) * 36 + rng.normal(0, 5, size=T), 0, 72)
    # 8. social_media_mentions (correlated scale)
    values[:, 7] = history * 2.5 + rng.normal(50, 20, size=T)
    # 9. weather_temperature (noise 30 to 90 F)
    values[:, 8] = rng.uniform(30, 90, size=T)
    # 10. random_sensor_noise (pure noise)
    values[:, 9] = rng.normal(0, 1, size=T)

    names = [
        "marketing_spend", "website_traffic", "previous_day_sales", 
        "competitor_promotion_index", "price_discount_percentage", 
        "holiday_proximity", "shipping_delay_hours", 
        "social_media_mentions", "weather_temperature", "random_sensor_noise"
    ]
    descriptions = {
        "marketing_spend": "Daily marketing spend in USD",
        "website_traffic": "Total unique daily website visitors",
        "previous_day_sales": "Unit sales from the previous day",
        "competitor_promotion_index": "Intensity of competitor discounts (0-100)",
        "price_discount_percentage": "Average discount applied to products (%)",
        "holiday_proximity": "Proximity to major shopping holidays (0-10 scale)",
        "shipping_delay_hours": "Average network shipping delay in hours",
        "social_media_mentions": "Total brand mentions across social platforms",
        "weather_temperature": "Average national daily temperature (F)",
        "random_sensor_noise": "Unrelated control metric noise",
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
