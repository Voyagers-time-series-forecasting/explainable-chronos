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

    Returns
    -------
    list[tuple[str, np.ndarray, np.ndarray, int]]
        ``(scenario_label, history_array, future_array, per_seed)``.
    """
    rng = np.random.default_rng(seed)
    labels = ["trending", "volatile", "flat"]
    scenarios: List[Tuple[str, np.ndarray, np.ndarray, int]] = []

    for i in range(n):
        label = labels[i % len(labels)]
        per_seed = int(rng.integers(0, 2**31))
        sub_rng = np.random.default_rng(per_seed)

        horizon = 14  # default
        if label == "trending":
            drift = sub_rng.choice([-1, 1]) * sub_rng.uniform(0.3, 1.0)
            full = 100.0 + np.cumsum(
                drift + sub_rng.normal(0, 0.3, size=50 + horizon)
            )
        elif label == "volatile":
            full = 100.0 + np.cumsum(
                sub_rng.normal(0, 2.0, size=50 + horizon)
            )
        else:  # flat
            full = 100.0 + np.cumsum(
                sub_rng.normal(0, 0.1, size=50 + horizon)
            )

        history = full[:50]
        future = full[50: 50 + horizon]
        scenarios.append((label, history, future, per_seed))

    return scenarios


def generate_synthetic_covariates(
    history: np.ndarray,
    n_covariates: int = 4,
    seed: int = 42,
):
    """Generate synthetic covariates, one correlated with the target.

    Parameters
    ----------
    history : np.ndarray
        Historical time-series.
    n_covariates : int
        Number of covariates.
    seed : int
        Random seed.

    Returns
    -------
    CovariateSet
        Synthetic covariates.
    """
    # Import CovariateSet here to avoid circular/hard dependency
    from extension_1.covariate_attribution import CovariateSet

    rng = np.random.default_rng(seed)
    T = len(history)
    values = np.zeros((T, n_covariates))

    # Covariate 0: correlated with the target (signal)
    values[:, 0] = history * 0.8 + rng.normal(0, 5, size=T)

    # Covariate 1: lagged version
    values[1:, 1] = history[:-1] * 0.5 + rng.normal(0, 3, size=T - 1)
    values[0, 1] = values[1, 1]

    # Covariate 2: pure noise
    values[:, 2] = rng.normal(50, 10, size=T)

    # Covariate 3: weak anti-correlation
    if n_covariates > 3:
        values[:, 3] = -history * 0.2 + rng.normal(0, 8, size=T)

    names = ["correlated_signal", "lagged_signal", "noise", "anti_correlated"][:n_covariates]
    descriptions = {
        "correlated_signal": "Strongly correlated with target",
        "lagged_signal": "One-step lagged version of target",
        "noise": "Pure random noise (control)",
        "anti_correlated": "Weakly anti-correlated with target",
    }

    return CovariateSet(
        names=names,
        values=values,
        descriptions={k: descriptions[k] for k in names},
    )


def generate_demo_time_series(seed: int = 42, length: int = 30) -> np.ndarray:
    """Generate a single dummy time series for demo use."""
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0, 0.5, size=length))
