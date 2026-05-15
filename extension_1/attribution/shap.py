"""SHAP-based covariate attribution using a surrogate XGBoost model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import shap  # type: ignore
import xgboost as xgb  # type: ignore
from sklearn.metrics import r2_score  # type: ignore

from extension_1.config import SHAP_TOP_K, SURROGATE_N_ESTIMATORS
from extension_1.attribution.types import (
    AttributionResult,
    CovariateAttribution,
    CovariateSet,
)

logger = logging.getLogger(__name__)


class SurrogateExplainer:
    """Surrogate XGBoost model + SHAP attribution.

    Uses rolling windows over the covariate history to create a training
    set with natural variation.  For each window the target is the *mean*
    of the P50 forecast, and features are per-covariate aggregates (mean,
    std, last, min, max).

    Parameters
    ----------
    n_estimators : int
    random_state : int
    top_k : int
    window : int
        Rolling-window size for feature construction.
    """

    def __init__(
        self,
        n_estimators: int = SURROGATE_N_ESTIMATORS,
        random_state: int = 42,
        top_k: int = SHAP_TOP_K,
        window: int = 10,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.top_k = top_k
        self.window = window
        self._model: xgb.XGBRegressor | None = None
        self._r2: float = 0.0
        self._feature_names: list[str] = []

    @staticmethod
    def _window_features(
        values: np.ndarray,
        names: list[str],
        start: int,
        end: int,
    ) -> np.ndarray:
        """Return (mean, std, last, min, max) per covariate for values[start:end]."""
        window = values[start:end]
        feats: list[float] = []
        for c in range(window.shape[1]):
            col = window[:, c]
            feats.extend([
                float(np.mean(col)),
                float(np.std(col)),
                float(col[-1]),
                float(np.min(col)),
                float(np.max(col)),
            ])
        return np.array(feats)

    @staticmethod
    def _make_feature_names(names: list[str]) -> list[str]:
        result: list[str] = []
        for n in names:
            result.extend([f"{n}_mean", f"{n}_std", f"{n}_last", f"{n}_min", f"{n}_max"])
        return result

    def fit(self, covariates: CovariateSet, forecast_target: np.ndarray) -> None:
        """Train surrogate to approximate Chronos-2 P50.

        Parameters
        ----------
        covariates : CovariateSet
        forecast_target : np.ndarray
            P50 forecast of shape ``(horizon,)``.
        """
        T = covariates.values.shape[0]
        w = min(self.window, T)
        target_mean = float(np.mean(forecast_target))
        target_std = float(np.std(forecast_target))

        X_rows: list[np.ndarray] = []
        y_rows: list[float] = []

        for start in range(T - w + 1):
            row = self._window_features(covariates.values, covariates.names, start, start + w)
            X_rows.append(row)
            frac = start / max(T - w, 1)
            y_rows.append(target_mean + target_std * (frac - 0.5))

        final_row = self._window_features(covariates.values, covariates.names, T - w, T)
        for val in forecast_target:
            X_rows.append(final_row.copy())
            y_rows.append(float(val))

        X = np.array(X_rows)
        y = np.array(y_rows)
        self._feature_names = self._make_feature_names(covariates.names)

        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_state,
            verbosity=0,
        )
        model.fit(X, y)
        self._r2 = float(r2_score(y, model.predict(X)))
        self._model = model
        logger.debug("Surrogate R² = %.4f", self._r2)

    def explain(self, covariates: CovariateSet) -> AttributionResult:
        """Compute SHAP attributions per covariate using the final history window.

        Returns
        -------
        AttributionResult
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before explain()")

        T = covariates.values.shape[0]
        w = min(self.window, T)
        X_explain = self._window_features(
            covariates.values, covariates.names, T - w, T,
        ).reshape(1, -1)

        shap_arr = np.asarray(
            shap.TreeExplainer(self._model).shap_values(X_explain)
        ).ravel()

        features_per_cov = 5
        cov_shap: dict[str, float] = {
            name: float(np.sum(shap_arr[i * features_per_cov:(i + 1) * features_per_cov]))
            for i, name in enumerate(covariates.names)
        }

        total_abs = sum(abs(v) for v in cov_shap.values()) or 1.0
        attributions = sorted(
            [
                CovariateAttribution(
                    name=name,
                    importance_score=abs(sv),
                    direction="positive" if sv >= 0 else "negative",
                    relative_impact_pct=abs(sv) / total_abs * 100,
                )
                for name, sv in cov_shap.items()
            ],
            key=lambda a: a.importance_score,
            reverse=True,
        )

        return AttributionResult(
            attributions=attributions,
            surrogate_r2=self._r2,
            top_k=self.top_k,
        )
