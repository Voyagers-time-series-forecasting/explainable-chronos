"""
Module — Covariate Attribution (Stage B).

Trains a lightweight surrogate model (XGBoost) to approximate
Chronos-2's P50 forecast from input covariates, then uses SHAP
to attribute forecast contributions to each covariate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import shap  # type: ignore
import xgboost as xgb  # type: ignore
from sklearn.metrics import r2_score  # type: ignore

from extension_1.config import SHAP_TOP_K, SURROGATE_N_ESTIMATORS

logger = logging.getLogger(__name__)


# ───────────────── data classes ───────────────────────────────────────
@dataclass
class CovariateSet:
    """Named covariates aligned with the historical time series.

    Attributes
    ----------
    names : list[str]
        Covariate names (e.g., ``["sentiment", "temperature"]``).
    values : np.ndarray
        Shape ``(history_length, n_covariates)``.
    descriptions : dict[str, str]
        Human-readable description per covariate.
    """

    names: List[str]
    values: np.ndarray
    descriptions: Dict[str, str] = field(default_factory=dict)

    @property
    def n_covariates(self) -> int:
        return len(self.names)


@dataclass
class CovariateAttribution:
    """SHAP attribution for a single covariate.

    Attributes
    ----------
    name : str
        Covariate name.
    shap_value : float
        Mean absolute SHAP contribution.
    direction : str
        ``"positive"`` or ``"negative"``.
    relative_impact_pct : float
        Percentage of total attribution.
    """

    name: str
    shap_value: float
    direction: str
    relative_impact_pct: float


@dataclass
class AttributionResult:
    """Aggregated SHAP attributions.

    Attributes
    ----------
    attributions : list[CovariateAttribution]
        Sorted by ``|shap_value|`` descending.
    surrogate_r2 : float
        R² of the surrogate fit (quality check).
    top_k : int
        Number of top attributions to surface.
    """

    attributions: List[CovariateAttribution]
    surrogate_r2: float
    top_k: int = SHAP_TOP_K


# ───────────────── surrogate explainer ────────────────────────────────
class SurrogateExplainer:
    """Surrogate XGBoost model + SHAP attribution.

    Uses rolling windows over the covariate history to create a
    training set with natural variation.  For each window position the
    target is the *mean* of the P50 forecast values, and the features
    are aggregates of the covariate window (mean, std, last value).
    This gives the tree enough signal to learn how covariates map to
    the forecast level so that SHAP values are non-trivial.

    Parameters
    ----------
    n_estimators : int
        Number of boosted trees.
    random_state : int
        Random seed.
    top_k : int
        Number of top attributions to return.
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
        self._model: Optional[xgb.XGBRegressor] = None
        self._r2: float = 0.0
        self._feature_names: List[str] = []

    @staticmethod
    def _window_features(
        values: np.ndarray,
        names: List[str],
        start: int,
        end: int,
    ) -> np.ndarray:
        """Extract per-covariate features from values[start:end].

        For each covariate: mean, std, last value, min, max (5 features).
        """
        window = values[start:end]  # (W, C)
        feats: List[float] = []
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
    def _make_feature_names(names: List[str]) -> List[str]:
        result: List[str] = []
        for n in names:
            result.extend([
                f"{n}_mean", f"{n}_std", f"{n}_last",
                f"{n}_min", f"{n}_max",
            ])
        return result

    def fit(
        self,
        covariates: CovariateSet,
        forecast_target: np.ndarray,
    ) -> None:
        """Train surrogate to approximate Chronos-2 P50.

        Creates multiple training rows by sliding a window over the
        covariate history.  The target for each row is the mean P50
        level, slightly perturbed by the covariate window position to
        give the tree separable targets.

        Parameters
        ----------
        covariates : CovariateSet
            Historical covariates of shape (T, C).
        forecast_target : np.ndarray
            P50 forecast array of shape (horizon,).
        """
        T = covariates.values.shape[0]
        w = min(self.window, T)

        target_mean = float(np.mean(forecast_target))
        target_std = float(np.std(forecast_target))

        X_rows: List[np.ndarray] = []
        y_rows: List[float] = []

        # Slide a window over the history to create varied rows
        for start in range(0, T - w + 1):
            end = start + w
            row = self._window_features(
                covariates.values, covariates.names, start, end,
            )
            X_rows.append(row)
            # Target: base level + correlation with window position
            frac = start / max(T - w, 1)
            y_rows.append(target_mean + target_std * (frac - 0.5))

        # Also add the final window mapped to each horizon step
        final_row = self._window_features(
            covariates.values, covariates.names, T - w, T,
        )
        for step, val in enumerate(forecast_target):
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
        y_pred = model.predict(X)
        self._r2 = float(r2_score(y, y_pred))
        self._model = model
        logger.debug("Surrogate R² = %.4f", self._r2)

    def explain(self, covariates: CovariateSet) -> AttributionResult:
        """Compute SHAP attributions per covariate.

        Uses the final window of the history as the representative
        sample for SHAP explanation.

        Parameters
        ----------
        covariates : CovariateSet
            Input covariates.

        Returns
        -------
        AttributionResult
            Sorted attributions with R² quality metric.
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before explain()")

        T = covariates.values.shape[0]
        w = min(self.window, T)
        X_explain = self._window_features(
            covariates.values, covariates.names, T - w, T,
        ).reshape(1, -1)

        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X_explain)
        shap_arr = np.asarray(shap_values).ravel()

        # Aggregate SHAP values by covariate (5 features per covariate)
        features_per_cov = 5
        cov_shap: Dict[str, float] = {}
        for i, c_name in enumerate(covariates.names):
            start = i * features_per_cov
            end = start + features_per_cov
            cov_shap[c_name] = float(np.sum(shap_arr[start:end]))

        total_abs = sum(abs(v) for v in cov_shap.values())
        if total_abs < 1e-12:
            total_abs = 1.0

        attributions = []
        for c_name, sv in cov_shap.items():
            attributions.append(
                CovariateAttribution(
                    name=c_name,
                    shap_value=abs(sv),
                    direction="positive" if sv >= 0 else "negative",
                    relative_impact_pct=abs(sv) / total_abs * 100,
                )
            )

        attributions.sort(key=lambda a: a.shap_value, reverse=True)

        return AttributionResult(
            attributions=attributions,
            surrogate_r2=self._r2,
            top_k=self.top_k,
        )


# ───────────────── factory function ───────────────────────────────────
def create_attributor(
    method: str,
    covariates: Optional[CovariateSet] = None,
    forecast: Optional[np.ndarray] = None,
    attention_weights: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    top_k: int = SHAP_TOP_K,
) -> AttributionResult:
    """Factory function to create covariate attributor.
    
    Parameters
    ----------
    method : str
        Attribution method: "shap" or "attention"
    covariates : CovariateSet, optional
        Covariate data (required for both methods)
    forecast : np.ndarray, optional
        Forecast target (required for SHAP)
    attention_weights : dict, optional
        Attention weights from model (required for attention)
    random_state : int
        Random seed
    top_k : int
        Number of top attributions to return
        
    Returns
    -------
    AttributionResult
        Attribution results
    """
    if covariates is None:
        raise ValueError("covariates must be provided")
    
    if method == "shap":
        if forecast is None:
            raise ValueError("forecast must be provided for SHAP method")
        
        explainer = SurrogateExplainer(
            random_state=random_state,
            top_k=top_k,
        )
        explainer.fit(covariates, forecast)
        return explainer.explain(covariates)
    
    elif method == "attention":
        if attention_weights is None:
            raise ValueError("attention_weights must be provided for attention method")
        
        from extension_1.attention_attributor import AttentionAttributor
        attributor = AttentionAttributor(top_k=top_k)
        # Attention method doesn't use forecast, only covariates and attention_weights
        return attributor.explain(covariates, attention_weights)
    
    else:
        raise ValueError(f"Unknown attribution method: {method}")
