"""Attribution factory — dispatches to SHAP or attention backend."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from extension_1.config import SHAP_TOP_K
from extension_1.attribution.types import AttributionResult, CovariateSet
from extension_1.attribution.shap import SurrogateExplainer
from extension_1.attribution.attention import AttentionAttributor

logger = logging.getLogger(__name__)


def create_attributor(
    method: str,
    covariates: CovariateSet | None = None,
    forecast: np.ndarray | None = None,
    attention_weights: dict[str, Any] | None = None,
    random_state: int = 42,
    top_k: int = SHAP_TOP_K,
) -> AttributionResult:
    """Dispatch to the requested attribution backend.

    Parameters
    ----------
    method : str
        ``"shap"`` or ``"attention"``.
    covariates : CovariateSet
        Required for both backends.
    forecast : np.ndarray, optional
        P50 array — required for SHAP.
    attention_weights : dict, optional
        Attention weights from Chronos-2 — required for attention.
    random_state : int
    top_k : int

    Returns
    -------
    AttributionResult
    """
    if covariates is None:
        raise ValueError("covariates must be provided")

    if method == "shap":
        if forecast is None:
            raise ValueError("forecast must be provided for SHAP method")
        explainer = SurrogateExplainer(random_state=random_state, top_k=top_k)
        explainer.fit(covariates, forecast)
        return explainer.explain(covariates)

    if method == "attention":
        if attention_weights is None:
            raise ValueError("attention_weights must be provided for attention method")
        return AttentionAttributor(top_k=top_k).explain(covariates, attention_weights)

    raise ValueError(f"Unknown attribution method: {method!r}")
