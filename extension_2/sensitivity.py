"""
Extension 2 — What-if sensitivity analysis.

Forward-looking what-if / counterfactual sensitivity: perturb a single
covariate (scale or remove it), re-run Chronos-2, and measure how much the
median (P50) forecast moves relative to the unperturbed baseline.

This is the *interventional* counterpart to Extension 1's attention-based
covariate attribution. It is grounded in counterfactual explanations
(Wachter et al., 2017): the minimal input change that alters the outcome.
The same machinery serves two purposes:

  * an interactive forward-looking what-if ("what if marketing spend
    doubled?"), and
  * the perturbation probe used by the attention-faithfulness experiment
    (``extension_2/faithfulness.py``).

The analyzer talks to ``ChronosForecastProvider`` directly (not the full
``VerbalizationPipeline``) because only the P50 forecast is needed; this
keeps the per-perturbation cost low. Covariate perturbations reuse
``InputModifier`` so the modification semantics match the dialogue system.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from extension_1.attribution.types import CovariateSet
from extension_2.modification import InputModifier
from extension_2.parsing.types import ParsedIntent
from shared.forecast_provider import ChronosForecastProvider

logger = logging.getLogger(__name__)


def _extract_p50(forecast_result) -> np.ndarray:
    """Pull the P50 array out of a provider.predict() result.

    ``predict`` returns either a forecast dict or, when attention is enabled,
    a ``(forecast, attention_weights)`` tuple.
    """
    forecast = forecast_result[0] if isinstance(forecast_result, tuple) else forecast_result
    return np.asarray(forecast["p50"], dtype=np.float64)


class WhatIfAnalyzer:
    """Measures how much a covariate perturbation moves the P50 forecast.

    Parameters
    ----------
    provider : ChronosForecastProvider
        Forecast backend (shared with the rest of the system). Attention may
        be enabled or not; only the P50 forecast is read here.
    horizon : int
        Default forecast horizon in steps.
    """

    def __init__(self, provider: ChronosForecastProvider, horizon: int = 14) -> None:
        self.provider = provider
        self.horizon = horizon
        self._modifier = InputModifier(default_horizon=horizon)

    # ── internals ────────────────────────────────────────────────────
    def _predict_p50(
        self,
        history: np.ndarray,
        covariates: CovariateSet,
        horizon: int,
    ) -> np.ndarray:
        past_cov = {name: covariates.values[:, i] for i, name in enumerate(covariates.names)}
        result = self.provider.predict(history, horizon=horizon, past_covariates=past_cov)
        return _extract_p50(result)

    def _perturbed_covariates(
        self,
        covariates: CovariateSet,
        target: str,
        factor: float,
        horizon: int,
    ):
        """Build a perturbed CovariateSet via InputModifier (scale, or remove if factor==0)."""
        if factor == 0.0:
            intent = ParsedIntent(
                intent_type="remove_covariate", raw_query="",
                target_covariate=target,
            )
        else:
            intent = ParsedIntent(
                intent_type="scale_covariate", raw_query="",
                target_covariate=target, scale_factor=float(factor),
            )
        return self._modifier.apply(intent, covariates, horizon)

    # ── public API ───────────────────────────────────────────────────
    def whatif(
        self,
        history: np.ndarray,
        covariates: CovariateSet,
        target: str,
        factor: float,
        horizon: Optional[int] = None,
        base_p50: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """Perturb ``target`` by ``factor`` (0 = remove) and report the P50 delta.

        Returns ``None`` if the modification could not be applied (e.g. the
        covariate was not found).
        """
        h = horizon or self.horizon
        mod = self._perturbed_covariates(covariates, target, factor, h)
        if not mod.modified or mod.covariates is None:
            logger.debug("What-if not applied for target=%s factor=%s: %s", target, factor, mod.description)
            return None

        if base_p50 is None:
            base_p50 = self._predict_p50(history, covariates, h)
        mod_p50 = self._predict_p50(history, mod.covariates, h)

        m = min(len(base_p50), len(mod_p50))
        base_p50, mod_p50 = base_p50[:m], mod_p50[:m]
        delta = mod_p50 - base_p50
        level = float(np.mean(np.abs(base_p50))) + 1e-9

        return {
            "target": target,
            "factor": float(factor),
            "base_p50": base_p50,
            "mod_p50": mod_p50,
            "delta_p50": delta,
            "mean_delta": float(np.mean(delta)),
            "pct_delta": float(np.mean(delta) / level * 100.0),
            "abs_displacement": float(np.mean(np.abs(delta))),
        }

    def sensitivity(
        self,
        history: np.ndarray,
        covariates: CovariateSet,
        mode: str = "remove",
        factors: Sequence[float] = (0.5, 1.5),
        horizon: Optional[int] = None,
    ) -> dict[str, float]:
        """Per-covariate what-if sensitivity score ``s_c``.

        Higher ``s_c`` ⇒ perturbing covariate ``c`` moves the forecast more.
        ``level = mean_t|base_p50|`` normalises across windows.

        ``mode`` selects the intervention:

        * ``"remove"`` (default): zero the covariate (erasure), the intervention
          used by attention-faithfulness work (Serrano & Smith, 2019).
          ``s_c = mean_t|ΔP50_t(remove c)| / level``.
        * ``"negate"``: flip the covariate's sign.
        * ``"scale"``: average ``mean_t|ΔP50_t(f)| / (|f-1|·level)`` over ``factors``.

        Note: Chronos-2 standardises each covariate internally, so it is
        **invariant to positive scaling** — ``"scale"`` therefore yields a
        near-zero (uninformative) signal and exists only for the robustness
        ablation. Prefer ``"remove"``.

        Returns ``{covariate_name: s_c}`` for every covariate.
        """
        h = horizon or self.horizon
        base_p50 = self._predict_p50(history, covariates, h)
        level = float(np.mean(np.abs(base_p50))) + 1e-9

        scores: dict[str, float] = {}
        for name in covariates.names:
            if mode in ("remove", "negate"):
                factor = 0.0 if mode == "remove" else -1.0
                r = self.whatif(history, covariates, name, factor, horizon=h, base_p50=base_p50)
                scores[name] = (r["abs_displacement"] / level) if r is not None else 0.0
            elif mode == "scale":
                contributions = []
                for f in factors:
                    denom = abs(float(f) - 1.0)
                    if denom < 1e-9:
                        continue
                    r = self.whatif(history, covariates, name, f, horizon=h, base_p50=base_p50)
                    if r is None:
                        continue
                    contributions.append(r["abs_displacement"] / (denom * level))
                scores[name] = float(np.mean(contributions)) if contributions else 0.0
            else:
                raise ValueError(f"Unknown sensitivity mode: {mode!r}")
        return scores
