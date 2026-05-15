"""
Extension 2 — Input Modifier.

Applies a ParsedIntent to the current forecast inputs, producing
modified CovariateSet and/or horizon values that can be passed
directly to ChronosForecastProvider.predict().

This module is intentionally stateless — it takes inputs and returns
modified copies without side effects.

Usage::

    modifier = InputModifier()
    modified_covs, new_horizon = modifier.apply(intent, covariates, current_horizon)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from covariate_attribution import CovariateSet
from extension_2.intent_parser import ParsedIntent

logger = logging.getLogger(__name__)


# ──────────────── modification result ────────────────────────────────

@dataclass
class ModificationResult:
    """Result of applying an intent to the forecast inputs.

    Attributes
    ----------
    covariates : CovariateSet | None
        Modified covariate set, or None if unchanged / not applicable.
    horizon : int
        Forecast horizon (may be unchanged).
    description : str
        Human-readable description of what was modified.
    modified : bool
        True if any actual modification was applied.
    """
    covariates: Optional[CovariateSet]
    horizon: int
    description: str
    modified: bool


# ──────────────── modifier class ─────────────────────────────────────

class InputModifier:
    """Applies parsed intents to forecast inputs.

    Parameters
    ----------
    default_horizon : int
        Fallback horizon when the intent does not specify one.
    """

    def __init__(self, default_horizon: int = 14) -> None:
        self.default_horizon = default_horizon

    def apply(
        self,
        intent: ParsedIntent,
        covariates: Optional[CovariateSet],
        current_horizon: int,
    ) -> ModificationResult:
        """Apply a parsed intent to the current forecast inputs.

        Parameters
        ----------
        intent : ParsedIntent
            Output of IntentParser.parse().
        covariates : CovariateSet | None
            Current covariate set (may be None if univariate mode).
        current_horizon : int
            Current forecast horizon in steps.

        Returns
        -------
        ModificationResult
            Modified inputs with a human-readable description.
        """
        if intent.intent_type == "remove_covariate":
            return self._remove_covariate(intent, covariates, current_horizon)

        if intent.intent_type == "scale_covariate":
            return self._scale_covariate(intent, covariates, current_horizon)

        if intent.intent_type == "change_horizon":
            return self._change_horizon(intent, covariates, current_horizon)

        if intent.intent_type in ("confidence_query", "counterfactual", "unknown"):
            # No modification needed — return inputs unchanged
            return ModificationResult(
                covariates=covariates,
                horizon=current_horizon,
                description="No modification applied — re-running with current inputs.",
                modified=False,
            )

        return ModificationResult(
            covariates=covariates,
            horizon=current_horizon,
            description=f"Unknown intent type '{intent.intent_type}' — inputs unchanged.",
            modified=False,
        )

    # ── private handlers ─────────────────────────────────────────────

    def _remove_covariate(
        self,
        intent: ParsedIntent,
        covariates: Optional[CovariateSet],
        horizon: int,
    ) -> ModificationResult:
        """Zero out a specific covariate channel."""
        if covariates is None:
            return ModificationResult(
                covariates=None,
                horizon=horizon,
                description="No covariates available to remove.",
                modified=False,
            )

        target = intent.target_covariate
        if target is None or target not in covariates.names:
            # Try partial match
            target = self._fuzzy_match(target, covariates.names)

        if target is None:
            available = ", ".join(covariates.names)
            return ModificationResult(
                covariates=covariates,
                horizon=horizon,
                description=(
                    f"Could not identify covariate to remove. "
                    f"Available covariates: {available}."
                ),
                modified=False,
            )

        modified_cov = self._clone_covariates(covariates)
        idx = modified_cov.names.index(target)
        modified_cov.values[:, idx] = 0.0

        return ModificationResult(
            covariates=modified_cov,
            horizon=horizon,
            description=f"Covariate '{target}' has been zeroed out.",
            modified=True,
        )

    def _scale_covariate(
        self,
        intent: ParsedIntent,
        covariates: Optional[CovariateSet],
        horizon: int,
    ) -> ModificationResult:
        """Multiply a covariate channel by a scale factor."""
        if covariates is None:
            return ModificationResult(
                covariates=None,
                horizon=horizon,
                description="No covariates available to scale.",
                modified=False,
            )

        target = intent.target_covariate
        if target is None or target not in covariates.names:
            target = self._fuzzy_match(target, covariates.names)

        factor = intent.scale_factor
        if factor is None:
            return ModificationResult(
                covariates=covariates,
                horizon=horizon,
                description="Scale factor not specified — inputs unchanged.",
                modified=False,
            )

        if target is None:
            available = ", ".join(covariates.names)
            return ModificationResult(
                covariates=covariates,
                horizon=horizon,
                description=(
                    f"Could not identify covariate to scale. "
                    f"Available: {available}."
                ),
                modified=False,
            )

        modified_cov = self._clone_covariates(covariates)
        idx = modified_cov.names.index(target)
        modified_cov.values[:, idx] = modified_cov.values[:, idx] * factor

        pct_change = (factor - 1.0) * 100
        direction = "increased" if pct_change >= 0 else "decreased"
        description = (
            f"Covariate '{target}' {direction} by {abs(pct_change):.1f}% "
            f"(scale factor: {factor:.2f})."
        )

        return ModificationResult(
            covariates=modified_cov,
            horizon=horizon,
            description=description,
            modified=True,
        )

    def _change_horizon(
        self,
        intent: ParsedIntent,
        covariates: Optional[CovariateSet],
        current_horizon: int,
    ) -> ModificationResult:
        """Change the forecast horizon."""
        new_horizon = intent.new_horizon
        if new_horizon is None or new_horizon <= 0:
            return ModificationResult(
                covariates=covariates,
                horizon=current_horizon,
                description="Invalid horizon value — keeping current horizon.",
                modified=False,
            )

        if new_horizon == current_horizon:
            return ModificationResult(
                covariates=covariates,
                horizon=current_horizon,
                description=f"Horizon is already {current_horizon} steps.",
                modified=False,
            )

        description = (
            f"Forecast horizon changed from {current_horizon} "
            f"to {new_horizon} steps."
        )

        return ModificationResult(
            covariates=covariates,
            horizon=new_horizon,
            description=description,
            modified=True,
        )

    # ── utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _clone_covariates(covariates: CovariateSet) -> CovariateSet:
        """Return a deep copy of a CovariateSet."""
        return CovariateSet(
            names=list(covariates.names),
            values=covariates.values.copy(),
            descriptions=dict(covariates.descriptions),
        )

    @staticmethod
    def _fuzzy_match(
        target: Optional[str],
        names: list,
    ) -> Optional[str]:
        """Find the best matching name via token overlap."""
        if target is None:
            return None
        target_tokens = set(target.lower().replace("_", " ").split())
        best, best_score = None, 0
        for name in names:
            name_tokens = set(name.lower().replace("_", " ").split())
            score = len(target_tokens & name_tokens)
            if score > best_score:
                best_score = score
                best = name
        return best if best_score > 0 else None