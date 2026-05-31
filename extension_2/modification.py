"""
Extension 2 — Input Modifier.

Applies a ParsedIntent to the current forecast inputs, producing modified
CovariateSet and/or horizon values. Stateless: takes inputs and returns
modified copies without side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from extension_1.attribution.types import CovariateSet
from extension_2.parsing.types import ParsedIntent

logger = logging.getLogger(__name__)


@dataclass
class ModificationResult:
    """Result of applying an intent to the forecast inputs."""
    covariates: Optional[CovariateSet]
    horizon: int
    description: str
    modified: bool


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
        """Apply a parsed intent to the current forecast inputs."""
        if intent.intent_type == "remove_covariate":
            return self._remove_covariate(intent, covariates, current_horizon)
        if intent.intent_type == "scale_covariate":
            return self._scale_covariate(intent, covariates, current_horizon)
        if intent.intent_type == "change_horizon":
            return self._change_horizon(intent, covariates, current_horizon)
        return ModificationResult(
            covariates=covariates,
            horizon=current_horizon,
            description="No modification applied — re-running with current inputs.",
            modified=False,
        )

    def _remove_covariate(self, intent, covariates, horizon):
        if covariates is None:
            return ModificationResult(None, horizon, "No covariates available to remove.", False)
        target = intent.target_covariate
        if target is None or target not in covariates.names:
            target = self._fuzzy_match(target, covariates.names)
        if target is None:
            available = ", ".join(covariates.names)
            return ModificationResult(covariates, horizon, f"Could not identify covariate to remove. Available: {available}.", False)
        modified_cov = self._clone_covariates(covariates)
        modified_cov.values[:, modified_cov.names.index(target)] = 0.0
        return ModificationResult(modified_cov, horizon, f"Covariate '{target}' has been zeroed out.", True)

    def _scale_covariate(self, intent, covariates, horizon):
        if covariates is None:
            return ModificationResult(None, horizon, "No covariates available to scale.", False)
        target = intent.target_covariate
        if target is None or target not in covariates.names:
            target = self._fuzzy_match(target, covariates.names)
        factor = intent.scale_factor
        if factor is None:
            return ModificationResult(covariates, horizon, "Scale factor not specified — inputs unchanged.", False)
        if target is None:
            return ModificationResult(covariates, horizon, f"Could not identify covariate to scale. Available: {', '.join(covariates.names)}.", False)
        modified_cov = self._clone_covariates(covariates)
        idx = modified_cov.names.index(target)
        modified_cov.values[:, idx] = modified_cov.values[:, idx] * factor
        pct = (factor - 1.0) * 100
        direction = "increased" if pct >= 0 else "decreased"
        return ModificationResult(modified_cov, horizon, f"Covariate '{target}' {direction} by {abs(pct):.1f}% (factor: {factor:.2f}).", True)

    def _change_horizon(self, intent, covariates, current_horizon):
        new_horizon = intent.new_horizon
        if new_horizon is None or new_horizon <= 0:
            return ModificationResult(covariates, current_horizon, "Invalid horizon value — keeping current horizon.", False)
        if new_horizon == current_horizon:
            return ModificationResult(covariates, current_horizon, f"Horizon is already {current_horizon} steps.", False)
        return ModificationResult(covariates, new_horizon, f"Forecast horizon changed from {current_horizon} to {new_horizon} steps.", True)

    @staticmethod
    def _clone_covariates(covariates: CovariateSet) -> CovariateSet:
        return CovariateSet(names=list(covariates.names), values=covariates.values.copy(), descriptions=dict(covariates.descriptions))

    @staticmethod
    def _fuzzy_match(target: Optional[str], names: list) -> Optional[str]:
        if target is None:
            return None
        target_tokens = set(target.lower().replace("_", " ").split())
        best, best_score = None, 0
        for name in names:
            score = len(target_tokens & set(name.lower().replace("_", " ").split()))
            if score > best_score:
                best_score = score
                best = name
        return best if best_score > 0 else None
