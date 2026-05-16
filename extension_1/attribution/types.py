"""Attribution data types."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CovariateSet:
    """Named covariates aligned with the historical time series.

    Attributes
    ----------
    names : list[str]
    values : np.ndarray
        Shape ``(history_length, n_covariates)``.
    descriptions : dict[str, str]
    """

    names: list[str]
    values: np.ndarray
    descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def n_covariates(self) -> int:
        return len(self.names)

    @classmethod
    def from_dict(cls, cov_dict: dict[str, np.ndarray]) -> CovariateSet:
        """Construct from a plain ``{name: array}`` mapping."""
        names = list(cov_dict)
        values = np.stack([cov_dict[n] for n in names], axis=1)
        return cls(names=names, values=values, descriptions={n: n for n in names})


@dataclass
class CovariateAttribution:
    """Attribution score for a single covariate."""

    name: str
    importance_score: float
    relative_impact_pct: float


@dataclass
class TemporalAttribution:
    """Per-covariate temporal saliency over the history window."""

    covariate_name: str
    # saliency[i] = aggregated attention score at history step i
    # normalized to sum to 1; length = history_length
    saliency: np.ndarray
    # History index with the highest saliency
    peak_step: int
    # Normalized entropy: 0 = attention on a single step, 1 = uniform
    focus_breadth: float


@dataclass
class AttributionResult:
    """Aggregated attributions from the attention rollout backend."""

    attributions: list[CovariateAttribution]
    top_k: int = 5
    temporal: list[TemporalAttribution] = field(default_factory=list)
    # history_length / n_patches — useful for converting patch indices back to steps
    patch_to_step_ratio: float = 1.0
