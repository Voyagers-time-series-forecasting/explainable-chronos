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
    direction: str
    relative_impact_pct: float


@dataclass
class AttributionResult:
    """Aggregated attributions from the attention rollout backend."""

    attributions: list[CovariateAttribution]
    top_k: int = 5
