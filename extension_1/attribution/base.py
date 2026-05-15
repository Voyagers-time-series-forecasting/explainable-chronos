"""Attribution — re-exports for backward-compatible imports."""

from extension_1.attribution.types import AttributionResult, CovariateAttribution, CovariateSet
from extension_1.attribution.attention import AttentionAttributor

__all__ = [
    "AttentionAttributor",
    "AttributionResult",
    "CovariateAttribution",
    "CovariateSet",
]
