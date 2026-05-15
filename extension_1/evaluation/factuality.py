"""Deterministic text-quality metrics for forecast verbalizations.

Two metrics, both computed without any model:

fact_recall
    Fraction of key numerical claims (attribution impacts, trajectory
    pct-change, horizon count) that appear in the text within a small
    tolerance.  Answers "Does the text reproduce all key facts?"

feature_completeness
    Fraction of expected feature categories (trend, uncertainty,
    attribution, risk, regime shift) that are mentioned in the text.
    Answers "Are all key features covered?"
"""

from __future__ import annotations

import re
from typing import Any

from extension_1.attribution.types import AttributionResult
from extension_1.features.extractor import ForecastFeatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+\.?\d*", text)]


def _number_present(value: float, numbers: list[float], rtol: float = 0.05) -> bool:
    abs_tol = max(abs(value) * rtol, 0.1)
    return any(abs(n - value) <= abs_tol for n in numbers)


# ---------------------------------------------------------------------------
# Fact recall
# ---------------------------------------------------------------------------

def compute_fact_recall(
    features: ForecastFeatures,
    attribution: AttributionResult,
    text: str,
) -> float:
    """Fraction of key numerical claims that appear in *text*.

    Checks the facts every verbalizer is expected to include:
    - Horizon count (e.g. "96 periods")
    - Attribution impact % for each top-k covariate
    - Trajectory pct_change (if trajectory present)

    Returns 1.0 when there are no checkable facts.
    """
    numbers = _extract_numbers(text)
    facts: list[float] = []

    facts.append(float(features.horizon))

    for attr in attribution.attributions[: attribution.top_k]:
        facts.append(attr.relative_impact_pct)

    if features.trajectory:
        pct = features.trajectory.get("pct_change")
        if pct is not None:
            facts.append(float(pct))

    if not facts:
        return 1.0

    recalled = sum(1 for f in facts if _number_present(f, numbers))
    return recalled / len(facts)


# ---------------------------------------------------------------------------
# Feature completeness
# ---------------------------------------------------------------------------

_TREND_KW = [
    "trend", "rising", "falling", "flat", "increasing", "decreasing",
    "upward", "downward", "stable", "project", "grow", "declin",
    "momentum", "trajectory",
]
_UNCERTAINTY_KW = [
    "uncertain", "interval", "confidence", "spread", "wide", "narrow",
    "tight", "broad", "prediction",
]
_RISK_KW = ["downside", "upside", "risk", "potential"]
_REGIME_KW = ["regime", "structural", "break", "shift"]


def compute_feature_completeness(
    features: ForecastFeatures,
    attribution: AttributionResult,
    text: str,
) -> float:
    """Fraction of *expected* feature categories mentioned in *text*.

    Expected categories (the denominator shrinks for forecasts that
    lack certain features):
    - trend       — always expected
    - uncertainty — always expected
    - attribution — expected when there are covariate attributions
    - risk        — expected only when downside_risk or upside_potential is True
    - regime      — expected only when regime_shift is True

    Returns 1.0 when no categories are expected.
    """
    tl = text.lower()

    def _hit(keywords: list[str]) -> bool:
        return any(kw in tl for kw in keywords)

    expected: list[tuple[str, bool]] = [
        ("trend", _hit(_TREND_KW)),
        ("uncertainty", _hit(_UNCERTAINTY_KW)),
    ]

    if attribution and attribution.attributions:
        cov_names = [
            a.name.replace("_", " ").lower()
            for a in attribution.attributions[: attribution.top_k]
        ]
        expected.append(("attribution", any(n in tl for n in cov_names)))

    if features.downside_risk or features.upside_potential:
        expected.append(("risk", _hit(_RISK_KW)))

    if features.regime_shift:
        expected.append(("regime_shift", _hit(_REGIME_KW)))

    if not expected:
        return 1.0

    return sum(1 for _, hit in expected if hit) / len(expected)
