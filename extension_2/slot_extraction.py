"""Slot extraction helpers for rule-based intent parsing."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from extension_2.intent_patterns import (
    DECREASE_MARKERS,
    FACTOR_WORDS,
    HORIZON_PATTERNS,
    HORIZON_UNITS,
    INCREASE_MARKERS,
)


_PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_MULTIPLIER_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*x\b")
_TOKEN_PATTERN = re.compile(r"\b\w+\b")


def find_covariate(query: str, covariate_names: List[str]) -> Optional[str]:
    """Find the best matching covariate name in the query."""
    q = query.lower()

    for name in covariate_names:
        normalized = name.lower().replace("_", " ")
        if normalized in q or name.lower() in q:
            return name

    query_tokens = set(_TOKEN_PATTERN.findall(q))
    best_name = None
    best_overlap = 0
    for name in covariate_names:
        name_tokens = set(_TOKEN_PATTERN.findall(name.lower()))
        overlap = len(name_tokens & query_tokens - {"a", "the", "of", "in"})
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name

    if best_overlap >= 1:
        return best_name

    for name in covariate_names:
        parts = name.lower().split("_")
        if any(len(part) > 3 and part in q for part in parts):
            return name

    return None


def extract_scale_factor(query: str) -> Optional[float]:
    """Extract a multiplicative scale factor from a query."""
    q = query.lower()

    for word, factor in FACTOR_WORDS.items():
        if word in q:
            return factor

    pct_match = _PERCENT_PATTERN.search(q)
    if pct_match:
        pct = float(pct_match.group(1))
        if any(marker in q for marker in DECREASE_MARKERS):
            return 1.0 - pct / 100.0
        if any(marker in q for marker in INCREASE_MARKERS):
            return 1.0 + pct / 100.0
        return pct / 100.0

    multiplier_match = _MULTIPLIER_PATTERN.search(q)
    if multiplier_match:
        return float(multiplier_match.group(1))

    return None


def extract_horizon(
    query: str,
    horizon_patterns: Iterable[str] = HORIZON_PATTERNS,
) -> Optional[int]:
    """Extract a forecast horizon in steps."""
    q = query.lower()
    for pattern in horizon_patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if not match:
            continue
        try:
            n = int(match.group(1))
        except (IndexError, ValueError):
            continue

        unit_match = re.search(
            r"\b(hour|day|week|month|period|step)s?\b",
            match.group(0),
            re.IGNORECASE,
        )
        if not unit_match:
            return n

        unit = unit_match.group(1).lower()
        return n * HORIZON_UNITS.get(unit, 1)

    return None
