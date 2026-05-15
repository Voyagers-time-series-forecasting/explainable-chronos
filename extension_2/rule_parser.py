"""Deterministic rule-based parser for Extension 2 intents."""

from __future__ import annotations

import re
from typing import List

from extension_2.intent_patterns import (
    CONFIDENCE_PATTERNS,
    COUNTERFACTUAL_PATTERN,
    REMOVE_COVARIATE_PATTERNS,
    SCALE_COVARIATE_PATTERNS,
)
from extension_2.intent_types import ParsedIntent
from extension_2.slot_extraction import (
    extract_horizon,
    extract_scale_factor,
    find_covariate,
)


def _matches_any(query: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, query) for pattern in patterns)


def rule_parse(query: str, covariate_names: List[str]) -> ParsedIntent:
    """Parse one query using the deterministic baseline.

    Priority order is part of the baseline definition:
    confidence queries are checked before horizon changes, and
    covariate edits are checked before generic counterfactuals.
    """
    q = query.lower()

    if _matches_any(q, CONFIDENCE_PATTERNS.patterns):
        return ParsedIntent(
            intent_type=CONFIDENCE_PATTERNS.intent_type,
            raw_query=query,
            confidence="rule",
        )

    if _matches_any(q, REMOVE_COVARIATE_PATTERNS.patterns):
        return ParsedIntent(
            intent_type=REMOVE_COVARIATE_PATTERNS.intent_type,
            raw_query=query,
            target_covariate=find_covariate(query, covariate_names),
            scale_factor=0.0,
            confidence="rule",
        )

    if _matches_any(q, SCALE_COVARIATE_PATTERNS.patterns):
        covariate = find_covariate(query, covariate_names)
        scale_factor = extract_scale_factor(query)
        if covariate is not None or scale_factor is not None:
            return ParsedIntent(
                intent_type=SCALE_COVARIATE_PATTERNS.intent_type,
                raw_query=query,
                target_covariate=covariate,
                scale_factor=scale_factor,
                confidence="rule",
            )

    horizon = extract_horizon(query)
    if horizon is not None:
        return ParsedIntent(
            intent_type="change_horizon",
            raw_query=query,
            new_horizon=horizon,
            confidence="rule",
        )

    if re.search(COUNTERFACTUAL_PATTERN, q):
        return ParsedIntent(
            intent_type="counterfactual",
            raw_query=query,
            target_covariate=find_covariate(query, covariate_names),
            confidence="rule",
        )

    return ParsedIntent(
        intent_type="unknown",
        raw_query=query,
        confidence="fallback",
    )
