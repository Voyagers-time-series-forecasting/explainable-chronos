"""Deterministic rule-based parser for Extension 2 intents (Tier 1)."""

from __future__ import annotations

import re
from typing import List

from extension_2.parsing.patterns import (
    CONFIDENCE_PATTERNS,
    COUNTERFACTUAL_PATTERN,
    REMOVE_COVARIATE_PATTERNS,
    SCALE_COVARIATE_PATTERNS,
)
from extension_2.parsing.types import ParsedIntent
from extension_2.parsing.slots import extract_horizon, extract_scale_factor, find_covariate


def _matches_any(query: str, patterns: tuple) -> bool:
    return any(re.search(pattern, query) for pattern in patterns)


def rule_parse(query: str, covariate_names: List[str]) -> ParsedIntent:
    """Parse one query using the deterministic rule baseline.

    Priority order: confidence_query → remove_covariate → scale_covariate
    → change_horizon → counterfactual → unknown.
    """
    q = query.lower()

    if _matches_any(q, CONFIDENCE_PATTERNS.patterns):
        return ParsedIntent(intent_type=CONFIDENCE_PATTERNS.intent_type, raw_query=query, confidence="rule")

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
        return ParsedIntent(intent_type="change_horizon", raw_query=query, new_horizon=horizon, confidence="rule")

    if re.search(COUNTERFACTUAL_PATTERN, q):
        return ParsedIntent(
            intent_type="counterfactual",
            raw_query=query,
            target_covariate=find_covariate(query, covariate_names),
            confidence="rule",
        )

    return ParsedIntent(intent_type="unknown", raw_query=query, confidence="fallback")
