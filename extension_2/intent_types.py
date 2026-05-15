"""Shared intent schema for Extension 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


INTENT_TYPES = [
    "remove_covariate",
    "scale_covariate",
    "change_horizon",
    "confidence_query",
    "counterfactual",
    "unknown",
]


@dataclass
class ParsedIntent:
    """Structured output of the intent parser."""

    intent_type: str
    raw_query: str
    target_covariate: Optional[str] = None
    scale_factor: Optional[float] = None
    new_horizon: Optional[int] = None
    confidence: str = "rule"

    def is_actionable(self) -> bool:
        """Return True if the intent has enough information to act on."""
        if self.intent_type == "remove_covariate":
            return self.target_covariate is not None
        if self.intent_type == "scale_covariate":
            return self.target_covariate is not None and self.scale_factor is not None
        if self.intent_type == "change_horizon":
            return self.new_horizon is not None
        if self.intent_type in ("confidence_query", "counterfactual"):
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_type": self.intent_type,
            "target_covariate": self.target_covariate,
            "scale_factor": self.scale_factor,
            "new_horizon": self.new_horizon,
            "raw_query": self.raw_query,
            "confidence": self.confidence,
        }
