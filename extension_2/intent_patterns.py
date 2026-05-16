"""Rule catalog for the deterministic Extension 2 intent baseline.

The parser deliberately keeps these patterns data-only. This makes the
rule-based baseline easy to inspect, test, and replace without burying
regular expressions in the public parser API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class PatternSet:
    """Named regex collection for one intent family."""

    intent_type: str
    patterns: Tuple[str, ...]


CONFIDENCE_PATTERNS = PatternSet(
    intent_type="confidence_query",
    patterns=(
        r"\bhow\s+confident\b",
        r"\bhow\s+certain\b",
        r"\bhow\s+sure\b",
        r"\bhow\s+wide\b",
        r"\buncertain(?:ty)?\b",
        r"\bconfidence\b",
        r"\bprediction\s+intervals?\b",
        r"\bp10\b",
        r"\bp90\b",
        r"\brange\b.{0,40}\bforecast\b",
        r"\bforecast\b.{0,40}\brange\b",
        r"\bwhat\s+range\b",
        r"\bmargin\s+of\s+error\b",
        r"\bupside\b",
        r"\bdownside\b",
        r"\bbest\s+case\b",
        r"\bworst\s+case\b",
        r"\binterval\s+width\b",
        r"\breliable\b",
        r"\btrust(?:worthy)?\b",
    ),
)

REMOVE_COVARIATE_PATTERNS = PatternSet(
    intent_type="remove_covariate",
    patterns=(
        r"\bremov(?:e|ing|ed)\b",
        r"\bwithout\b",
        r"\bno\b.{0,20}\bcovariate\b",
        r"\beliminate\b",
        r"\bdrop\b",
        r"\bexclud(?:e|ing)\b",
        r"\bzero(?:\s+out)?\b",
        r"\bif\b.{0,30}\bweren'?t\b",
        r"\bif\b.{0,30}\bdidn'?t\s+have\b",
        r"\bwhat\s+if\b.{0,50}\bgone\b",
        r"\bwere\s+gone\b",
        r"\bwhat\s+if\b.{0,30}\bno\b",
        r"\bif\b.{0,30}\bno\b.{0,20}\bdata\b",
        r"\bdidn'?t\s+affect\b",
        r"\bwouldn'?t\s+affect\b",
    ),
)

SCALE_COVARIATE_PATTERNS = PatternSet(
    intent_type="scale_covariate",
    patterns=(
        r"\bincreas(?:e|ing)\b",
        r"\bdecreas(?:e|ing)\b",
        r"\breduc(?:e|ing)\b",
        r"\bdoubl(?:e|ing|ed)\b",
        r"\bhalv(?:e|ing|ed)\b",
        r"\btriple[d]?\b",
        r"\bby\s+\d+\s*%",
        r"\b\d+\s*%\s+(?:more|less|higher|lower)\b",
        r"\bdropped?\s+by\b",
        r"\brose?\s+by\b",
        r"\bscal(?:e|ing)\b",
        r"\bmultipl(?:y|ied)\b",
        r"\brose\b",
        r"\bsoared\b",
    ),
)

HORIZON_PATTERNS = (
    r"\bnext\s+(\d+)\s*(?:day|week|month|hour|period|step)s?\b",
    r"\b(\d+)\s*(?:day|week|month|hour|period|step)s?\s+(?:ahead|forward|forecast)\b",
    r"\bforecast\s+(?:for\s+)?(\d+)\s*(?:day|week|month|hour|period|step)s?\b",
    r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:next\s+)?(\d+)\s*(?:day|week|month|hour|period|step)s?\b",
    r"\bhorizon\s+(?:of\s+)?(\d+)\b",
    r"\bpredict\s+(\d+)\s*(?:day|week|month|hour|period|step)s?\b",
    r"\b(\d+)[\s-](?:day|week|month|hour|period|step)\s+(?:forecast|prediction|outlook)\b",
    r"\bextend\b.{0,20}\bto\s+(\d+)\s*(?:day|week|month|hour|period|step)s?\b",
    r"\b(\d+)\s*(?:day|week|month|hour|period|step)s?\s+prediction\b",
    r"\bneed\s+a\s+(\d+)[\s-](?:day|week|month|hour|period|step)\b",
)

HORIZON_WORD_NUMBERS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
}

HORIZON_UNITS: Dict[str, int] = {
    "hour": 1,
    "hours": 1,
    "day": 24,
    "days": 24,
    "week": 168,
    "weeks": 168,
    "month": 720,
    "months": 720,
    "period": 1,
    "periods": 1,
    "step": 1,
    "steps": 1,
}

FACTOR_WORDS: Dict[str, float] = {
    "double": 2.0,
    "doubled": 2.0,
    "triple": 3.0,
    "tripled": 3.0,
    "half": 0.5,
    "halve": 0.5,
    "halved": 0.5,
    "quarter": 0.25,
}

DECREASE_MARKERS = ("decreas", "reduc", "drop", "less", "lower", "cut")
INCREASE_MARKERS = ("increas", "rise", "more", "higher", "boost")
COUNTERFACTUAL_PATTERN = r"\bwhat\s+if\b|\bwhat\s+would\b|\bif\s+we\b|\bhypothetical\b"
