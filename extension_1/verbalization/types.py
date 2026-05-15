"""Verbalization domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerbalizationResult:
    """Output of any verbalization step.

    Attributes
    ----------
    summary : str
        Full paragraph combining all sentences.
    sentences : list[str]
        Individual sentences that compose the summary.
    grounding : dict[str, Any]
        Maps each sentence by index to the numerical features that
        generated it — used by NLI consistency checking.
    rst_relations : list[str]
        RST discourse relations triggered during verbalization.
    """

    summary: str
    sentences: list[str]
    grounding: dict[str, Any]
    rst_relations: list[str] = field(default_factory=list)
    draft_summary: str | None = None
    prompt: str | None = None
