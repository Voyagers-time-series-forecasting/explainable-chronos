"""Extension 2 — conversational dialogue interface for Chronos-2."""

from extension_2.dialogue import DialogueSystem
from extension_2.modification import InputModifier, ModificationResult
from extension_2.parsing import INTENT_TYPES, IntentParser, ParsedIntent

__all__ = [
    "DialogueSystem",
    "INTENT_TYPES",
    "InputModifier",
    "IntentParser",
    "ModificationResult",
    "ParsedIntent",
]
