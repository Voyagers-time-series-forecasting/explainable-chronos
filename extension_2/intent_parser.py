"""
Extension 2 - Intent Parser.

Public parser API for dialogue queries. The deterministic rule baseline
lives in ``rule_parser.py``; this module keeps orchestration and the
optional LLM fallback in one place.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from extension_2.intent_types import INTENT_TYPES, ParsedIntent
from extension_2.rule_parser import rule_parse

logger = logging.getLogger(__name__)

__all__ = ["INTENT_TYPES", "IntentParser", "ParsedIntent"]


_LLM_SYSTEM_PROMPT = """You are an intent parser for a time series forecasting dialogue system.

Given a user query, extract the intent and return ONLY a JSON object with these fields:
- intent_type: one of ["remove_covariate", "scale_covariate", "change_horizon", "confidence_query", "counterfactual", "unknown"]
- target_covariate: the covariate name from the list, or null
- scale_factor: a float multiplier (e.g. 0.5 for "halve", 1.3 for "increase by 30%"), or null
- new_horizon: integer number of forecast steps, or null

Available covariates: {covariate_names}

Return ONLY the JSON object, no explanation, no markdown."""


def _llm_parse(
    query: str,
    covariate_names: List[str],
    model_name: str = "Qwen/Qwen1.5-1.8B-Chat",
) -> ParsedIntent:
    """Use a local LLM to parse complex queries.

    Falls back gracefully if the LLM is not available or returns malformed
    output. This tier is optional so the default evaluation remains fully
    deterministic and reproducible.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        system = _LLM_SYSTEM_PROMPT.format(
            covariate_names=", ".join(covariate_names)
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        json_match = re.search(r"\{.*\}", generated, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in LLM output: {generated}")

        data = json.loads(json_match.group(0))

        return ParsedIntent(
            intent_type=data.get("intent_type", "unknown"),
            raw_query=query,
            target_covariate=data.get("target_covariate"),
            scale_factor=data.get("scale_factor"),
            new_horizon=data.get("new_horizon"),
            confidence="llm",
        )

    except Exception as e:
        logger.warning("LLM parsing failed (%s), falling back to unknown.", e)
        return ParsedIntent(
            intent_type="unknown",
            raw_query=query,
            confidence="fallback",
        )


class IntentParser:
    """Two-tier intent parser for the dialogue system.

    Tier 1 is a deterministic rule-based baseline. Tier 2 is an optional
    local LLM fallback for unknown or unactionable parses.
    """

    def __init__(
        self,
        covariate_names: Optional[List[str]] = None,
        use_llm_fallback: bool = False,
        llm_model_name: str = "Qwen/Qwen1.5-1.8B-Chat",
    ) -> None:
        self.covariate_names: List[str] = covariate_names or []
        self.use_llm_fallback = use_llm_fallback
        self.llm_model_name = llm_model_name

    def parse(self, query: str) -> ParsedIntent:
        """Parse a natural-language query into a structured intent."""
        if not query or not query.strip():
            return ParsedIntent(
                intent_type="unknown",
                raw_query=query,
                confidence="fallback",
            )

        cleaned_query = query.strip()
        intent = rule_parse(cleaned_query, self.covariate_names)

        if self.use_llm_fallback and (
            intent.intent_type == "unknown" or not intent.is_actionable()
        ):
            logger.info("Rule-based tier returned '%s', trying LLM.", intent.intent_type)
            intent = _llm_parse(cleaned_query, self.covariate_names, self.llm_model_name)

        logger.debug("Parsed intent: %s", intent.to_dict())
        return intent

    def parse_batch(self, queries: List[str]) -> List[ParsedIntent]:
        """Parse a list of queries."""
        return [self.parse(query) for query in queries]
