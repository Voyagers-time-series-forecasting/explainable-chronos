"""
Extension 2 — Intent Parser (three-tier pipeline).

  Tier 1: deterministic rule-based parser (always runs first).
  Tier 2: BERT few-shot classifier (when Tier 1 returns 'unknown').
  Tier 3: LLM fallback (when Tier 2 is still uncertain).
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extension_2.parsing.bert import BertIntentClassifier
from extension_2.parsing.rules import rule_parse
from extension_2.parsing.types import INTENT_TYPES, ParsedIntent

logger = logging.getLogger(__name__)

__all__ = ["INTENT_TYPES", "IntentParser", "ParsedIntent"]

DEFAULT_BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "Qwen/Qwen1.5-1.8B-Chat"

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
    model_name: str = DEFAULT_LLM_MODEL,
) -> ParsedIntent:
    """Use a local LLM to parse queries that neither rules nor BERT resolved."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        system = _LLM_SYSTEM_PROMPT.format(covariate_names=", ".join(covariate_names))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=128, temperature=0.1,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
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
        logger.warning("LLM parsing failed (%s), returning unknown.", e)
        return ParsedIntent(intent_type="unknown", raw_query=query, confidence="fallback")


class IntentParser:
    """Three-tier intent parser.

    Parameters
    ----------
    covariate_names : list of str
        Known covariate names for slot extraction.
    use_bert_tier : bool
        Enable Tier 2. Default True.
    use_llm_fallback : bool
        Enable Tier 3. Default True.
    bert_model_name : str
        Sentence-BERT model for Tier 2.
    llm_model_name : str
        LLM model for Tier 3.
    """

    def __init__(
        self,
        covariate_names: Optional[List[str]] = None,
        use_bert_tier: bool = True,
        use_llm_fallback: bool = True,
        bert_model_name: str = DEFAULT_BERT_MODEL,
        llm_model_name: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self.covariate_names: List[str] = covariate_names or []
        self.use_bert_tier = use_bert_tier
        self.use_llm_fallback = use_llm_fallback
        self.bert_model_name = bert_model_name
        self.llm_model_name = llm_model_name
        self._bert_classifier: Optional[BertIntentClassifier] = None

    def _get_bert_classifier(self) -> BertIntentClassifier:
        if self._bert_classifier is None:
            clf = BertIntentClassifier(model_name=self.bert_model_name)
            clf.fit_from_eval_sets()
            self._bert_classifier = clf
        return self._bert_classifier

    def parse(self, query: str) -> ParsedIntent:
        """Parse a natural-language query through the three-tier pipeline."""
        if not query or not query.strip():
            return ParsedIntent(intent_type="unknown", raw_query=query, confidence="fallback")

        q = query.strip()

        # Tier 1: rules
        intent = rule_parse(q, self.covariate_names)
        if intent.intent_type != "unknown":
            logger.debug("Tier 1 resolved: %s", intent.intent_type)
            return intent

        # Tier 2: BERT few-shot
        if self.use_bert_tier:
            logger.info("Tier 1 unknown — trying BERT tier.")
            intent = self._get_bert_classifier().predict(q, self.covariate_names)
            if intent.intent_type != "unknown":
                logger.debug("Tier 2 resolved: %s", intent.intent_type)
                return intent

        # Tier 3: LLM fallback
        if self.use_llm_fallback and (intent.intent_type == "unknown" or not intent.is_actionable()):
            logger.info("Tier 2 unknown — trying LLM fallback.")
            intent = _llm_parse(q, self.covariate_names, self.llm_model_name)

        logger.debug("Final intent: %s", intent.to_dict())
        return intent

    def parse_batch(self, queries: List[str]) -> List[ParsedIntent]:
        return [self.parse(q) for q in queries]
