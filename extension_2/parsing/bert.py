"""BERT-based few-shot intent classifier for Extension 2 (Tier 2)."""

from __future__ import annotations

import logging
from collections import Counter
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from extension_2.datasets import EXAMPLE_SET
from extension_2.parsing.slots import extract_horizon, extract_scale_factor, find_covariate
from extension_2.parsing.types import ParsedIntent

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class BertIntentClassifier:
    """Few-shot intent classifier backed by sentence-BERT embeddings.

    Uses the DEV + TEST labeled examples (70 queries total) as the
    few-shot pool. At inference the incoming query is embedded and
    compared via cosine similarity; the top-k neighbours vote on the
    intent type. Abstains (returns 'unknown') when the best similarity
    falls below the confidence threshold, escalating to Tier 3.

    Parameters
    ----------
    model_name : str
        Sentence-BERT model identifier.
    k : int
        Number of nearest neighbours for majority voting.
    confidence_threshold : float
        Minimum cosine similarity required to accept a prediction.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        k: int = 5,
        confidence_threshold: float = 0.45,
    ) -> None:
        self.model_name = model_name
        self.k = k
        self.confidence_threshold = confidence_threshold
        self._model: Optional[SentenceTransformer] = None
        self._example_embeddings: Optional[np.ndarray] = None
        self._example_labels: List[str] = []
        self._example_queries: List[str] = []

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading sentence-BERT model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, queries: List[str], labels: List[str]) -> None:
        """Embed and store a labeled example pool."""
        if len(queries) != len(labels):
            raise ValueError("queries and labels must have the same length.")
        model = self._load_model()
        self._example_queries = list(queries)
        self._example_labels = list(labels)
        self._example_embeddings = model.encode(
            queries, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )
        logger.info("BERT tier: embedded %d examples (%d unique intents).", len(queries), len(set(labels)))

    def fit_from_eval_sets(self) -> None:
        """Populate the pool from the 70-query example set."""
        self.fit(
            queries=[tc.query for tc in EXAMPLE_SET],
            labels=[tc.expected_intent for tc in EXAMPLE_SET],
        )

    def predict(self, query: str, covariate_names: List[str]) -> ParsedIntent:
        """Classify a single query via nearest-neighbour voting."""
        if self._example_embeddings is None:
            raise RuntimeError("Call fit() or fit_from_eval_sets() before predict().")

        model = self._load_model()
        query_emb = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )

        similarities = (self._example_embeddings @ query_emb.T).squeeze()
        top_k_idx = np.argsort(similarities)[::-1][: self.k]
        top_k_sims = similarities[top_k_idx]
        top_k_labels = [self._example_labels[i] for i in top_k_idx]
        best_sim = float(top_k_sims[0])

        if best_sim < self.confidence_threshold:
            logger.debug("BERT tier: best sim %.3f below threshold — abstaining.", best_sim)
            return ParsedIntent(intent_type="unknown", raw_query=query, confidence="fallback")

        predicted_intent, vote_count = Counter(top_k_labels).most_common(1)[0]
        logger.debug("BERT tier: '%s' (sim=%.3f, votes=%d/%d).", predicted_intent, best_sim, vote_count, self.k)

        return ParsedIntent(
            intent_type=predicted_intent,
            raw_query=query,
            target_covariate=find_covariate(query, covariate_names),
            scale_factor=extract_scale_factor(query),
            new_horizon=extract_horizon(query),
            confidence="bert",
        )
