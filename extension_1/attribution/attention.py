"""
Attention-based covariate attribution using Attention Rollout.

Implements the Attention Rollout technique ("Quantifying Attention Flow
in Transformers") to aggregate transformer attention weights across
layers and heads, providing more reliable explanations than raw
attention summation.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from extension_1.attribution.types import (
    AttributionResult,
    CovariateAttribution,
    CovariateSet,
)

logger = logging.getLogger(__name__)


class AttentionAttributor:
    """Attributor using Attention Rollout for covariate importance.

    Parameters
    ----------
    top_k : int
        Maximum number of top attributions to return.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def explain(
        self,
        covariates: CovariateSet,
        attention_weights: dict[str, Any],
    ) -> AttributionResult:
        """Compute covariate attributions using Attention Rollout.

        Parameters
        ----------
        covariates : CovariateSet
        attention_weights : dict
            ``{"attentions": {"enc_time_self_attn_weights": ...}}``

        Returns
        -------
        AttributionResult
        """
        if not attention_weights or "attentions" not in attention_weights:
            raise ValueError(
                "Attention weights not available. "
                "Ensure ChronosForecastProvider is created with enable_attention=True."
            )

        attentions = attention_weights["attentions"]
        if attentions is None:
            raise ValueError("No attention weights captured from model.")

        time_attn = attentions.get("enc_time_self_attn_weights")
        if not time_attn:
            raise ValueError("No enc_time_self_attn_weights available.")

        rollout = self._compute_rollout(list(time_attn))
        importance = self._aggregate_by_covariate(rollout, covariates)

        attributions = sorted(
            [
                CovariateAttribution(
                    name=name,
                    importance_score=score,
                    direction="positive" if score >= 0 else "negative",
                    relative_impact_pct=abs(score) * 100,
                )
                for name, score in importance.items()
            ],
            key=lambda a: abs(a.importance_score),
            reverse=True,
        )[: self.top_k]

        total = sum(abs(a.importance_score) for a in attributions)
        if total > 0:
            for attr in attributions:
                attr.relative_impact_pct = abs(attr.importance_score) / total * 100

        return AttributionResult(
            attributions=attributions,
            top_k=self.top_k,
        )

    def _compute_rollout(self, attentions: list[Any]) -> torch.Tensor:
        """Compute Attention Rollout across all layers.

        Each attention tensor has shape ``(batch, heads, seq_len, seq_len)``.
        Returns a rollout matrix of shape ``(seq_len, seq_len)``.
        """
        rollout: torch.Tensor | None = None
        for layer_attn in attentions:
            avg = layer_attn.mean(dim=1)  # avg over heads
            seq = avg.shape[-1]
            identity = (
                torch.eye(seq, device=avg.device, dtype=avg.dtype)
                .unsqueeze(0)
                .expand_as(avg)
            )
            layer_norm = avg + identity
            layer_norm = layer_norm / layer_norm.sum(dim=-1, keepdim=True)
            rollout = layer_norm if rollout is None else torch.matmul(rollout, layer_norm)
        return rollout.squeeze(0) if rollout is not None else torch.empty(0)

    def _aggregate_by_covariate(
        self,
        rollout: torch.Tensor,
        covariates: CovariateSet,
    ) -> dict[str, float]:
        """Map rollout attention sums to covariate importance scores."""
        n = len(covariates.names)
        if rollout.numel() == 0:
            return {name: 1.0 / n for name in covariates.names}

        seq = rollout.shape[-1]
        if n == 0 or seq < n:
            base = rollout.sum().item() / max(n, 1)
            return {name: base for name in covariates.names}

        seg = seq // n
        rem = seq % n
        result: dict[str, float] = {}
        pos = 0
        for i, name in enumerate(covariates.names):
            end = pos + seg + (1 if i < rem else 0)
            result[name] = rollout[:, pos:end].sum().item()
            pos = end
        return result
