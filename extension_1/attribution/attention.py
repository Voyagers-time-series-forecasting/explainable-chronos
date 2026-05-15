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

import numpy as np
import torch

from extension_1.attribution.types import (
    AttributionResult,
    CovariateAttribution,
    CovariateSet,
    TemporalAttribution,
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
        future_covariates: CovariateSet | None = None,
    ) -> AttributionResult:
        """Compute covariate attributions using Attention Rollout over groups.

        In Chronos-2, covariates are processed as separate 'groups'. The
        ``enc_group_self_attn_weights`` tracks attention between the target
        series and these covariates. Past and future values of a covariate 
        are concatenated in time and belong to the same group.

        Parameters
        ----------
        covariates : CovariateSet
            Past covariates (history window).
        attention_weights : dict
            ``{"attentions": {"enc_group_self_attn_weights": ...}}``
        future_covariates : CovariateSet, optional
            Provided for signature compatibility; Chronos treats past and 
            future as a single continuous group.

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

        group_attn = attentions.get("enc_group_self_attn_weights")
        if not group_attn:
            raise ValueError("No enc_group_self_attn_weights available. Model may not support group attention.")

        # Compute rollout over the Group dimension (Target + Covariates)
        rollout = self._compute_rollout(list(group_attn))

        # The first row (index 0) is the Target's attention to all groups.
        # Index 1 to N are the covariates in the exact order they were provided.
        importance = self._aggregate_by_covariate(rollout, list(covariates.names))

        attributions = sorted(
            [
                CovariateAttribution(
                    name=name,
                    importance_score=score,
                    direction="positive", # Attention is strictly positive magnitude
                    relative_impact_pct=score * 100,
                )
                for name, score in importance.items()
            ],
            key=lambda a: a.importance_score,
            reverse=True,
        )[: self.top_k]

        total = sum(a.importance_score for a in attributions)
        if total > 0:
            for attr in attributions:
                attr.relative_impact_pct = (attr.importance_score / total) * 100

        history_length = len(covariates.values)
        temporal, patch_ratio = self._compute_temporal_saliency(
            list(group_attn), list(covariates.names), history_length
        )

        return AttributionResult(
            attributions=attributions,
            top_k=self.top_k,
            temporal=temporal,
            patch_to_step_ratio=patch_ratio,
        )

    def _compute_temporal_saliency(
        self,
        group_attn_layers: list[Any],
        covariate_names: list[str],
        history_length: int,
    ) -> tuple[list[TemporalAttribution], float]:
        """Compute per-covariate temporal saliency from group attention layers.

        For each layer the group attention has shape ``(Time, Heads, Groups, Groups)``.
        We extract how much the Target group (index 0) attends to each covariate
        group (indices 1+) at every patch position, average across layers, then
        upsample from patch resolution to history-step resolution.

        Returns
        -------
        temporal : list[TemporalAttribution]
        patch_to_step_ratio : float
        """
        if not group_attn_layers:
            return [], 1.0

        first = group_attn_layers[0]
        n_patches = first.shape[0]
        n_groups = first.shape[-1]
        n_covariates = min(len(covariate_names), n_groups - 1)

        if n_covariates == 0 or n_patches == 0:
            return [], 1.0

        patch_to_step_ratio = history_length / n_patches

        # Accumulate mean-over-heads target→covariate scores: (n_patches, n_covariates)
        accumulated = torch.zeros(
            n_patches, n_covariates, device=first.device, dtype=torch.float32
        )
        for layer_attn in group_attn_layers:
            # (Time, Heads, Groups, Groups) → mean over heads → (Time, Groups, Groups)
            avg_heads = layer_attn.float().mean(dim=1)
            # row 0 = Target; cols 1..n_covariates = covariate groups
            accumulated += avg_heads[:n_patches, 0, 1 : n_covariates + 1]

        saliency_matrix = (accumulated / len(group_attn_layers)).cpu().numpy()

        patch_idx = np.arange(n_patches, dtype=float)
        hist_idx = np.linspace(0, n_patches - 1, history_length)

        max_entropy = float(np.log(history_length))
        temporal: list[TemporalAttribution] = []
        for c, name in enumerate(covariate_names[:n_covariates]):
            sal_patch = saliency_matrix[:, c]

            # Upsample patch-resolution saliency to history-step resolution
            sal_hist = np.interp(hist_idx, patch_idx, sal_patch)
            total = sal_hist.sum()
            sal_hist = sal_hist / total if total > 1e-12 else np.full(history_length, 1.0 / history_length)

            peak_step = int(np.argmax(sal_hist))

            p = sal_hist + 1e-10
            p /= p.sum()
            entropy = float(-np.dot(p, np.log(p)))
            focus_breadth = entropy / max_entropy if max_entropy > 0 else 0.0

            temporal.append(TemporalAttribution(
                covariate_name=name,
                saliency=sal_hist,
                peak_step=peak_step,
                focus_breadth=focus_breadth,
            ))

        # Per plan: if cross-group temporal signal is near-uniform for every covariate
        # (entropy > 95% of maximum), it adds noise rather than signal — omit it.
        _ENTROPY_THRESHOLD = 0.95
        if temporal and all(t.focus_breadth > _ENTROPY_THRESHOLD for t in temporal):
            logger.debug(
                "enc_group temporal saliency is near-uniform (min focus_breadth=%.3f > %.2f); "
                "omitting temporal attribution.",
                min(t.focus_breadth for t in temporal),
                _ENTROPY_THRESHOLD,
            )
            return [], patch_to_step_ratio

        return temporal, patch_to_step_ratio

    def _compute_rollout(self, attentions: list[Any]) -> torch.Tensor:
        """Compute Attention Rollout across all layers for the Group dimension.

        Each group attention tensor has shape ``(Time, Heads, Groups, Groups)``.
        Returns a rollout matrix of shape ``(Groups, Groups)``.
        """
        rollout: torch.Tensor | None = None
        for layer_attn in attentions:
            # Average over Time (dim=0) and Heads (dim=1) -> shape: (Groups, Groups)
            avg = layer_attn.mean(dim=(0, 1))
            
            groups = avg.shape[-1]
            identity = torch.eye(groups, device=avg.device, dtype=avg.dtype)
            
            layer_norm = avg + identity
            # Normalize rows to sum to 1
            layer_norm = layer_norm / layer_norm.sum(dim=-1, keepdim=True)
            
            rollout = layer_norm if rollout is None else torch.matmul(rollout, layer_norm)
            
        return rollout if rollout is not None else torch.empty(0)

    def _aggregate_by_covariate(
        self,
        rollout: torch.Tensor,
        covariate_names: list[str],
    ) -> dict[str, float]:
        """Extract target-to-covariate attention scores.

        Index 0 of the rollout matrix is the Target series.
        Indices 1 to len(covariate_names) are the covariates.
        """
        if rollout.numel() == 0:
            return {name: 1.0 / max(1, len(covariate_names)) for name in covariate_names}

        # rollout shape: (Groups, Groups)
        # We want row 0 (Target), starting from column 1 (Covariates)
        target_attention = rollout[0, 1:].cpu().numpy()
        
        result: dict[str, float] = {}
        for i, name in enumerate(covariate_names):
            if i < len(target_attention):
                result[name] = float(target_attention[i])
            else:
                result[name] = 0.0
                
        return result
