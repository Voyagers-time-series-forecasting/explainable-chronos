"""
Attention-based covariate attribution using Attention Rollout.

Implements Attention Rollout (Abnar & Zuidema, "Quantifying Attention Flow
in Transformers", 2020) adapted for Chronos-2's two attention axes:
  - enc_group_self_attn_weights  (Patches, Heads, Series, Series):
    cross-series attention at each patch; used for covariate importance.
  - enc_time_self_attn_weights   (Series, Heads, Patches, Patches):
    within-series temporal attention; used for temporal saliency.

Raw per-layer attention becomes near-uniform in deep layers and is unreliable
as an attribution signal. Rollout composes attention matrices across layers,
accounting for residual connections, to recover input-level attribution.
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
    temporal_entropy_threshold : float
        Normalised entropy above which temporal saliency is considered
        near-uniform and suppressed.
    """

    def __init__(self, top_k: int = 5, temporal_entropy_threshold: float = 0.95) -> None:
        self.top_k = top_k
        self.temporal_entropy_threshold = temporal_entropy_threshold

    def explain(
        self,
        covariates: CovariateSet,
        attention_weights: dict[str, Any],
    ) -> AttributionResult:
        """Compute covariate attributions using Attention Rollout.

        In Chronos-2 the target and its covariates share a group ID and are
        processed as separate batch items (series). ``enc_group_self_attn_weights``
        records how those series attend to each other at every patch position.

        Parameters
        ----------
        covariates : CovariateSet
            Past covariates (history window).
        attention_weights : dict
            ``{"attentions": {"enc_group_self_attn_weights": ...}}``

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

        # Rollout over the series dimension: traces how covariate information
        # routes into the target across all group-attention layers.
        rollout = self._compute_rollout(list(group_attn))

        # Row 0 = target series; columns 1..N = covariates in input order.
        importance = self._aggregate_by_covariate(rollout, list(covariates.names))

        attributions = sorted(
            [
                CovariateAttribution(
                    name=name,
                    importance_score=score,
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

        # Temporal saliency via enc_time_self_attn_weights (per-series temporal
        # self-attention, shape per layer: (series, heads, patches, patches)).
        # dim-0 indexes individual series in the group (0=target, 1+=covariates),
        # NOT the group-as-collection axis.
        # Empirically focus_breadth ≈ 0.77–0.82; enc_group is near-uniform
        # (~0.998, entropy guard active) — see experiments/temporal_attention_probe.ipynb.
        time_attn = attentions.get("enc_time_self_attn_weights")
        history_length = len(covariates.values)
        temporal, patch_ratio = self._compute_temporal_saliency(
            list(time_attn) if time_attn else [], list(covariates.names), history_length
        )

        return AttributionResult(
            attributions=attributions,
            top_k=self.top_k,
            temporal=temporal,
            patch_to_step_ratio=patch_ratio,
        )

    def _compute_temporal_saliency(
        self,
        time_attn_layers: list[Any],
        covariate_names: list[str],
        history_length: int,
    ) -> tuple[list[TemporalAttribution], float]:
        """Compute per-covariate temporal saliency via Attention Rollout over time layers.

        Each layer has shape ``(Series, Heads, Patches, Patches)`` where dim-0
        indexes individual series within the group (0 = target, 1+ = covariates).
        Time attention is within-series, so rollout is computed independently per
        covariate; the residual adjustment and left-multiply recursion follow the
        same paper formula as ``_compute_rollout``.

        Returns
        -------
        temporal : list[TemporalAttribution]
        patch_to_step_ratio : float
        """
        if not time_attn_layers:
            return [], 1.0

        first = time_attn_layers[0]
        # (series, heads, patches, patches)
        n_series  = first.shape[0]
        n_patches = first.shape[2]
        n_covariates = min(len(covariate_names), n_series - 1)

        if n_covariates == 0 or n_patches == 0:
            return [], 1.0

        patch_to_step_ratio = history_length / n_patches

        # Per-covariate rollout through time-attention layers.
        # Each series occupies one independent slice along dim-0.
        rollouts: list[torch.Tensor | None] = [None] * n_covariates
        for layer in time_attn_layers:
            for c in range(n_covariates):
                g = c + 1  # slice 0 = target series; 1+ = covariate series
                if g >= layer.shape[0]:
                    continue
                # Head-average → (patches, patches)
                avg = layer[g].float().mean(dim=0)
                n_p = avg.shape[-1]
                eye = torch.eye(n_p, device=avg.device, dtype=avg.dtype)
                # Residual adjustment (paper §3): (avg + I) row-normalized = 0.5·avg + 0.5·I
                layer_norm = avg + eye
                layer_norm = layer_norm / layer_norm.sum(dim=-1, keepdim=True)
                # Rollout recursion (paper §4): Ã(lᵢ) = A(lᵢ) · Ã(lᵢ₋₁)
                rollouts[c] = layer_norm if rollouts[c] is None else torch.matmul(layer_norm, rollouts[c])

        # Key-patch saliency: mean over query positions of the final rollout matrix.
        saliency_patch = np.zeros((n_covariates, n_patches), dtype=np.float32)
        for c in range(n_covariates):
            if rollouts[c] is not None:
                saliency_patch[c] = rollouts[c].mean(dim=0).cpu().numpy()

        patch_idx = np.arange(n_patches, dtype=float)
        hist_idx  = np.linspace(0, n_patches - 1, history_length)
        max_entropy = float(np.log(history_length))

        temporal: list[TemporalAttribution] = []
        for c, name in enumerate(covariate_names[:n_covariates]):
            sal_patch = saliency_patch[c]

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

        if temporal and all(t.focus_breadth > self.temporal_entropy_threshold for t in temporal):
            logger.debug(
                "enc_time temporal saliency is near-uniform (min fb=%.3f); omitting.",
                min(t.focus_breadth for t in temporal),
            )
            return [], patch_to_step_ratio

        return temporal, patch_to_step_ratio

    def _compute_rollout(self, attentions: list[Any]) -> torch.Tensor:
        """Compute Attention Rollout across all layers for the series dimension.

        Each group-attention tensor has shape ``(Patches, Heads, Series, Series)``
        where the Series dims index individual series members of the group
        (0 = target, 1+ = covariates). Group attention operates on the batch axis:
        at each patch index the series attend to each other. Averaging over patches
        yields one (Series, Series) matrix per layer before rollout.

        Residual adjustment:
            A = 0.5 · W_att + 0.5 · I
        Implemented as (avg + I) row-normalized, which is algebraically equivalent
        since both avg and I have rows summing to 1 (sum = 2, divide by 2 = 0.5 each).

        Rollout recursion:
            Ã(lᵢ) = A(lᵢ) · Ã(lᵢ₋₁)   (new layer left-multiplied)
        So Ã(L) = A(L) · … · A(1), and Ã[0, c] gives how much covariate c's input
        contributed to the target's output representation.

        Returns a rollout matrix of shape ``(Series, Series)``.
        """
        rollout: torch.Tensor | None = None
        for layer_attn in attentions:
            # Average over Patches (dim=0) and Heads (dim=1) → (Series, Series)
            avg = layer_attn.mean(dim=(0, 1))

            n_series = avg.shape[-1]
            identity = torch.eye(n_series, device=avg.device, dtype=avg.dtype)

            # Residual adjustment: (avg + I) row-normalized = 0.5·avg + 0.5·I
            layer_norm = avg + identity
            layer_norm = layer_norm / layer_norm.sum(dim=-1, keepdim=True)

            # Rollout recursion: Ã(lᵢ) = A(lᵢ) · Ã(lᵢ₋₁)
            rollout = layer_norm if rollout is None else torch.matmul(layer_norm, rollout)

        return rollout if rollout is not None else torch.empty(0)

    def _aggregate_by_covariate(
        self,
        rollout: torch.Tensor,
        covariate_names: list[str],
    ) -> dict[str, float]:
        """Extract target-to-covariate importance scores from the rollout matrix.

        rollout[i, j] = how much series j's input contributed to series i's output.
        Row 0 is the target; columns 1..N are covariates in input order.
        """
        if rollout.numel() == 0:
            return {name: 1.0 / max(1, len(covariate_names)) for name in covariate_names}

        # Row 0 (target), columns 1+ (covariates)
        target_attention = rollout[0, 1:].cpu().numpy()

        result: dict[str, float] = {}
        for i, name in enumerate(covariate_names):
            result[name] = float(target_attention[i]) if i < len(target_attention) else 0.0

        return result
