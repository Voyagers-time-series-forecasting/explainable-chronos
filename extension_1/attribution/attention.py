"""
Attention-based covariate attribution using Attention Rollout.

Implements Attention Rollout (Abnar & Zuidema, "Quantifying Attention Flow
in Transformers", 2020) adapted for Chronos-2's two attention axes:
  - enc_group_self_attn_weights  (Patches, Heads, Series, Series):
    cross-series attention at each patch; used for covariate importance.
  - enc_time_self_attn_weights   (Series, Heads, Patches, Patches):
    within-series temporal attention; used for temporal saliency.
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

        # --- Covariate importance (group axis) ---
        # enc_group_self_attn_weights records how each series (target + covariates)
        # attends to every other series at each patch position.
        # Question answered: which covariate influenced the target most?
        group_attn = attentions.get("enc_group_self_attn_weights")
        if not group_attn:
            raise ValueError("No enc_group_self_attn_weights available. Model may not support group attention.")

        # Apply rollout across all group-attention layers to get a
        # (Series x Series) matrix. Row 0 is the target; column c gives
        # how much covariate c contributed to the target's representation.
        rollout = self._compute_rollout(list(group_attn))
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

        # --- Temporal saliency (time axis) ---
        # enc_time_self_attn_weights records how each series attends to its own
        # history patches (within-series, not cross-series).
        # Question answered: which part of the past did each covariate focus on?
        # Shape per layer: (Series, Heads, Patches, Patches). Series dim-0 is
        # the target; dim 1+ are individual covariates — NOT the group axis.
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

        # Rollout is computed independently per covariate: each covariate's
        # time attention is entirely separate from the others.
        # Covariate c occupies slice c+1 of the Series dimension (slice 0 = target).
        rollouts: list[torch.Tensor | None] = [None] * n_covariates
        for layer in time_attn_layers:
            for c in range(n_covariates):
                g = c + 1  # slice 0 = target; covariate c is at slice c+1
                if g >= layer.shape[0]:
                    continue
                # Head-average → (patches, patches)
                avg = layer[g].float().mean(dim=0)
                n_p = avg.shape[-1]
                eye = torch.eye(n_p, device=avg.device, dtype=avg.dtype)
                # Residual adjustment: same formula as in _compute_rollout
                layer_norm = avg + eye
                layer_norm = layer_norm / layer_norm.sum(dim=-1, keepdim=True)
                # Rollout recursion: Ã(lᵢ) = A(lᵢ) · Ã(lᵢ₋₁)
                rollouts[c] = layer_norm if rollouts[c] is None else torch.matmul(layer_norm, rollouts[c])

        # --- Collapse (P x P) rollout to (P,) per covariate ---
        # rollouts[c] is a (patches x patches) matrix where entry [q, p] encodes
        # how much history patch p contributed to current patch q.
        # We want one importance score per history patch, regardless of which
        # current patch was affected, so we average over all query positions (rows).
        saliency_patch = np.zeros((n_covariates, n_patches), dtype=np.float32)
        for c in range(n_covariates):
            if rollouts[c] is not None:
                saliency_patch[c] = rollouts[c].mean(dim=0).cpu().numpy()

        # --- Re-sample from patch space to time-step space ---
        # saliency_patch[c] has one value per patch, but each patch covers multiple
        # consecutive time steps, so this is not yet at the natural time resolution.
        # We spread each patch score across the time steps it covers to produce
        # a saliency vector of length history_length.
        patch_idx = np.arange(n_patches, dtype=float)
        hist_idx  = np.linspace(0, n_patches - 1, history_length)
        max_entropy = float(np.log(history_length))

        temporal: list[TemporalAttribution] = []
        for c, name in enumerate(covariate_names[:n_covariates]):
            sal_patch = saliency_patch[c]

            sal_hist = np.interp(hist_idx, patch_idx, sal_patch)
            total = sal_hist.sum()
            sal_hist = sal_hist / total if total > 1e-12 else np.full(history_length, 1.0 / history_length)

            # Peak step: the single history moment the covariate attended to most.
            peak_step = int(np.argmax(sal_hist))

            # Focus breadth: normalised entropy of the saliency distribution.
            # Low  → attention concentrated on a narrow window of the history.
            # High → attention spread evenly across the whole history.
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

        # If every covariate shows near-uniform attention, the signal is too
        # diffuse to be informative and we suppress temporal saliency entirely.
        if temporal and all(t.focus_breadth > self.temporal_entropy_threshold for t in temporal):
            logger.debug(
                "enc_time temporal saliency is near-uniform (min fb=%.3f); omitting.",
                min(t.focus_breadth for t in temporal),
            )
            return [], patch_to_step_ratio

        return temporal, patch_to_step_ratio

    def _compute_rollout(self, attentions: list[Any]) -> torch.Tensor:
        """Apply Attention Rollout across all group-attention layers.

        Each layer tensor has shape (Patches, Heads, Series, Series).
        Series index 0 is the target; indices 1..C are the covariates.

        Steps per layer:
          1. Average over Patches and Heads → one (Series x Series) matrix.
          2. Residual adjustment: A = 0.5·W + 0.5·I, via (avg + I) row-normalised
             (equivalent because softmax rows sum to 1, so avg + I rows sum to 2).
          3. Rollout recursion: Ã(lᵢ) = A(lᵢ) · Ã(lᵢ₋₁), starting from I.

        The final Ã[i, j] approximates how much series j's input contributed to
        series i's output. Row 0 (target) holds the target's attribution to each
        covariate; normalising those values gives the relative impact percentages.

        Returns a rollout matrix of shape (Series, Series).
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
        Row 0 is the target; columns 1..C are the covariates in input order.
        We read row 0 to get each covariate's raw contribution to the target.
        Normalisation to 100% is done in the caller after top-k selection.
        """
        if rollout.numel() == 0:
            return {name: 1.0 / max(1, len(covariate_names)) for name in covariate_names}

        # Row 0 (target), columns 1+ (covariates)
        target_attention = rollout[0, 1:].cpu().numpy()

        result: dict[str, float] = {}
        for i, name in enumerate(covariate_names):
            result[name] = float(target_attention[i]) if i < len(target_attention) else 0.0

        return result
