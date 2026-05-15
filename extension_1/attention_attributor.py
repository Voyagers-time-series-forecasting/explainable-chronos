"""
Attention-based covariate attribution using Attention Rollout.

This module implements attention-based explanation of Chronos-2 forecasts
by aggregating attention weights across transformer layers using the
Attention Rollout technique.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import torch

from extension_1.covariate_attribution import AttributionResult, CovariateAttribution

logger = logging.getLogger(__name__)


class AttentionAttributor:
    """Attributor using Attention Rollout for covariate importance.
    
    Implements the Attention Rollout method from "Quantifying Attention Flow
    in Transformers" to aggregate attention weights across layers and heads,
    providing more reliable explanations than raw attention summation.
    """
    
    def __init__(self, top_k: int = 5) -> None:
        """Initialize the attention attributor.
        
        Parameters
        ----------
        top_k : int
            Maximum number of top attributions to return.
        """
        self.top_k = top_k
    
    def explain(
        self, 
        covariates: Any,  # CovariateSet
        attention_weights: Dict[str, Any]
    ) -> AttributionResult:
        """Compute covariate attributions using Attention Rollout.
        
        Parameters
        ----------
        covariates : CovariateSet
            The covariate data with names, values, and descriptions.
        attention_weights : dict
            Attention weights extracted from Chronos-2 model.
            
        Returns
        -------
        AttributionResult
            Attribution results with rollout-based importance scores.
        """
        if attention_weights is None or 'attentions' not in attention_weights:
            raise ValueError("Attention weights not available. Ensure forecast_provider has enable_attention=True")
        
        attentions = attention_weights['attentions']
        if attentions is None:
            raise ValueError("No attention weights captured from model")
        
        # Use time self-attention weights for rollout (more relevant for temporal covariates)
        time_attentions = attentions.get('enc_time_self_attn_weights')
        if time_attentions is None or len(time_attentions) == 0:
            raise ValueError("No time self-attention weights available")
        
        # Convert tuple to list for processing
        attentions_list = list(time_attentions)
        
        # Apply Attention Rollout
        rollout_attention = self._compute_attention_rollout(attentions_list)
        
        # Aggregate attention by covariate
        covariate_importance = self._aggregate_by_covariate(
            rollout_attention, covariates
        )
        
        # Create attribution objects
        attributions = []
        for name, importance in covariate_importance.items():
            # Determine direction (simplified - could be more sophisticated)
            direction = "positive" if importance >= 0 else "negative"
            relative_impact_pct = abs(importance) * 100
            
            attributions.append(CovariateAttribution(
                name=name,
                shap_value=importance,  # Using importance as shap_value for compatibility
                direction=direction,
                relative_impact_pct=relative_impact_pct
            ))
        
        # Sort by absolute importance
        attributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
        attributions = attributions[:self.top_k]
        
        # Normalize relative impact percentages
        if attributions:
            total_impact = sum(abs(attr.shap_value) for attr in attributions)
            if total_impact > 0:
                for attr in attributions:
                    attr.relative_impact_pct = (abs(attr.shap_value) / total_impact) * 100
        
        return AttributionResult(
            attributions=attributions,
            surrogate_r2=1.0,  # Not applicable for attention-based method
            top_k=self.top_k
        )
    
    def _compute_attention_rollout(self, attentions: list) -> torch.Tensor:
        """Compute Attention Rollout across all layers.
        
        Parameters
        ----------
        attentions : list
            List of attention tensors, one per layer.
            Each tensor has shape (batch, heads, seq_len, seq_len)
            
        Returns
        -------
        torch.Tensor
            Rollout attention matrix of shape (seq_len, seq_len)
        """
        rollout = None
        
        for layer_attention in attentions:
            # Average across heads: (batch, heads, seq_len, seq_len) -> (batch, seq_len, seq_len)
            layer_avg = layer_attention.mean(dim=1)
            
            # Add identity matrix for residual connection: A' = A + I
            seq_len = layer_avg.shape[-1]
            identity = torch.eye(seq_len, device=layer_avg.device, dtype=layer_avg.dtype)
            identity = identity.unsqueeze(0).expand_as(layer_avg)
            layer_with_residual = layer_avg + identity
            
            # Normalize rows to sum to 1
            layer_normalized = layer_with_residual / layer_with_residual.sum(dim=-1, keepdim=True)
            
            # Multiply with previous rollout
            if rollout is None:
                rollout = layer_normalized
            else:
                rollout = torch.matmul(rollout, layer_normalized)
        
        # Return the final rollout matrix (squeeze batch dimension)
        return rollout.squeeze(0) if rollout is not None else torch.empty(0)
    
    def _aggregate_by_covariate(
        self, 
        rollout_attention: torch.Tensor, 
        covariates: Any  # CovariateSet
    ) -> Dict[str, float]:
        """Aggregate rollout attention by covariate groups.
        
        This is a simplified implementation that assumes covariates are
        represented in the attention matrix in some systematic way.
        In practice, this would need to be calibrated based on how
        Chronos-2 tokenizes and positions covariates in the sequence.
        
        Parameters
        ----------
        rollout_attention : torch.Tensor
            Attention rollout matrix of shape (seq_len, seq_len)
        covariates : CovariateSet
            Covariate information
            
        Returns
        -------
        Dict[str, float]
            Mapping from covariate name to importance score
        """
        covariate_importance = {}
        
        if rollout_attention.numel() == 0:
            # Fallback: equal importance
            base_importance = 1.0 / len(covariates.names)
            for name in covariates.names:
                covariate_importance[name] = base_importance
            return covariate_importance
        
        seq_len = rollout_attention.shape[-1]
        num_covariates = len(covariates.names)
        
        # Simplified assumption: covariates are represented in the latter
        # part of the sequence. We'll divide the sequence into roughly
        # equal segments for each covariate.
        if num_covariates > 0 and seq_len >= num_covariates:
            # Calculate segment size for each covariate
            segment_size = seq_len // num_covariates
            remainder = seq_len % num_covariates
            
            start_pos = 0
            for i, name in enumerate(covariates.names):
                # Give extra positions to earlier covariates if there's remainder
                extra = 1 if i < remainder else 0
                end_pos = start_pos + segment_size + extra
                
                # Sum attention flowing to this covariate's segment
                segment_attention = rollout_attention[:, start_pos:end_pos]
                importance = segment_attention.sum().item()
                
                covariate_importance[name] = importance
                start_pos = end_pos
        else:
            # Fallback for edge cases
            base_importance = rollout_attention.sum().item() / num_covariates
            for name in covariates.names:
                covariate_importance[name] = base_importance
        
        return covariate_importance