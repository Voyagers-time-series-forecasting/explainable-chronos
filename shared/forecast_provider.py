"""
Shared Forecast Provider.

Wraps the Chronos-2 pipeline (``autogluon/chronos-2-small``) to produce
quantile forecasts (P10 / P50 / P90) from a 1-D historical time series.

This module is shared across all extensions.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

warnings.filterwarnings(
    "ignore", 
    message=".*'pin_memory' argument is set as true but no accelerator is found.*"
)

logger = logging.getLogger(__name__)

ForecastDict = Dict[str, Any]
AttentionWeights = Dict[str, Any]


class ChronosForecastProvider:
    """Wraps chronos.Chronos2Pipeline (autogluon/chronos-2-small).

    This is the single forecast provider for all extensions.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for Chronos-2.
    device : str
        Torch device string (``"cuda"``, ``"cpu"``).
    history_tail_length : int
        Number of trailing history values to include in the result.
    enable_attention : bool
        Whether to enable attention weight extraction for explainability.
    """

    def __init__(
        self,
        model_id: str = "autogluon/chronos-2-small",
        device: str = "cpu",
        history_tail_length: int = 5,
        enable_attention: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.history_tail_length = history_tail_length
        self.enable_attention = enable_attention
        self._pipeline: Chronos2Pipeline | None = None

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return self.model_id

    def _load_pipeline(self) -> Chronos2Pipeline:
        """Load Chronos2Pipeline on first call (cached after)."""
        if self._pipeline is None:
            logger.info(
                "Loading Chronos-2 model %s on %s …",
                self.model_id,
                self.device,
            )
            if self.enable_attention:
                # Load with config that enables attention extraction
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(self.model_id)
                config.attn_implementation = 'eager'
                config.output_attentions = True
                self._pipeline = Chronos2Pipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    config=config,
                )
                logger.info("Attention extraction enabled")
            else:
                self._pipeline = Chronos2Pipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                )
        return self._pipeline

    def predict(
        self,
        history: np.ndarray | pd.Series,
        horizon: int = 14,
        past_covariates: dict | None = None,
        future_covariates: dict | None = None,
    ) -> ForecastDict | tuple[ForecastDict, AttentionWeights]:
        """Produce quantile forecasts (P10, P50, P90).

        Parameters
        ----------
        history : np.ndarray | pd.Series
            1-D historical time-series values.
        horizon : int
            Forecast horizon in steps.
        past_covariates : dict, optional
            Dictionary of past covariate arrays, each of length
            equal to len(history). Keys are covariate names.
        future_covariates : dict, optional
            Dictionary of future covariate arrays, each of length
            equal to horizon. Must be a subset of past_covariates keys.

        Returns
        -------
        ForecastDict or tuple[ForecastDict, AttentionWeights]
            If attention extraction is disabled, returns ForecastDict.
            If enabled, returns (ForecastDict, attention_weights) where
            attention_weights is a dict with keys 'attentions' containing
            the attention tensors from the model.
        """
        pipe = self._load_pipeline()
        history_arr = np.asarray(history, dtype=np.float64)

        # Store attentions if enabled
        captured_attentions = None
        if self.enable_attention:
            # Monkey patch _predict_step to capture attentions
            original_predict_step = pipe._predict_step
            
            def patched_predict_step(context, group_ids, future_covariates, num_output_patches):
                kwargs = {}
                if future_covariates is not None:
                    output_size = num_output_patches * pipe.model_output_patch_size

                    if output_size > future_covariates.shape[1]:
                        batch_size = len(future_covariates)
                        padding_size = output_size - future_covariates.shape[1]
                        padding_tensor = torch.full(
                            (batch_size, padding_size), fill_value=torch.nan, device=future_covariates.device
                        )
                        future_covariates = torch.cat([future_covariates, padding_tensor], dim=1)

                    else:
                        future_covariates = future_covariates[..., :output_size]
                    kwargs["future_covariates"] = future_covariates
                
                nonlocal captured_attentions
                with torch.no_grad():
                    model_output = pipe.model(
                        context=context, group_ids=group_ids, num_output_patches=num_output_patches, 
                        output_attentions=True, **kwargs
                    )
                    prediction = model_output.quantile_preds.to(context)
                    
                    # Capture attentions if available
                    captured_attentions = {
                        'enc_time_self_attn_weights': model_output.enc_time_self_attn_weights,
                        'enc_group_self_attn_weights': model_output.enc_group_self_attn_weights,
                    }
                
                return prediction
            
            pipe._predict_step = patched_predict_step

        # Build input — with or without covariates
        if past_covariates is not None:
            cov_input: dict = {
                "target": torch.tensor(history_arr, dtype=torch.float32),
                "past_covariates": {
                    k: torch.tensor(np.asarray(v, dtype=np.float64), dtype=torch.float32)
                    for k, v in past_covariates.items()
                },
            }
            if future_covariates is not None:
                cov_input["future_covariates"] = {
                    k: torch.tensor(np.asarray(v, dtype=np.float64), dtype=torch.float32)
                    for k, v in future_covariates.items()
                }
            inputs = [cov_input]
            logger.info(
                "Predicting with %d past covariates%s.",
                len(past_covariates),
                f" and {len(future_covariates)} future covariates"
                if future_covariates else "",
            )
        else:
            # Original univariate path — unchanged
            inputs = torch.tensor(
                history_arr, dtype=torch.float32
            ).reshape(1, 1, -1)
            logger.info("Predicting univariate (no covariates).")

        forecast = pipe.predict(inputs, prediction_length=horizon)

        # Handle list-of-tensors or tensor output
        if isinstance(forecast, list):
            forecast_arr = np.stack([
                f.numpy() if hasattr(f, "numpy") else np.asarray(f)
                for f in forecast
            ])
        elif hasattr(forecast, "numpy"):
            forecast_arr = forecast.numpy()
        else:
            forecast_arr = np.asarray(forecast)

        samples = forecast_arr.squeeze()
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        result = {
            "timestamps": list(range(1, horizon + 1)),
            "p10": np.quantile(samples, 0.1, axis=0),
            "p50": np.quantile(samples, 0.5, axis=0),
            "p90": np.quantile(samples, 0.9, axis=0),
            "history_tail": history_arr[-self.history_tail_length:],
        }

        if self.enable_attention:
            return result, {"attentions": captured_attentions}
        else:
            return result
    