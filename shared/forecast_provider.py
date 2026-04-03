"""
Shared Forecast Provider.

Wraps the Chronos-2 pipeline (``autogluon/chronos-2-small``) to produce
quantile forecasts (P10 / P50 / P90) from a 1-D historical time series.

This module is shared across all extensions.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict

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
    """

    def __init__(
        self,
        model_id: str = "autogluon/chronos-2-small",
        device: str = "cpu",
        history_tail_length: int = 5,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.history_tail_length = history_tail_length
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
            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
            )
        return self._pipeline

    def predict(
        self,
        history: np.ndarray | pd.Series,
        horizon: int = 14,
    ) -> ForecastDict:
        """Produce quantile forecasts (P10, P50, P90).

        Parameters
        ----------
        history : np.ndarray | pd.Series
            1-D historical time-series values.
        horizon : int
            Forecast horizon in steps.

        Returns
        -------
        ForecastDict
            Keys: ``timestamps``, ``p10``, ``p50``, ``p90``, ``history_tail``.
        """
        pipe = self._load_pipeline()
        history_arr = np.asarray(history, dtype=np.float64)
        context = torch.tensor(history_arr, dtype=torch.float32).reshape(1, 1, -1)
        forecast = pipe.predict(context, prediction_length=horizon)

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

        return {
            "timestamps": list(range(1, horizon + 1)),
            "p10": np.quantile(samples, 0.1, axis=0),
            "p50": np.quantile(samples, 0.5, axis=0),
            "p90": np.quantile(samples, 0.9, axis=0),
            "history_tail": history_arr[-self.history_tail_length :],
        }
