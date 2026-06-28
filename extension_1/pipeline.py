"""Orchestrates forecast; features; attribution; verbalization; scoring."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from extension_1.config import PipelineConfig
from extension_1.evaluation.scoring import ConsistencyReport, NLIConsistencyScorer
from extension_1.attribution.attention import AttentionAttributor
from extension_1.attribution.types import AttributionResult, CovariateSet
from extension_1.features.extractor import ForecastFeatures, extract_features
from extension_1.verbalization.template import TemplateVerbalizer, VerbalizationResult
from extension_1.verbalization.llm import LLMVerbalizer
from extension_1.verbalization.fusion import FusionVerbalizer
from shared.forecast_provider import ChronosForecastProvider, ForecastDict

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete output of a single pipeline run."""

    forecast: ForecastDict
    features: ForecastFeatures
    attribution: AttributionResult
    verbalization: VerbalizationResult
    consistency_report: ConsistencyReport
    attention_weights: Optional[dict] = None
    future_covariates: Optional[CovariateSet] = None
    verbalization_time_sec: float = 0.0


class VerbalizationPipeline:
    """End-to-end forecast verbalization and consistency scoring."""

    def __init__(
        self,
        forecast_provider: ChronosForecastProvider,
        verbalizer: Union[TemplateVerbalizer, LLMVerbalizer, FusionVerbalizer],
        scorer: NLIConsistencyScorer,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.forecast_provider = forecast_provider
        self.verbalizer = verbalizer
        self.scorer = scorer
        self.config = config or PipelineConfig()

    def run(
        self,
        time_series: np.ndarray | pd.Series,
        horizon: int | None = None,
        covariates: CovariateSet | None = None,
        future_covariates: CovariateSet | None = None,
    ) -> PipelineResult:
        """Run the full pipeline for a single forecast window."""
        if covariates is None:
            raise ValueError("covariates is required — univariate mode is not supported.")

        h = horizon or self.config.horizon
        logger.info("Running pipeline with horizon=%d …", h)

        # Stage A — Forecast
        past_cov = {name: covariates.values[:, i] for i, name in enumerate(covariates.names)}
        fut_cov = (
            {name: future_covariates.values[:, i] for i, name in enumerate(future_covariates.names)}
            if future_covariates is not None else None
        )

        forecast_result = self.forecast_provider.predict(
            time_series, horizon=h, past_covariates=past_cov, future_covariates=fut_cov,
        )

        if isinstance(forecast_result, tuple):
            forecast, attention_weights = forecast_result
        else:
            forecast, attention_weights = forecast_result, None

        # Stage B — Feature extraction
        features = extract_features(forecast, config=self.config)
        logger.info(
            "Features: trend=%s %s, uncertainty=%s %s",
            features.trend_magnitude, features.trend_direction,
            features.uncertainty_level, features.uncertainty_trend,
        )

        # Stage C — Covariate attribution (attention rollout)
        attribution = AttentionAttributor(
            top_k=self.config.attribution_top_k,
        ).explain(covariates, attention_weights=attention_weights)
        logger.info(
            "Attribution: top=%s (%.1f%%)",
            attribution.attributions[0].name if attribution.attributions else "?",
            attribution.attributions[0].relative_impact_pct if attribution.attributions else 0,
        )

        # Stage D — Verbalization
        verbalization_start = time.perf_counter()
        verbalization = self.verbalizer.verbalize(features, attribution=attribution)
        verbalization_time_sec = time.perf_counter() - verbalization_start
        logger.info(
            "Verbalization (%.3fs): %s", verbalization_time_sec, verbalization.summary
        )

        # Stage E — NLI consistency scoring
        report = self.scorer.score(verbalization)
        logger.info(
            "Consistency: %.4f (%s)",
            report.overall_score,
            "PASS" if report.is_consistent else "FAIL",
        )

        return PipelineResult(
            forecast=forecast,
            features=features,
            attribution=attribution,
            verbalization=verbalization,
            consistency_report=report,
            attention_weights=attention_weights,
            future_covariates=future_covariates,
            verbalization_time_sec=verbalization_time_sec,
        )
