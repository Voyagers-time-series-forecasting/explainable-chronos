"""Pipeline — orchestrates the full end-to-end pipeline."""

from __future__ import annotations

import logging
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
from shared.forecast_provider import ChronosForecastProvider, ForecastDict

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete output of a single pipeline run.

    Attributes
    ----------
    forecast : ForecastDict
    features : ForecastFeatures
    attribution : AttributionResult | None
    verbalization : VerbalizationResult
    consistency_report : ConsistencyReport
    """

    forecast: ForecastDict
    features: ForecastFeatures
    attribution: AttributionResult
    verbalization: VerbalizationResult
    consistency_report: ConsistencyReport
    attention_weights: Optional[dict] = None


class VerbalizationPipeline:
    """End-to-end forecast verbalization and consistency pipeline.

    Parameters
    ----------
    forecast_provider : ChronosForecastProvider
    verbalizer : TemplateVerbalizer | LLMVerbalizer
    scorer : NLIConsistencyScorer
    config : PipelineConfig, optional
    """

    def __init__(
        self,
        forecast_provider: ChronosForecastProvider,
        verbalizer: Union[TemplateVerbalizer, LLMVerbalizer],
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
    ) -> PipelineResult:
        """Execute the full pipeline.

        Parameters
        ----------
        time_series : np.ndarray | pd.Series
        horizon : int, optional
        covariates : CovariateSet
            Required. Covariate arrays aligned with the history window.

        Returns
        -------
        PipelineResult
        """
        if covariates is None:
            raise ValueError("covariates is required — univariate mode is not supported.")

        h = horizon or self.config.horizon
        logger.info("Running pipeline with horizon=%d …", h)

        # Stage A — Forecast
        past_cov = {name: covariates.values[:, i] for i, name in enumerate(covariates.names)}

        forecast_result = self.forecast_provider.predict(
            time_series, horizon=h, past_covariates=past_cov,
        )

        if isinstance(forecast_result, tuple):
            forecast, attention_weights = forecast_result
        else:
            forecast, attention_weights = forecast_result, None

        logger.info("Forecast produced (P10/P50/P90 × %d steps).", h)

        # Stage A — Feature extraction
        features = extract_features(forecast, config=self.config)
        logger.info(
            "Features: trend=%s %s, uncertainty=%s %s",
            features.trend_magnitude, features.trend_direction,
            features.uncertainty_level, features.uncertainty_trend,
        )

        # Stage B — Covariate attribution (attention rollout)
        attribution = AttentionAttributor(
            top_k=self.config.attribution_top_k,
        ).explain(covariates, attention_weights=attention_weights)
        logger.info(
            "Attribution: top=%s (%.1f%%)",
            attribution.attributions[0].name if attribution.attributions else "?",
            attribution.attributions[0].relative_impact_pct if attribution.attributions else 0,
        )

        # Stage C — Verbalization
        verbalization = self.verbalizer.verbalize(features, attribution=attribution)
        logger.info("Verbalization: %s", verbalization.summary)

        # Stage D — Consistency scoring
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
        )
