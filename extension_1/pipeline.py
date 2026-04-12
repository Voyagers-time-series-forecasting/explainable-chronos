"""
Module 5 — Pipeline.

Orchestrates the full end-to-end pipeline:

    Forecast → Feature Extraction → [Covariate Attribution] → Verbalization → NLI Consistency Scoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from config import PipelineConfig
from consistency_scorer import ConsistencyReport, NLIConsistencyScorer
from covariate_attribution import AttributionResult, CovariateSet, SurrogateExplainer
from feature_extractor import ForecastFeatures, extract_features
from shared.forecast_provider import ChronosForecastProvider, ForecastDict
from verbalizer import TemplateVerbalizer, VerbalizationResult, LLMVerbalizer

logger = logging.getLogger(__name__)


# ───────────────── result dataclass ───────────────────────────────────
@dataclass
class PipelineResult:
    """Complete output of a single pipeline run.

    Attributes
    ----------
    forecast : ForecastDict
        Raw quantile forecast.
    features : ForecastFeatures
        Extracted interpretable features.
    attribution : AttributionResult | None
        SHAP-based covariate attributions (None when no covariates).
    verbalization : VerbalizationResult
        Natural-language summary with grounding.
    consistency_report : ConsistencyReport
        NLI factual-consistency scores.
    """

    forecast: ForecastDict
    features: ForecastFeatures
    attribution: Optional[AttributionResult]
    verbalization: VerbalizationResult
    consistency_report: ConsistencyReport


# ──────────── pipeline class ──────────────────────────────────────────
class VerbalizationPipeline:
    """End-to-end forecast verbalization and consistency pipeline.

    Parameters
    ----------
    forecast_provider : ChronosForecastProvider
        Chronos-2 forecast provider.
    verbalizer : TemplateVerbalizer
        Sentence planner.
    scorer : NLIConsistencyScorer
        NLI consistency scorer.
    config : PipelineConfig, optional
        Runtime configuration.
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
            Historical time-series values.
        horizon : int, optional
            Forecast steps; falls back to ``config.horizon``.
        covariates : CovariateSet, optional
            Named covariates for SHAP attribution (Stage B).

        Returns
        -------
        PipelineResult
            Aggregated outputs from every stage.
        """
        h = horizon or self.config.horizon
        logger.info("Running pipeline with horizon=%d …", h)

        # Stage A — Forecast
        # Stage A — Forecast
        past_cov = None
        if covariates is not None:
            past_cov = {
                name: covariates.values[:, i]
                for i, name in enumerate(covariates.names)
            }

        forecast = self.forecast_provider.predict(
                time_series,
                horizon=h,
                past_covariates=past_cov,
        )
        logger.info("Forecast produced (P10/P50/P90 × %d steps).", h)

        # Stage A — Feature extraction
        features = extract_features(forecast, config=self.config)
        logger.info(
            "Features: trend=%s %s, uncertainty=%s %s",
            features.trend_magnitude,
            features.trend_direction,
            features.uncertainty_level,
            features.uncertainty_trend,
        )

        # Stage B — Covariate attribution (optional)
        attribution: AttributionResult | None = None
        if covariates is not None:
            explainer = SurrogateExplainer(
                random_state=self.config.seed,
            )
            explainer.fit(covariates, np.asarray(forecast["p50"]))
            attribution = explainer.explain(covariates)
            logger.info(
                "Attribution: R²=%.4f, top=%s (%.1f%%)",
                attribution.surrogate_r2,
                attribution.attributions[0].name if attribution.attributions else "?",
                attribution.attributions[0].relative_impact_pct
                if attribution.attributions
                else 0,
            )

        # Stage C — Verbalization
        verbalization = self.verbalizer.verbalize(features, attribution=attribution)
        logger.info("Verbalization: %s", verbalization.summary)

        # Consistency scoring
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
        )
