import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "extension_1"))

import numpy as np
from shared.forecast_provider import ChronosForecastProvider
from shared.data_generators import generate_demo_time_series, generate_synthetic_covariates
from extension_1.config import PipelineConfig
from extension_1.pipeline import VerbalizationPipeline
from extension_1.verbalizer import TemplateVerbalizer
from extension_1.consistency_scorer import NLIConsistencyScorer

history = generate_demo_time_series(seed=42, length=50)
covariates = generate_synthetic_covariates(history, seed=42)

pipeline = VerbalizationPipeline(
    forecast_provider=ChronosForecastProvider(),
    verbalizer=TemplateVerbalizer(),
    scorer=NLIConsistencyScorer(),
    config=PipelineConfig(),
)

result = pipeline.run(history, covariates=covariates)

print("=== FORECAST ===")
print("p50:", result.forecast["p50"].round(2))

print("\n=== VERBALIZATION ===")
print(result.verbalization.summary)

print("\n=== TOP COVARIATE ===")
top = result.attribution.attributions[0]
print(f"{top.name}: {top.relative_impact_pct:.1f}% ({top.direction})")

print("\n=== CONSISTENCY ===")
print(f"{result.consistency_report.overall_score:.4f} "
      f"({'PASS' if result.consistency_report.is_consistent else 'FAIL'})")