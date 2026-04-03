"""
Central runner for all Explainable-Chronos extensions.

Usage::

    python run_extensions.py ext1 demo        # Run Extension 1 demo
    python run_extensions.py ext1 evaluate    # Run Extension 1 evaluation
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from shared.forecast_provider import ChronosForecastProvider


def run_ext1_demo(seed: int = 42) -> None:
    """Run Extension 1 (Forecast Narration) demo."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "extension_1"))

    from extension_1.config import PipelineConfig
    from extension_1.pipeline import VerbalizationPipeline
    from extension_1.verbalizer import TemplateVerbalizer
    from extension_1.consistency_scorer import NLIConsistencyScorer
    from shared.data_generators import generate_demo_time_series


    config = PipelineConfig(seed=seed)
    history = generate_demo_time_series(seed=seed, length=30)

    pipeline = VerbalizationPipeline(
        forecast_provider=ChronosForecastProvider(),
        verbalizer=TemplateVerbalizer(seed=seed),
        scorer=NLIConsistencyScorer(),
        config=config,
    )
    result = pipeline.run(history)

    # Pretty-print
    print("\n" + "=" * 65)
    print("  EXTENSION 1 — FORECAST VERBALIZATION REPORT")
    print("=" * 65)
    print(
        f"\n  Trend      : {result.features.trend_magnitude} "
        f"{result.features.trend_direction} "
        f"(slope={result.features.trend_slope:+.4f})"
    )
    print(
        f"  Uncertainty: {result.features.uncertainty_level} "
        f"({result.features.uncertainty_trend})"
    )
    print(f"  Downside   : {result.features.downside_risk}")
    print(f"  Upside     : {result.features.upside_potential}")
    print(f"  Regime shift: {result.features.regime_shift}")
    print(f"\n  Summary:\n   {result.verbalization.summary}")
    print(
        f"\n  Consistency : {result.consistency_report.overall_score:.4f} "
        f"({'PASS' if result.consistency_report.is_consistent else 'FAIL'})"
    )
    for ss in result.consistency_report.sentence_scores:
        tag = "+" if ss.entailment_prob >= result.consistency_report.threshold else "-"
        print(f"   {tag} [{ss.entailment_prob:.3f}] {ss.sentence[:75]}")
    print()


def run_ext1_evaluate() -> None:
    """Run Extension 1 full evaluation."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "extension_1"))
    from extension_1.evaluation import main as eval_main

    eval_main()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Explainable-Chronos Extension Runner",
    )
    parser.add_argument(
        "extension",
        choices=["ext1"],  # expand as extensions are added
        help="Which extension to run",
    )
    parser.add_argument(
        "action",
        choices=["demo", "evaluate"],
        help="Action to perform",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    dispatch = {
        ("ext1", "demo"): lambda: run_ext1_demo(seed=args.seed),
        ("ext1", "evaluate"): run_ext1_evaluate,
    }
    dispatch[(args.extension, args.action)]()


if __name__ == "__main__":
    main()
