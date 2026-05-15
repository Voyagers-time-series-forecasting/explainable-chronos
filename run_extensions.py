"""
Central runner for all Explainable-Chronos extensions.

Usage::

    python run_extensions.py ext1 demo                     # Extension 1 demo
    python run_extensions.py ext1 demo --covariates        # Demo with covariates
    python run_extensions.py ext1 evaluate                 # Synthetic evaluation
    python run_extensions.py ext1 evaluate-real            # Real dataset evaluation
    python run_extensions.py ext2 demo                     # Extension 2 demo
    python run_extensions.py ext2 evaluate                 # Intent parsing evaluation
    python run_extensions.py ext2 evaluate-full            # Full pipeline evaluation
"""

from __future__ import annotations

import argparse
import logging
import sys

from shared.forecast_provider import ChronosForecastProvider


# ──────────────── Extension 1 ─────────────────────────────────────────

def run_ext1_demo(seed: int = 42, with_covariates: bool = False) -> None:
    """Run Extension 1 demo, optionally with synthetic covariates."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "extension_1"))

    from extension_1.config import PipelineConfig
    from extension_1.pipeline import VerbalizationPipeline
    from extension_1.verbalizer import TemplateVerbalizer
    from extension_1.consistency_scorer import NLIConsistencyScorer
    from shared.data_generators import generate_demo_time_series, generate_synthetic_covariates

    config = PipelineConfig(seed=seed)
    history = generate_demo_time_series(seed=seed, length=50)
    covariates = None

    if with_covariates:
        covariates = generate_synthetic_covariates(history, seed=seed)

    pipeline = VerbalizationPipeline(
        forecast_provider=ChronosForecastProvider(),
        verbalizer=TemplateVerbalizer(seed=seed),
        scorer=NLIConsistencyScorer(),
        config=config,
    )
    result = pipeline.run(history, covariates=covariates)

    title = "WITH COVARIATES" if with_covariates else "UNIVARIATE"
    print("\n" + "=" * 65)
    print(f"  EXTENSION 1 — FORECAST VERBALIZATION REPORT ({title})")
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

    if result.attribution:
        print(f"\n  Top covariates:")
        for attr in result.attribution.attributions[:3]:
            print(f"   - {attr.name}: {attr.relative_impact_pct:.1f}% ({attr.direction})")
        print(f"   Surrogate R²: {result.attribution.surrogate_r2:.4f}")

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
    """Run Extension 1 full evaluation on synthetic scenarios."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "extension_1"))
    from extension_1.evaluation import main as eval_main
    eval_main()


def run_ext1_evaluate_real(
    datasets: list | None = None,
    mode: str = "dev",
    verbalizers: list | None = None,
) -> None:
    """Run Extension 1 evaluation on real datasets."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "extension_1"))
    from extension_1.evaluation_real import main as real_main
    real_main(dataset_keys=datasets, mode_key=mode, verbalizer_names=verbalizers)


# ──────────────── Extension 2 ─────────────────────────────────────────

def run_ext2_demo(seed: int = 42) -> None:
    """Run Extension 2 demo — four example dialogue turns."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "extension_1"))
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "extension_2"))

    from extension_2.dialogue import DialogueSystem
    from shared.data_generators import generate_demo_time_series, generate_synthetic_covariates

    history = generate_demo_time_series(seed=seed, length=50)
    covariates = generate_synthetic_covariates(history, seed=seed)

    system = DialogueSystem(history=history, covariates=covariates, horizon=14, seed=seed)

    queries = [
        "What if we removed the marketing spend covariate?",
        "How confident are you in this forecast?",
        "Show me the next 7 days instead.",
        "What would happen if website traffic increased by 50%?",
    ]

    print("\n" + "=" * 65)
    print("  EXTENSION 2 — DIALOGUE SYSTEM DEMO")
    print("=" * 65)

    for q in queries:
        response = system.query(q)
        print(response.summary())


def run_ext2_evaluate(full_pipeline: bool = False, evaluation_set: str = "heldout") -> None:
    """Run Extension 2 evaluation on the selected query set."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "extension_1"))
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "extension_2"))

    from extension_2.evaluation import main as ext2_eval_main
    ext2_eval_main(run_full_pipeline=full_pipeline, evaluation_set=evaluation_set)


# ──────────────── CLI ─────────────────────────────────────────────────

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Explainable-Chronos Extension Runner",
    )
    parser.add_argument(
        "extension",
        choices=["ext1", "ext2"],
        help="Which extension to run",
    )
    parser.add_argument(
        "action",
        choices=["demo", "demo-cov", "evaluate", "evaluate-real", "evaluate-full"],
        help="Action to perform",
    )
    parser.add_argument(
        "--covariates",
        action="store_true",
        default=False,
        help="Include synthetic covariates in the ext1 demo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["etth1", "ettm1", "m4", "aapl", "sp500"],
        default=None,
        help="Real datasets to evaluate on (ext1 evaluate-real only)",
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "paper", "dev_daily", "paper_daily"],
        default="dev",
        help="Evaluation mode (ext1 evaluate-real only)",
    )
    parser.add_argument(
        "--verbalizers",
        nargs="+",
        choices=["template", "llm_guided", "llm_raw"],
        default=None,
        help="Verbalizers to use (ext1 evaluate-real only)",
    )
    parser.add_argument(
        "--eval-set",
        choices=["heldout", "dev"],
        default="heldout",
        help="Evaluation set to use for ext2 evaluate/evaluate-full",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    dispatch = {
        # Extension 1
        ("ext1", "demo"): lambda: run_ext1_demo(
            seed=args.seed,
            with_covariates=args.covariates,
        ),
        ("ext1", "demo-cov"): lambda: run_ext1_demo(
            seed=args.seed,
            with_covariates=True,
        ),
        ("ext1", "evaluate"): run_ext1_evaluate,
        ("ext1", "evaluate-real"): lambda: run_ext1_evaluate_real(
            datasets=args.dataset,
            mode=args.mode,
            verbalizers=args.verbalizers,
        ),
        # Extension 2
        ("ext2", "demo"): lambda: run_ext2_demo(seed=args.seed),
        ("ext2", "evaluate"): lambda: run_ext2_evaluate(
            full_pipeline=False,
            evaluation_set=args.eval_set,
        ),
        ("ext2", "evaluate-full"): lambda: run_ext2_evaluate(
            full_pipeline=True,
            evaluation_set=args.eval_set,
        ),
    }

    key = (args.extension, args.action)
    if key not in dispatch:
        parser.error(
            f"Action '{args.action}' is not supported for '{args.extension}'. "
            f"Valid combinations: {list(dispatch.keys())}"
        )

    dispatch[key]()


if __name__ == "__main__":
    main()
