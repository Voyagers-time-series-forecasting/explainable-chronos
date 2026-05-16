"""
Central runner for all Explainable-Chronos extensions.

Colab quick-start::

    !git clone <repo-url> && cd explainable-chronos
    !pip install -e .
    !python run_extensions.py ext1 evaluate --mode dev

Local usage::

    python run_extensions.py ext1 evaluate
    python run_extensions.py ext1 evaluate --dataset etth1 --mode dev --verbalizers template
    python run_extensions.py ext1 evaluate --mode dev --save-traces
    python run_extensions.py ext1 evaluate --verbalizers template llm --judge

    python run_extensions.py ext2 evaluate            # intent-only (fast)
    python run_extensions.py ext2 evaluate-full       # full pipeline with Chronos-2 + NLI
    python run_extensions.py ext2 evaluate --eval-set dev
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "extension_1"))

from extension_1.evaluation.runner import main as evaluation_main


def run_ext1_evaluate(
    datasets: list | None = None,
    mode: str = "dev",
    verbalizers: list | None = None,
    save_traces: bool = False,
    use_judge: bool = False,
    output_dir: str | None = None,
) -> None:
    """Run Extension 1 evaluation on benchmark datasets."""
    evaluation_main(
        dataset_keys=datasets,
        mode_key=mode,
        verbalizer_names=verbalizers,
        save_traces=save_traces,
        use_judge=use_judge,
        output_dir=output_dir,
    )



# ──────────────── Extension 2 ─────────────────────────────────────────

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
        choices=["evaluate", "evaluate-full"],
        help=(
            "Action to perform. "
            "ext1: evaluate (full benchmark). "
            "ext2: evaluate (intent-only, fast) | evaluate-full (Chronos-2 + NLI)."
        ),
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["etth1", "ettm1", "weather", "sp500"],
        default=None,
        help="Real datasets to evaluate on (default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "full"],
        default="dev",
        help="Evaluation mode: dev (5 windows, fast) or full (200 windows, exhaustive)",
    )
    parser.add_argument(
        "--verbalizers",
        nargs="+",
        choices=["template", "llm"],
        default=None,
        help="Verbalizers to use (default: both)",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        default=False,
        help="Save per-scenario trace PNGs to results/extension_1/traces/",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Run LLM-as-judge pairwise comparisons and write judge_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save evaluation results (default: extension_1/results/extension_1)",
    )
    parser.add_argument(
        "--eval-set",
        choices=["dev", "test", "heldout", "blind"],
        default="test",
        help=(
            "Query set for ext2 evaluate. "
            "'test': 30-query set (patterns were tuned on it). "
            "'blind': 10-query set, patterns frozen — final reported score. "
            "'dev': 40-query development set."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    dispatch = {
        ("ext1", "evaluate"): lambda: run_ext1_evaluate(
            datasets=args.dataset,
            mode=args.mode,
            verbalizers=args.verbalizers,
            save_traces=args.save_traces,
            use_judge=args.judge,
            output_dir=args.output_dir,
        ),
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
            f"Action '{args.action}' is not valid for '{args.extension}'. "
            f"Valid combinations: {[f'{e} {a}' for e, a in dispatch]}"
        )
    dispatch[key]()


if __name__ == "__main__":
    main()
