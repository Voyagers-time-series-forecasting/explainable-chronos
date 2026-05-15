"""
Central runner for all Explainable-Chronos extensions.

Colab quick-start::

    !git clone <repo-url> && cd explainable-chronos
    !pip install -e .                        # installs all deps from pyproject.toml
    # GPU users: also run the line below for CUDA-enabled PyTorch
    # !pip install torch --index-url https://download.pytorch.org/whl/cu121
    !python run_extensions.py ext1 evaluate --mode dev
    # Or via the installed entry point:
    !explainable-chronos ext1 evaluate --mode dev

Local usage::

    python run_extensions.py ext1 evaluate
    python run_extensions.py ext1 evaluate --dataset etth1 --mode dev --verbalizers template
    python run_extensions.py ext1 evaluate --mode dev --save-traces
    python run_extensions.py ext1 evaluate --mode paper --verbalizers template llm --judge
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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Explainable-Chronos Extension Runner",
    )
    parser.add_argument(
        "extension",
        choices=["ext1"],
        help="Which extension to run",
    )
    parser.add_argument(
        "action",
        choices=["evaluate"],
        help="Action to perform",
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
    }

    dispatch[(args.extension, args.action)]()


if __name__ == "__main__":
    main()
