"""
Extension 2 — Evaluation harness.

Evaluates the three-tier intent parser on predefined query sets.
Metrics: intent accuracy, covariate resolution accuracy, Task Completion Rate (TCR).

Usage::

    python run_extensions.py ext2 evaluate
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from extension_1.attribution.types import CovariateSet
from extension_2.datasets import (
    COVARIATE_NAMES,
    MULTI_TURN_SET,
    TEST_SET,
)
from extension_2.dialogue import DialogueSystem
from extension_2.parsing import IntentParser

# Optional: ETTh1 data loader for full-pipeline evaluation.
try:
    from extension_1.evaluation.runner import DATASET_SPECS, load_dataset_df
    _ETT_AVAILABLE = True
except Exception:
    _ETT_AVAILABLE = False

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent / "eval_results"


# ──────────────── utilities ───────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    """Pin all random sources for reproducible runs."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _save_eval_config(
    out_path: Path,
    seed: int,
    evaluation_set: str,
    run_full_pipeline: bool,
) -> None:
    """Save a JSON snapshot of every parameter that affects the run."""
    cfg: dict = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "evaluation_set": evaluation_set,
        "run_full_pipeline": run_full_pipeline,
        "models": {"chronos": "autogluon/chronos-2-small"},
        "python": sys.version,
        "package_versions": {},
    }

    for pkg in ("torch", "transformers", "sentence-transformers", "chronos"):
        try:
            cfg["package_versions"][pkg] = importlib.metadata.version(pkg)
        except Exception:
            cfg["package_versions"][pkg] = "unknown"

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        cfg["git_commit"] = commit
    except Exception:
        cfg["git_commit"] = "unavailable"

    out_path.mkdir(parents=True, exist_ok=True)
    config_path = out_path / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Run config saved → %s", config_path)


def _make_eval_scenario(
    covariate_names: List[str] = COVARIATE_NAMES,
    n_history: int = 100,
    seed: int = 42,
) -> tuple:
    """Return (history, CovariateSet) for full-pipeline tests.

    Uses real ETTh1 data if available, otherwise falls back to a
    synthetic sine-wave scenario.
    """
    if _ETT_AVAILABLE:
        try:
            spec = DATASET_SPECS["etth1"]
            df = load_dataset_df(spec)
            history = df["OT"].values[-n_history:].astype(np.float64)
            cov_cols = [c for c in df.columns if c not in ("OT", "date")]
            n_real = min(len(cov_cols), len(covariate_names))
            cov_values = df[cov_cols[:n_real]].values[-n_history:].astype(np.float64)
            if n_real < len(covariate_names):
                rng = np.random.default_rng(seed)
                extra = rng.standard_normal((n_history, len(covariate_names) - n_real))
                cov_values = np.concatenate([cov_values, extra], axis=1)
            covariates = CovariateSet(names=covariate_names, values=cov_values)
            logger.info("Full-pipeline scenario: real ETTh1 data (%d steps).", n_history)
            return history, covariates
        except Exception as exc:
            logger.warning("ETTh1 load failed (%s) — falling back to synthetic.", exc)

    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_history)
    history = np.sin(t) + 0.1 * rng.standard_normal(n_history)
    cov_values = rng.standard_normal((n_history, len(covariate_names)))
    return history, CovariateSet(names=covariate_names, values=cov_values)


def _check_delta_faithfulness(delta_p50: Optional[np.ndarray], answer: str) -> Optional[bool]:
    """Return True if the answer text reflects the direction of the forecast delta."""
    if delta_p50 is None:
        return None
    mean_delta = float(np.mean(delta_p50))
    answer_lower = answer.lower()
    if mean_delta > 0.05:
        return any(w in answer_lower for w in ["increas", "higher", "ris", "up", "more"])
    if mean_delta < -0.05:
        return any(w in answer_lower for w in ["decreas", "lower", "fall", "down", "less", "reduc"])
    return None


# ──────────────── evaluator ───────────────────────────────────────────

def run_dialogue_evaluation(
    n_scenarios: int = 5,
    seed: int = 42,
    run_full_pipeline: bool = False,
    evaluation_set: str = "test",
) -> pd.DataFrame:
    """Evaluate the dialogue system on a predefined query set.

    Parameters
    ----------
    n_scenarios : int
        Unused; kept for backward compatibility.
    seed : int
        Random seed.
    run_full_pipeline : bool
        If True, runs Chronos-2 for each query (slow).
        If False, evaluates only intent parsing (fast).
    evaluation_set : str
        One of: "dev", "test", "heldout", "blind".
    """
    _seed_everything(seed)

    parser = IntentParser(
        covariate_names=COVARIATE_NAMES,
        use_bert_tier=True,
        use_llm_fallback=True,
    )
    records: List[Dict[str, Any]] = []
    if evaluation_set.lower() not in ("test",):
        raise ValueError(f"Unknown evaluation_set={evaluation_set!r}; use 'test'.")
    eval_cases = TEST_SET
    set_label = "test"

    if run_full_pipeline:
        history, covariates = _make_eval_scenario(COVARIATE_NAMES, seed=seed)
        system = DialogueSystem(history=history, covariates=covariates, horizon=14, seed=seed)

        for i, tc in enumerate(eval_cases):
            logger.info("[%d/%d] %s", i + 1, len(eval_cases), tc.description)
            try:
                response = system.query(tc.query)
                intent = response.intent
                modification = response.modification
                intent_correct = intent.intent_type == tc.expected_intent
                covariate_correct = tc.expected_covariate is None or intent.target_covariate == tc.expected_covariate
                horizon_correct = tc.expected_horizon is None or intent.new_horizon == tc.expected_horizon
                delta_faithful = _check_delta_faithfulness(response.delta_p50, response.answer)

                records.append({
                    "query": tc.query, "evaluation_set": set_label, "description": tc.description,
                    "expected_intent": tc.expected_intent, "parsed_intent": intent.intent_type,
                    "intent_correct": intent_correct, "expected_covariate": tc.expected_covariate,
                    "parsed_covariate": intent.target_covariate, "covariate_correct": covariate_correct,
                    "expected_horizon": tc.expected_horizon, "parsed_horizon": intent.new_horizon,
                    "horizon_correct": horizon_correct, "modification_applied": modification.modified,
                    "delta_mean": float(np.mean(response.delta_p50)) if response.delta_p50 is not None else None,
                    "delta_faithful": delta_faithful, "task_completed": response.task_completed,
                    "confidence_tier": intent.confidence, "error": None,
                })
            except Exception as e:
                logger.warning("Test case failed: %s — %s", tc.description, e)
                records.append({
                    "query": tc.query, "evaluation_set": set_label, "description": tc.description,
                    "expected_intent": tc.expected_intent, "parsed_intent": "error",
                    "intent_correct": False, "expected_covariate": tc.expected_covariate,
                    "parsed_covariate": None, "covariate_correct": False,
                    "expected_horizon": tc.expected_horizon, "parsed_horizon": None,
                    "horizon_correct": False, "modification_applied": False,
                    "delta_mean": None, "delta_faithful": None, "task_completed": False,
                    "confidence_tier": "error", "error": str(e),
                })
    else:
        for tc in eval_cases:
            intent = parser.parse(tc.query)
            intent_correct = intent.intent_type == tc.expected_intent
            covariate_correct = tc.expected_covariate is None or intent.target_covariate == tc.expected_covariate
            horizon_correct = tc.expected_horizon is None or intent.new_horizon == tc.expected_horizon

            records.append({
                "query": tc.query, "evaluation_set": set_label, "description": tc.description,
                "expected_intent": tc.expected_intent, "parsed_intent": intent.intent_type,
                "intent_correct": intent_correct, "expected_covariate": tc.expected_covariate,
                "parsed_covariate": intent.target_covariate, "covariate_correct": covariate_correct,
                "expected_horizon": tc.expected_horizon, "parsed_horizon": intent.new_horizon,
                "horizon_correct": horizon_correct, "modification_applied": False,
                "delta_mean": None, "delta_faithful": None,
                "task_completed": intent_correct and covariate_correct and horizon_correct,
                "confidence_tier": intent.confidence, "error": None,
            })

    return pd.DataFrame(records)


def print_evaluation_report(df: pd.DataFrame) -> None:
    """Print a summary of the evaluation results."""
    print("\n" + "=" * 65)
    print("  EXTENSION 2 — DIALOGUE EVALUATION REPORT")
    print("=" * 65)

    total = len(df)
    intent_acc = df["intent_correct"].mean() * 100
    cov_mask = df["expected_covariate"].notna()
    cov_acc = df.loc[cov_mask, "covariate_correct"].mean() * 100 if cov_mask.any() else float("nan")
    tcr = df["task_completed"].mean() * 100
    set_label = df["evaluation_set"].iloc[0] if "evaluation_set" in df.columns and total else "unknown"

    print(f"\n  Evaluation set        : {set_label}")
    print(f"  Total test cases      : {total}")
    print(f"  Intent accuracy       : {intent_acc:.1f}%")
    if not pd.isna(cov_acc):
        print(f"  Covariate resolution  : {cov_acc:.1f}%  ({cov_mask.sum()} queries with covariate slot)")
    print(f"  Task completion rate  : {tcr:.1f}%")

    if "delta_faithful" in df.columns and df["delta_faithful"].notna().any():
        delta_verifiable = df["delta_faithful"].notna()
        delta_rate = df.loc[delta_verifiable, "delta_faithful"].mean() * 100
        print(f"  Delta faithfulness    : {delta_rate:.1f}%  ({delta_verifiable.sum()} verifiable)")

    if "confidence_tier" in df.columns:
        print("\n  --- Accuracy by tier (queries handled by each tier) ---")
        for tier in ("rule", "bert", "llm", "fallback", "error"):
            mask = df["confidence_tier"] == tier
            if not mask.any():
                continue
            tier_acc = df.loc[mask, "intent_correct"].mean() * 100
            print(f"  {tier:10s}: {mask.sum():3d} queries, {tier_acc:.1f}% accuracy")

    print("\n  --- Breakdown by intent type ---")
    breakdown = df.groupby("expected_intent")["intent_correct"].agg(["sum", "count"])
    breakdown["accuracy"] = breakdown["sum"] / breakdown["count"] * 100
    print(breakdown[["sum", "count", "accuracy"]].rename(
        columns={"sum": "correct", "count": "total", "accuracy": "acc %"}
    ).to_string(float_format="%.1f"))

    print("\n  --- Failed cases ---")
    failed = df[~df["intent_correct"]]
    if len(failed) == 0:
        print("  None — all intents parsed correctly!")
    else:
        for _, row in failed.iterrows():
            print(f"  x [{row['expected_intent']} -> {row['parsed_intent']}] {row['query'][:60]}")
    print()


def write_evaluation_report(df: pd.DataFrame, save_dir: Path = EVAL_DIR) -> str:
    """Write a markdown report of the evaluation results."""
    report_path = save_dir / "evaluation_report_ext2.md"
    total = len(df)
    set_label = df["evaluation_set"].iloc[0] if "evaluation_set" in df.columns and total else "unknown"
    intent_acc = df["intent_correct"].mean() * 100
    tcr = df["task_completed"].mean() * 100

    lines = [
        "# Extension 2 — Dialogue System Evaluation Report", "",
        "## 1. Overview", "",
        "| Metric | Value |", "|---|---|",
        f"| Evaluation set | {set_label} |",
        f"| Total test cases | {total} |",
        f"| Intent classification accuracy | {intent_acc:.1f}% |",
        f"| Task completion rate | {tcr:.1f}% |",
        "",
        "## 2. Accuracy by Tier", "",
        "| Tier | Queries | Accuracy |", "|---|---|---|",
    ]
    if "confidence_tier" in df.columns:
        for tier in ("rule", "bert", "llm", "fallback", "error"):
            mask = df["confidence_tier"] == tier
            if mask.any():
                tier_acc = df.loc[mask, "intent_correct"].mean() * 100
                lines.append(f"| {tier} | {mask.sum()} | {tier_acc:.1f}% |")
    lines += [
        "",
        "## 3. Breakdown by Intent Type", "",
        "| Intent | Correct | Total | Accuracy |", "|---|---|---|---|",
    ]
    for intent_type, grp in df.groupby("expected_intent"):
        correct = grp["intent_correct"].sum()
        total_g = len(grp)
        lines.append(f"| {intent_type} | {correct} | {total_g} | {correct/total_g*100:.1f}% |")

    lines += ["", "## 4. Failed Cases", ""]
    failed = df[~df["intent_correct"]]
    if len(failed) == 0:
        lines.append("No failures — all intents parsed correctly.")
    else:
        lines += ["| Expected | Parsed | Tier | Query |", "|---|---|---|---|"]
        for _, row in failed.iterrows():
            tier = row.get("confidence_tier", "?")
            lines.append(f"| {row['expected_intent']} | {row['parsed_intent']} | {tier} | {row['query'][:60]} |")

    lines += ["", "## 5. Success Cases", "", f"{len(df[df['intent_correct']])} / {total} queries parsed correctly.", ""]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved to %s", report_path)
    return str(report_path)


def run_multi_turn_evaluation(seed: int = 42) -> pd.DataFrame:
    """Evaluate state persistence across multi-turn dialogue sequences."""
    _seed_everything(seed)
    history, initial_covariates = _make_eval_scenario(COVARIATE_NAMES, seed=seed)
    records: List[Dict[str, Any]] = []

    for seq in MULTI_TURN_SET:
        system = DialogueSystem(history=history, covariates=initial_covariates, horizon=14, seed=seed)
        prev_response: Optional[Any] = None

        for turn_idx, tc in enumerate(seq.turns):
            try:
                response = system.query(tc.query, dry_run=True)
                intent = response.intent
                modification = response.modification
                intent_correct = intent.intent_type == tc.expected_intent
                covariate_correct = tc.expected_covariate is None or intent.target_covariate == tc.expected_covariate
                horizon_correct = tc.expected_horizon is None or intent.new_horizon == tc.expected_horizon

                state_persisted: Optional[bool] = None
                if turn_idx > 0 and prev_response is not None and prev_response.modification.modified:
                    prev_tc = seq.turns[turn_idx - 1]
                    if (
                        prev_tc.expected_intent == "remove_covariate"
                        and prev_tc.expected_covariate is not None
                        and prev_tc.expected_covariate in COVARIATE_NAMES
                    ):
                        cov_idx = COVARIATE_NAMES.index(prev_tc.expected_covariate)
                        state_persisted = bool(np.allclose(system.covariates.values[:, cov_idx], 0.0))

                prev_response = response
                records.append({
                    "sequence": seq.description, "turn": turn_idx + 1, "query": tc.query,
                    "expected_intent": tc.expected_intent, "parsed_intent": intent.intent_type,
                    "intent_correct": intent_correct, "covariate_correct": covariate_correct,
                    "horizon_correct": horizon_correct, "modification_applied": modification.modified,
                    "state_persisted": state_persisted,
                    "task_completed": intent_correct and covariate_correct and horizon_correct,
                    "error": None,
                })
            except Exception as exc:
                records.append({
                    "sequence": seq.description, "turn": turn_idx + 1, "query": tc.query,
                    "expected_intent": tc.expected_intent, "parsed_intent": "error",
                    "intent_correct": False, "covariate_correct": False, "horizon_correct": False,
                    "modification_applied": False, "state_persisted": None,
                    "task_completed": False, "error": str(exc),
                })

    return pd.DataFrame(records)


def print_multi_turn_report(df: pd.DataFrame) -> None:
    """Print a summary of the multi-turn evaluation."""
    print("\n" + "=" * 65)
    print("  EXTENSION 2 — MULTI-TURN STATE PERSISTENCE REPORT")
    print("=" * 65)

    intent_acc = df["intent_correct"].mean() * 100
    state_checks = df["state_persisted"].dropna()
    state_rate = state_checks.mean() * 100 if len(state_checks) > 0 else float("nan")

    print(f"\n  Total turns      : {len(df)}")
    print(f"  Intent accuracy  : {intent_acc:.1f}%")
    if len(state_checks) > 0:
        print(f"  State persisted  : {state_rate:.1f}% ({len(state_checks)} verifiable)")

    print("\n  --- Per-sequence breakdown ---")
    for seq_name, grp in df.groupby("sequence"):
        acc = grp["intent_correct"].mean() * 100
        print(f"  [{acc:.0f}%] {seq_name}")
        for _, row in grp.iterrows():
            tag = "v" if row["intent_correct"] else "x"
            state = ""
            if row["state_persisted"] is True:
                state = " [state OK]"
            elif row["state_persisted"] is False:
                state = " [state LOST]"
            print(f"    Turn {row['turn']} {tag} {row['query'][:55]}{state}")
    print()


def main(
    run_full_pipeline: bool = False,
    seed: int = 42,
    evaluation_set: str = "test",
    output_dir: Optional[Path] = None,
) -> None:
    """Entry point for Extension 2 evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    out = Path(output_dir) if output_dir else EVAL_DIR
    out.mkdir(parents=True, exist_ok=True)
    _save_eval_config(out, seed, evaluation_set, run_full_pipeline)

    logger.info("Running Extension 2 evaluation (set=%s, full_pipeline=%s) ...", evaluation_set, run_full_pipeline)
    df = run_dialogue_evaluation(seed=seed, run_full_pipeline=run_full_pipeline, evaluation_set=evaluation_set)
    df.to_csv(out / "evaluation_results_ext2.csv", index=False)
    print_evaluation_report(df)
    write_evaluation_report(df, save_dir=out)

    logger.info("Running multi-turn state-persistence evaluation ...")
    df_mt = run_multi_turn_evaluation(seed=seed)
    df_mt.to_csv(out / "multi_turn_results_ext2.csv", index=False)
    print_multi_turn_report(df_mt)
