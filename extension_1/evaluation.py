"""
Module 6 — Evaluation.

Systematically evaluates the verbalization pipeline across many
synthetic time-series scenarios using real Chronos-2 forecasts and
NLI consistency scoring.

Supports two ground truths:
1. Explanation Faithfulness (NLI consistency + quantile round-trip)
2. Forecast Accuracy (model vs. synthetic actuals)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI / headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    ASYMMETRY_THRESHOLD,
    EVAL_NUM_SCENARIOS,
    HIGH_UNCERTAINTY_THRESHOLD,
    LOW_UNCERTAINTY_THRESHOLD,
    MODERATE_THRESHOLD,
    SHARP_THRESHOLD,
    PipelineConfig,
    RANDOM_SEED,
)
from consistency_scorer import NLIConsistencyScorer
from covariate_attribution import CovariateSet
from feature_extractor import extract_features
from pipeline import PipelineResult, VerbalizationPipeline
from shared.forecast_provider import ChronosForecastProvider
from shared.data_generators import generate_scenarios, generate_synthetic_covariates
from verbalizer import TemplateVerbalizer, LLMVerbalizer

logger = logging.getLogger(__name__)

# ──────────────── output directory ────────────────────────────────────
EVAL_DIR = Path(__file__).resolve().parent / "eval_results"


# ──────────── quantile faithfulness check ─────────────────────────────
def check_faithfulness(
    features: Any,
    forecast: Dict[str, Any],
    config: PipelineConfig,
) -> List[str]:
    """Re-derive feature labels from raw quantiles and check for mismatches.

    Returns
    -------
    list[str]
        List of mismatch descriptions (empty = fully faithful).
    """
    errors: List[str] = []
    p50 = np.asarray(forecast["p50"], dtype=np.float64)
    p10 = np.asarray(forecast["p10"], dtype=np.float64)
    p90 = np.asarray(forecast["p90"], dtype=np.float64)
    horizon = len(p50)

    # Re-derive trend
    steps = np.arange(horizon, dtype=np.float64)
    slope, _ = np.polyfit(steps, p50, deg=1)
    mean_abs = float(np.mean(np.abs(p50)))
    norm_slope = slope / mean_abs if mean_abs > 1e-9 else 0.0
    abs_norm = abs(norm_slope)

    if abs_norm > config.sharp_threshold:
        expected_mag = "sharply"
    elif abs_norm > config.moderate_threshold:
        expected_mag = "moderately"
    else:
        expected_mag = "slightly"

    if abs_norm <= config.moderate_threshold:
        expected_dir = "flat"
    elif norm_slope > 0:
        expected_dir = "rising"
    else:
        expected_dir = "falling"

    if features.trend_direction != expected_dir:
        errors.append(f"trend_direction: got {features.trend_direction}, expected {expected_dir}")
    if features.trend_magnitude != expected_mag:
        errors.append(f"trend_magnitude: got {features.trend_magnitude}, expected {expected_mag}")

    # Re-derive uncertainty
    widths = p90 - p10
    rel_unc = float(np.mean(widths)) / mean_abs if mean_abs > 1e-9 else 0.0
    if rel_unc > config.high_uncertainty:
        expected_unc = "high"
    elif rel_unc > config.low_uncertainty:
        expected_unc = "moderate"
    else:
        expected_unc = "low"

    if features.uncertainty_level != expected_unc:
        errors.append(f"uncertainty_level: got {features.uncertainty_level}, expected {expected_unc}")

    return errors


# ──────────── forecast accuracy metrics ───────────────────────────────
def compute_forecast_accuracy(
    forecast: Dict[str, Any],
    actuals: np.ndarray,
    history: np.ndarray,
) -> Dict[str, float]:
    """Compute forecast accuracy against synthetic actuals.

    The returned metrics are:
    - ``mase``: the traditional mean absolute scaled error using history-based
      one-step differences as the baseline.
    - ``mase_first``: the first-step version of MASE, which evaluates only the
      first forecasted point and is useful for one-step forecast quality.
    - ``fair_mase``: relative mean absolute error compared to a multi-step naive
      forecast that repeats the last observed input value for the entire
      forecast horizon.
    - ``coverage_pct``: percent of actual values inside the model's P10-P90
      prediction interval.
    - ``interval_sharpness``: the average interval width divided by the actuals'
      value range.

    ``fair_mase`` is included because long-horizon forecasts should be judged
    against a full-horizon naive baseline, not only against one-step changes
    in the history.
    """
    p10 = np.asarray(forecast["p10"])
    p50 = np.asarray(forecast["p50"])
    p90 = np.asarray(forecast["p90"])
    actuals = np.asarray(actuals)[: len(p50)]

    forecast_mae = float(np.mean(np.abs(p50 - actuals)))
    forecast_first_mae = float(np.abs(p50[0] - actuals[0])) if len(p50) > 0 and len(actuals) > 0 else 0.0

    # Legacy baseline: one-step changes from history.
    naive_errors = np.abs(np.diff(history))
    naive_mae = float(np.mean(naive_errors)) if len(naive_errors) > 0 else 1.0
    mase = forecast_mae / naive_mae if naive_mae > 1e-9 else 0.0
    mase_first = forecast_first_mae / naive_mae if naive_mae > 1e-9 else 0.0

    # Relative MAE vs multi-step naive forecast (repeat last history value).
    if len(actuals) > 0:
        naive_multi_errors = np.abs(actuals - history[-1])
        naive_multi_mae = float(np.mean(naive_multi_errors))
    else:
        naive_multi_mae = 1.0
    fair_mase = forecast_mae / naive_multi_mae if naive_multi_mae > 1e-9 else 0.0

    # Coverage: % of actuals within P10-P90
    in_interval = (actuals >= p10) & (actuals <= p90)
    coverage = float(np.mean(in_interval)) * 100

    # Interval sharpness: mean width / range of actuals
    mean_width = float(np.mean(p90 - p10))
    actual_range = float(np.ptp(actuals)) if np.ptp(actuals) > 1e-9 else 1.0
    sharpness = mean_width / actual_range

    return {
        "mase": mase,
        "mase_first": mase_first,
        "fair_mase": fair_mase,
        "coverage_pct": coverage,
        "interval_sharpness": sharpness,
    }


# ──────────────── evaluation runner ───────────────────────────────────
def run_evaluation(
    n: int = EVAL_NUM_SCENARIOS,
    seed: int = RANDOM_SEED,
    use_covariates: bool = True,
) -> pd.DataFrame:
    """Run the pipeline on *n* synthetic scenarios and tabulate results.

    Parameters
    ----------
    n : int
        Number of scenarios.
    seed : int
        Global seed.
    use_covariates : bool
        Whether to include synthetic covariates.

    Returns
    -------
    pd.DataFrame
        One row per scenario with features, scores, and accuracy metrics.
    """
    config = PipelineConfig(seed=seed)
    scenarios = generate_scenarios(n=n, seed=seed)

    provider = ChronosForecastProvider()
    scorer = NLIConsistencyScorer()

    # FIX 1: create separate TemplateVerbalizer instances for each pipeline
    # to avoid shared mutable state between LLMVerbalizer instances.
    template_verbalizer = TemplateVerbalizer(seed=seed)

    try:
        llm_guided = LLMVerbalizer(
            template_verbalizer=TemplateVerbalizer(seed=seed),
        )
        llm_raw = LLMVerbalizer(
            template_verbalizer=TemplateVerbalizer(seed=seed),
        )

        pipe_template = VerbalizationPipeline(
            forecast_provider=provider,
            verbalizer=template_verbalizer,
            scorer=scorer,
            config=config,
        )
        pipe_llm_guided = VerbalizationPipeline(
            forecast_provider=provider,
            verbalizer=llm_guided,
            scorer=scorer,
            config=config,
        )
        pipe_llm_raw = VerbalizationPipeline(
            forecast_provider=provider,
            verbalizer=llm_raw,
            scorer=scorer,
            config=config,
        )
        pipelines_to_run = [
            ("Template", pipe_template),
            ("LLM Guided", pipe_llm_guided),
            ("LLM Raw", pipe_llm_raw),
        ]
    except Exception as e:
        logger.warning("Could not load LLMs: %s — running Template only.", e)
        pipe_template = VerbalizationPipeline(
            forecast_provider=provider,
            verbalizer=template_verbalizer,
            scorer=scorer,
            config=config,
        )
        pipelines_to_run = [("Template", pipe_template)]

    records: list[dict] = []

    for idx, (label, history, future, per_seed) in enumerate(scenarios):
        covariates = None
        if use_covariates:
            covariates = generate_synthetic_covariates(
                history, n_covariates=10, seed=per_seed,
            )

        for v_type, pipe in pipelines_to_run:
            result = pipe.run(history, horizon=len(future), covariates=covariates)

            # Per-sentence scores
            sent_scores = [
                s.entailment_prob for s in result.consistency_report.sentence_scores
            ]
            min_sent = min(sent_scores) if sent_scores else 0.0
            max_sent = max(sent_scores) if sent_scores else 0.0

            # Faithfulness check
            faith_errors = check_faithfulness(result.features, result.forecast, config)

            # Forecast accuracy
            accuracy = compute_forecast_accuracy(result.forecast, future, history)

            record: Dict[str, Any] = {
                "scenario_idx": idx,
                "scenario_type": label,
                "seed": per_seed,
                "verbalizer_type": v_type,
                "trend_direction": result.features.trend_direction,
                "trend_magnitude": result.features.trend_magnitude,
                "uncertainty_level": result.features.uncertainty_level,
                "uncertainty_trend": result.features.uncertainty_trend,
                "interval_asymmetry": result.features.interval_asymmetry,
                "asymmetry_label": result.features.asymmetry_label,
                "downside_risk": result.features.downside_risk,
                "upside_potential": result.features.upside_potential,
                "regime_shift": result.features.regime_shift,
                "overall_consistency": result.consistency_report.overall_score,
                "is_consistent": result.consistency_report.is_consistent,
                "num_sentences": len(result.verbalization.sentences),
                "verbalization_summary": result.verbalization.summary,
                "rst_relations": ",".join(result.verbalization.rst_relations),
                "faithfulness_errors": "; ".join(faith_errors) if faith_errors else "",
                "mase": accuracy["mase"],
                "mase_first": accuracy["mase_first"],
                "fair_mase": accuracy["fair_mase"],
                "coverage_pct": accuracy["coverage_pct"],
                "interval_sharpness": accuracy["interval_sharpness"],
            }

            # Per-sentence scores
            for si, ss in enumerate(result.consistency_report.sentence_scores):
                record[f"sent_{si}_score"] = ss.entailment_prob

            # Attribution info
            if result.attribution:
                record["surrogate_r2"] = result.attribution.surrogate_r2
                if result.attribution.attributions:
                    top = result.attribution.attributions[0]
                    record["top_covariate"] = top.name
                    record["top_covariate_impact_pct"] = top.relative_impact_pct
            else:
                record["surrogate_r2"] = None
                record["top_covariate"] = None
                record["top_covariate_impact_pct"] = None

            records.append(record)
            logger.info(
                "[%d/%d] [%s] %s  consistency=%.4f  mase=%.4f  fair_mase=%.4f  coverage=%.1f%%",
                idx + 1, n, v_type, label,
                result.consistency_report.overall_score,
                accuracy["mase"],
                accuracy["fair_mase"],
                accuracy["coverage_pct"],
            )

    return pd.DataFrame(records)


# ══════════════════ reporting & visualization ═════════════════════════
def plot_results(df: pd.DataFrame, save_dir: Path = EVAL_DIR) -> str:
    """Generate evaluation visualizations.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``run_evaluation``.
    save_dir : Path
        Directory for saved figure.

    Returns
    -------
    str
        Path to the saved figure.
    """
    save_path = save_dir / "evaluation_plots.png"
    has_covariates = "surrogate_r2" in df.columns and df["surrogate_r2"].notna().any()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    scenario_types = sorted(df["scenario_type"].unique())
    scenario_colors = {"trending": "#4c72b0", "volatile": "#dd8452", "flat": "#55a868"}

    # FIX 2: Plot 1 — boxplot by verbalizer_type, not scenario_type.
    # This correctly separates the three pipeline approaches.
    v_types = sorted(df["verbalizer_type"].unique())
    v_colors = {"Template": "#4c72b0", "LLM Guided": "#dd8452", "LLM Raw": "#55a868"}
    data_groups = [
        df.loc[df["verbalizer_type"] == vt, "overall_consistency"].to_numpy()
        for vt in v_types
    ]
    bp = axes[0].boxplot(data_groups, tick_labels=v_types, patch_artist=True)
    for patch, vt in zip(bp["boxes"], v_types):
        patch.set_facecolor(v_colors.get(vt, "grey"))
        patch.set_alpha(0.7)
    axes[0].set_title("Consistency by Verbalizer Type")
    axes[0].set_ylabel("Entailment Score")
    axes[0].axhline(0.7, color="red", linestyle="--", alpha=0.5, label="threshold (0.70)")
    axes[0].legend()

    # Plot 2 — scatter: scenario_idx vs consistency, coloured by verbalizer_type
    for vt in v_types:
        subset = df[df["verbalizer_type"] == vt]
        axes[1].scatter(
            subset["scenario_idx"], subset["overall_consistency"],
            label=vt, alpha=0.7, color=v_colors.get(vt, "grey"),
            edgecolors="w", linewidth=0.5,
        )
    axes[1].axhline(0.7, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("Consistency per Scenario (by Verbalizer)")
    axes[1].set_xlabel("Scenario Index")
    axes[1].set_ylabel("Consistency Score")
    axes[1].legend()

    # Plot 3 — histogram: per-sentence entailment scores
    sent_cols = [c for c in df.columns if c.startswith("sent_") and c.endswith("_score")]
    all_sent_scores = []
    for col in sent_cols:
        all_sent_scores.extend(df[col].dropna().to_numpy().tolist())
    if all_sent_scores:
        axes[2].hist(
            all_sent_scores, bins=np.arange(0, 1.05, 0.05),
            color="#4c72b0", alpha=0.7, edgecolor="white",
        )
        mean_s = float(np.mean(all_sent_scores))
        med_s = float(np.median(all_sent_scores))
        axes[2].axvline(mean_s, color="red", linestyle="-", label=f"mean={mean_s:.2f}")
        axes[2].axvline(med_s, color="orange", linestyle="--", label=f"median={med_s:.2f}")
        axes[2].legend()
    axes[2].set_title("Per-Sentence Entailment Distribution")
    axes[2].set_xlabel("Entailment Probability")
    axes[2].set_ylabel("Count")

    # Plot 4 — bar: feature value distribution
    feat_cols = ["trend_direction", "uncertainty_level", "asymmetry_label"]
    all_cats: Dict[str, int] = {}
    for col in feat_cols:
        if col in df.columns:
            for val, cnt in df[col].value_counts().items():
                all_cats[f"{col}:{val}"] = int(cnt)
    if all_cats:
        bars = list(all_cats.keys())
        vals = list(all_cats.values())
        axes[3].barh(bars, vals, color="#55a868", alpha=0.7)
        axes[3].set_title("Feature Value Distribution")
        axes[3].set_xlabel("Count")

    # Plot 5 — scatter: coverage % vs consistency, coloured by scenario type
    if "coverage_pct" in df.columns:
        for st in scenario_types:
            subset = df[df["scenario_type"] == st]
            axes[4].scatter(
                subset["coverage_pct"], subset["overall_consistency"],
                label=st, alpha=0.7, color=scenario_colors.get(st, "grey"),
                edgecolors="w", linewidth=0.5,
            )
        axes[4].axhline(0.7, color="red", linestyle="--", alpha=0.5)
        axes[4].set_title("Coverage vs Consistency (by Scenario Type)")
        axes[4].set_xlabel("Coverage %")
        axes[4].set_ylabel("Consistency Score")
        axes[4].legend()

    # Plot 6 — bar: mean top covariate impact (only if covariates present)
    if has_covariates:
        cov_df = df[df["top_covariate"].notna()]
        if not cov_df.empty:
            mean_impact = (
                cov_df.groupby("top_covariate")["top_covariate_impact_pct"]
                .mean()
                .sort_values(ascending=False)
            )
            axes[5].bar(mean_impact.index, mean_impact.values, color="#dd8452", alpha=0.7)
            axes[5].set_title("Mean Top Covariate Impact")
            axes[5].set_xlabel("Covariate")
            axes[5].set_ylabel("Relative Impact %")
            axes[5].tick_params(axis="x", rotation=30)
    else:
        axes[5].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    logger.info("Plots saved to %s", save_path)
    plt.close(fig)
    return str(save_path)


def write_report(df: pd.DataFrame, save_dir: Path = EVAL_DIR) -> str:
    """Write comprehensive markdown evaluation report.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``run_evaluation``.
    save_dir : Path
        Directory for the report.

    Returns
    -------
    str
        Path to the saved report.
    """
    report_path = save_dir / "evaluation_report.md"

    n = len(df)
    mean_sc = df["overall_consistency"].mean()
    std_sc = df["overall_consistency"].std()
    min_sc = df["overall_consistency"].min()
    max_sc = df["overall_consistency"].max()
    pct_consistent = df["is_consistent"].mean() * 100

    lines: list[str] = []

    # ── 1. Overview ───────────────────────────────────────────────
    lines.append("# Evaluation Report — Extension 1")
    lines.append("")
    lines.append(
        "> This evaluation separates two questions: "
        "(1) Is the explanation faithful to the forecast? "
        "(2) Is the forecast itself accurate? "
        "A high-scoring explanation of an inaccurate forecast "
        "is still a correct explanation."
    )
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Scenarios evaluated | {n} |")
    lines.append(f"| Forecast model | `autogluon/chronos-2-small` |")
    lines.append(f"| NLI model | `facebook/bart-large-mnli` |")
    lines.append(f"| Consistency threshold | 0.70 |")
    lines.append(f"| Mean consistency | {mean_sc:.4f} |")
    lines.append(f"| Std consistency | {std_sc:.4f} |")
    lines.append(f"| Min consistency | {min_sc:.4f} |")
    lines.append(f"| Max consistency | {max_sc:.4f} |")
    lines.append(f"| % consistent (≥ 0.7) | {pct_consistent:.1f}% |")
    lines.append(f"| Mean MASE | {df['mase'].mean():.4f} |")
    lines.append(f"| Mean First-Step MASE | {df['mase_first'].mean():.4f} |")
    if "fair_mase" in df.columns:
        lines.append(f"| Mean fair_mase | {df['fair_mase'].mean():.4f} |")
    lines.append("")

    # ── 2. Breakdown by verbalizer type ───────────────────────────
    lines.append("## 2. Breakdown by Verbalizer Type")
    lines.append("")
    grouped_v = df.groupby("verbalizer_type")["overall_consistency"].agg(
        ["mean", "std", "min", "max", "count"]
    )
    lines.append("| Verbalizer | Mean | Std | Min | Max | Count |")
    lines.append("|---|---|---|---|---|---|")
    for vtype, row in grouped_v.iterrows():
        lines.append(
            f"| {vtype} | {row['mean']:.4f} | {row['std']:.4f} "
            f"| {row['min']:.4f} | {row['max']:.4f} | {int(row['count'])} |"
        )
    lines.append("")

    # ── 3. Breakdown by scenario type ─────────────────────────────
    lines.append("## 3. Breakdown by Scenario Type")
    lines.append("")
    grouped_s = df.groupby("scenario_type")["overall_consistency"].agg(
        ["mean", "std", "min", "max", "count"]
    )
    lines.append("| Scenario | Mean | Std | Min | Max | Count |")
    lines.append("|---|---|---|---|---|---|")
    for stype, row in grouped_s.iterrows():
        lines.append(
            f"| {stype} | {row['mean']:.4f} | {row['std']:.4f} "
            f"| {row['min']:.4f} | {row['max']:.4f} | {int(row['count'])} |"
        )
    lines.append("")

    # ── 4. Feature distribution ───────────────────────────────────
    lines.append("## 4. Feature Distribution")
    lines.append("")
    for col in ["trend_direction", "trend_magnitude", "uncertainty_level",
                "uncertainty_trend", "asymmetry_label"]:
        if col in df.columns:
            counts = df[col].value_counts()
            lines.append(f"**{col}**: " + ", ".join(
                f"{v} ({c})" for v, c in counts.items()
            ))
    lines.append("")
    lines.append(f"- Downside risk flagged: {df['downside_risk'].sum()} / {n}")
    lines.append(f"- Upside potential flagged: {df['upside_potential'].sum()} / {n}")
    lines.append(f"- Regime shift detected: {df['regime_shift'].sum()} / {n}")
    lines.append("")

    # ── 5. Covariate attribution summary ──────────────────────────
    if "surrogate_r2" in df.columns and df["surrogate_r2"].notna().any():
        lines.append("## 5. Covariate Attribution Summary")
        lines.append("")
        cov_df = df[df["surrogate_r2"].notna()]
        lines.append(f"- Mean surrogate R²: {cov_df['surrogate_r2'].mean():.4f}")
        if "top_covariate" in cov_df.columns:
            top_counts = cov_df["top_covariate"].value_counts()
            lines.append(
                "- Top covariate distribution: " + ", ".join(
                    f"{v} ({c})" for v, c in top_counts.items()
                )
            )
            if "top_covariate_impact_pct" in cov_df.columns:
                lines.append(
                    f"- Mean top covariate impact: "
                    f"{cov_df['top_covariate_impact_pct'].mean():.1f}%"
                )
        lines.append("")

    # ── 6. Explanation Faithfulness ────────────────────────────────
    lines.append("## 6. Explanation Faithfulness (NLI consistency + quantile round-trip)")
    lines.append("")
    faith_col = "faithfulness_errors"
    if faith_col in df.columns:
        n_errors = int((df[faith_col] != "").sum())
        lines.append(f"- Quantile round-trip mismatches: {n_errors} / {n}")
        if n_errors > 0:
            for _, row in df[df[faith_col] != ""].iterrows():
                lines.append(f"  - Scenario {row['scenario_idx']}: {row[faith_col]}")
    lines.append("")

    # ── 7. Forecast Accuracy ──────────────────────────────────────
    lines.append("## 7. Forecast Accuracy (model vs. synthetic actuals)")
    lines.append("")
    if "mase" in df.columns:
        lines.append("| Metric | Mean | Std | Min | Max |")
        lines.append("|---|---|---|---|---|")
        for metric in ["mase", "mase_first", "fair_mase", "coverage_pct", "interval_sharpness"]:
            if metric in df.columns:
                lines.append(
                    f"| {metric} | {df[metric].mean():.4f} | "
                    f"{df[metric].std():.4f} | {df[metric].min():.4f} | "
                    f"{df[metric].max():.4f} |"
                )
    lines.append("")

    # ── 8. Lowest and highest scoring sentences ───────────────────
    sent_cols = [c for c in df.columns if c.startswith("sent_") and c.endswith("_score")]
    lines.append("## 8. 5 Lowest-Scoring Sentences")
    lines.append("")
    all_sents: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        for sc in sent_cols:
            if pd.notna(row.get(sc)):
                all_sents.append({
                    "score": row[sc],
                    "scenario_type": row["scenario_type"],
                    "scenario_idx": row["scenario_idx"],
                    "verbalizer_type": row.get("verbalizer_type", ""),
                })
    if all_sents:
        all_sents.sort(key=lambda x: x["score"])
        lines.append("| Score | Verbalizer | Scenario | Index |")
        lines.append("|---|---|---|---|")
        for s in all_sents[:5]:
            lines.append(
                f"| {s['score']:.3f} | {s['verbalizer_type']} "
                f"| {s['scenario_type']} | {s['scenario_idx']} |"
            )
    lines.append("")

    lines.append("## 9. 5 Highest-Scoring Sentences")
    lines.append("")
    if all_sents:
        lines.append("| Score | Verbalizer | Scenario | Index |")
        lines.append("|---|---|---|---|")
        for s in all_sents[-5:]:
            lines.append(
                f"| {s['score']:.3f} | {s['verbalizer_type']} "
                f"| {s['scenario_type']} | {s['scenario_idx']} |"
            )
    lines.append("")

    # ── 9. Failure analysis ───────────────────────────────────────
    failures = df[~df["is_consistent"]]
    lines.append("## 10. Failure Analysis")
    lines.append("")
    if len(failures) == 0:
        lines.append("No failures — all scenarios passed the consistency threshold.")
    else:
        lines.append(f"**{len(failures)} scenarios** scored below the 0.70 threshold:")
        lines.append("")
        for _, row in failures.iterrows():
            lines.append(
                f"### Scenario {row['scenario_idx']} "
                f"({row['scenario_type']} / {row.get('verbalizer_type', '')})"
            )
            lines.append(f"- Overall score: {row['overall_consistency']:.4f}")
            for sc in sent_cols:
                if pd.notna(row.get(sc)):
                    lines.append(f"- {sc}: {row[sc]:.4f}")
            lines.append("")
    lines.append("")

    # ── 10. RST relation distribution ─────────────────────────────
    lines.append("## 11. RST Relation Distribution")
    lines.append("")
    if "rst_relations" in df.columns:
        all_rels: Dict[str, int] = {}
        for rels_str in df["rst_relations"]:
            if rels_str:
                for r in str(rels_str).split(","):
                    r = r.strip()
                    if r:
                        all_rels[r] = all_rels.get(r, 0) + 1
        if all_rels:
            lines.append("| Relation | Count |")
            lines.append("|---|---|")
            for rel, cnt in sorted(all_rels.items(), key=lambda x: -x[1]):
                lines.append(f"| {rel} | {cnt} |")
        else:
            lines.append("No RST relations were triggered.")
    lines.append("")

    # ── Visualizations ────────────────────────────────────────────
    lines.append("## Visualizations")
    lines.append("")
    lines.append("![Evaluation plots](evaluation_plots.png)")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved to %s", report_path)
    return str(report_path)


def print_report(df: pd.DataFrame) -> None:
    """Print a textual summary of evaluation results."""
    print("\n" + "=" * 65)
    print("  EVALUATION REPORT")
    print("=" * 65)
    print(f"\n  Scenarios evaluated : {len(df)}")
    print(f"  Mean consistency    : {df['overall_consistency'].mean():.4f}")
    print(f"  Std consistency     : {df['overall_consistency'].std():.4f}")
    print(f"  Min consistency     : {df['overall_consistency'].min():.4f}")
    print(f"  Max consistency     : {df['overall_consistency'].max():.4f}")
    print(
        f"  % consistent (>=0.7): "
        f"{df['is_consistent'].mean() * 100:.1f}%"
    )

    if "mase" in df.columns:
        print(f"\n  Mean MASE           : {df['mase'].mean():.4f}")
        print(f"  Mean Coverage       : {df['coverage_pct'].mean():.1f}%")

    print("\n  --- Breakdown by verbalizer type ---")
    grouped_v = df.groupby("verbalizer_type")["overall_consistency"]
    print(grouped_v.agg(["mean", "std", "min", "max", "count"]).to_string(float_format="%.4f"))

    print("\n  --- Breakdown by scenario type ---")
    grouped_s = df.groupby("scenario_type")["overall_consistency"]
    print(grouped_s.agg(["mean", "std", "min", "max", "count"]).to_string(float_format="%.4f"))
    print()


# ─────────────────────── main ─────────────────────────────────────────
def main() -> None:
    """Entry point for evaluation (callable from run_extensions.py)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Run 1 — univariato
    logger.info("=== Running evaluation WITHOUT covariates ===")
    df_no_cov = run_evaluation(
        n=EVAL_NUM_SCENARIOS, seed=RANDOM_SEED, use_covariates=False
    )
    df_no_cov["covariate_mode"] = "univariate"

    # Run 2 — con covariate
    logger.info("=== Running evaluation WITH covariates ===")
    df_cov = run_evaluation(
        n=EVAL_NUM_SCENARIOS, seed=RANDOM_SEED, use_covariates=True
    )
    df_cov["covariate_mode"] = "with_covariates"

    # Salva separati
    df_no_cov.to_csv(EVAL_DIR / "results_univariate.csv", index=False)
    df_cov.to_csv(EVAL_DIR / "results_covariates.csv", index=False)

    # Salva combinato
    # Filter out columns that are completely empty/all-NA to avoid pandas warning
    df_no_cov = df_no_cov.dropna(axis=1, how='all')
    df_cov = df_cov.dropna(axis=1, how='all')
    df_all = pd.concat([df_no_cov, df_cov], ignore_index=True)
    df_all.to_csv(EVAL_DIR / "evaluation_results.csv", index=False)

    # Report e plot sul combinato
    print_report(df_all)
    plot_results(df_all, save_dir=EVAL_DIR)
    write_report(df_all, save_dir=EVAL_DIR)

    # Confronto diretto
    print("\n" + "=" * 65)
    print("  COMPARISON: UNIVARIATE vs WITH COVARIATES")
    print("=" * 65)
    for mode, df_mode in [("Univariate", df_no_cov), ("With covariates", df_cov)]:
        print(f"\n  {mode}:")
        print(f"   Mean consistency : {df_mode['overall_consistency'].mean():.4f}")
        print(f"   PASS rate        : {df_mode['is_consistent'].mean() * 100:.1f}%")
        if "mase" in df_mode.columns:
            print(f"   Mean MASE        : {df_mode['mase'].mean():.4f}")
        print(f"   Mean sentences   : {df_mode['num_sentences'].mean():.1f}")

    logger.info("All outputs saved to %s", EVAL_DIR)


if __name__ == "__main__":
    main()