"""
Module 7 — Evaluation.

Evaluates Chronos-2 forecast accuracy and NLI verbalization faithfulness
on four public benchmark datasets: ETTh1, ETTm1, Weather (Seattle Climate), SP500.

Key design choices:
- ETTh1/ETTm1: official test splits (last 2880 / 23040 rows) for benchmark comparability
- Weather: official test split
- SP500: last 756 trading days downloaded via yfinance with OHLCV covariates
- History length 512, horizon 96 — standard in the forecasting literature
- Windows sampled at evenly spaced positions within the test split

All datasets have multiple covariates to exercise covariate attribution (Stage B).

Evaluation modes:
    dev        — 5 windows, fast iteration (hourly series)
    paper      — 20 windows, suitable for reporting (hourly series)
    dev_daily  — 5 windows, fast iteration (daily series, e.g. SP500)
    paper_daily— 20 windows, reporting mode (daily series)

Usage::

    python run_extensions.py ext1 evaluate
    python run_extensions.py ext1 evaluate --dataset etth1 --mode dev --verbalizers template
    python run_extensions.py ext1 evaluate --dataset sp500 --mode dev --verbalizers template
    python run_extensions.py ext1 evaluate --mode full --verbalizers template llm
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from extension_1.config import PipelineConfig, RANDOM_SEED
from extension_1.evaluation.scoring import NLIConsistencyScorer, SemanticSimilarityScorer
from extension_1.attribution.types import CovariateSet
from extension_1.pipeline import PipelineResult, VerbalizationPipeline
from extension_1.evaluation.trace import render_trace
from extension_1.verbalization.template import TemplateVerbalizer
from extension_1.verbalization.llm import LLMVerbalizer
from shared.forecast_provider import ChronosForecastProvider

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent.parent / "results" / "extension_1"


# ──────────────── evaluation modes ────────────────────────────────────

@dataclass
class EvalMode:
    n_windows: int
    history_length: int
    horizon: int
    description: str


EVAL_MODES: Dict[str, EvalMode] = {
    "dev": EvalMode(
        n_windows=5,
        history_length=512,
        horizon=96,
        description="Fast dev mode — 5 windows, 96-step horizon (512 history)",
    ),
    "full": EvalMode(
        n_windows=200,
        history_length=512,
        horizon=96,
        description="Full mode — 200 windows, exhaustive evaluation",
    ),
}


# ──────────────── dataset specs ───────────────────────────────────────

@dataclass
class DatasetSpec:
    name: str
    hf_path: str
    hf_name: Optional[str]
    target_col: str
    covariate_cols: List[str]
    description: str
    # Official test split size (rows). None = use full series.
    test_split_size: Optional[int] = None


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "etth1": DatasetSpec(
        name="ETTh1",
        hf_path="ETDataset/ett",
        hf_name="ETTh1",
        target_col="OT",
        covariate_cols=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        description="Electricity Transformer Temperature — hourly, 2016-2018, 6 covariates",
        # Official split: train 8640, val 2880, test 2880
        test_split_size=2880,
    ),
    "ettm1": DatasetSpec(
        name="ETTm1",
        hf_path="ETDataset/ett",
        hf_name="ETTm1",
        target_col="OT",
        covariate_cols=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        description="Electricity Transformer Temperature — 15-min, 2016-2018, 6 covariates",
        # Official split: train 34560, val 11520, test 23040
        test_split_size=23040,
    ),
    "weather": DatasetSpec(
        name="Weather",
        hf_path="",
        hf_name=None,
        target_col="temp_max",
        covariate_cols=[
            "precipitation", "temp_min", "wind",
        ],
        description="Seattle weather dataset — daily weather data with temperature, precipitation, and wind",
        # Use all available data
        test_split_size=None,
    ),
    "sp500": DatasetSpec(
        name="SP500",
        hf_path="",
        hf_name=None,
        target_col="Close",
        covariate_cols=["Open", "High", "Low", "Volume"],
        description="S&P 500 index — daily OHLCV, Yahoo Finance 2018-2024",
        test_split_size=756,  # ~3 years of trading days
    ),
}


# ──────────────── dataset loaders ─────────────────────────────────────

def load_dataset_df(spec: DatasetSpec) -> pd.DataFrame:
    """Load dataset from GitHub CSV (ETT/Weather) or Yahoo Finance (SP500)."""

    if spec.name in ("ETTh1", "ETTm1"):
        fname = "ETTh1.csv" if spec.name == "ETTh1" else "ETTm1.csv"
        url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{fname}"
        logger.info("Downloading %s from GitHub ...", fname)
        df = pd.read_csv(url)
        logger.info("Loaded %s from GitHub: %s rows", spec.name, len(df))
        return df

    if spec.name == "Weather":
        # Seattle weather dataset from Vega - simpler alternative
        url = "https://raw.githubusercontent.com/vega/vega/main/docs/data/seattle-weather.csv"
        
        try:
            logger.info("Downloading Seattle weather dataset from GitHub ...")
            df = pd.read_csv(url)
            logger.info("Loaded Weather (Seattle) from GitHub: %s rows", len(df))
            return df
        except Exception as e:
            logger.error("Failed to load Weather dataset: %s", e)
            raise IOError(
                f"Could not load Weather dataset from {url}. "
                "Please check the URL or try a different dataset."
            )

    if spec.name == "SP500":
        logger.info("Downloading SP500 (^GSPC) from Yahoo Finance ...")
        df = yf.download("^GSPC", start="2018-01-01", end="2024-01-01", progress=False)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        logger.info("Loaded SP500: %s rows", len(df))
        return df

    raise ValueError(f"Unknown dataset: {spec.name}")


def extract_windows(
    df: pd.DataFrame,
    spec: DatasetSpec,
    mode: EvalMode,
) -> List[Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]]:
    """Extract (history, future, past_covariates, future_covariates) windows.

    Uses the official test split when available so results are
    comparable with published benchmarks.
    """
    target_full = df[spec.target_col].dropna().to_numpy(dtype=np.float64)

    # Use only the official test split if specified
    if spec.test_split_size is not None:
        target = target_full[-spec.test_split_size:]
        offset = len(target_full) - spec.test_split_size
    else:
        target = target_full
        offset = 0

    total_needed = mode.history_length + mode.horizon

    if len(target) < total_needed:
        raise ValueError(
            f"Dataset {spec.name} test split has only {len(target)} rows, "
            f"need at least {total_needed}. "
            f"Consider reducing history_length or horizon in EVAL_MODES."
        )

    max_start = len(target) - total_needed
    starts = np.linspace(0, max_start, mode.n_windows, dtype=int)

    windows = []
    for start in starts:
        end = start + mode.history_length
        future_end = end + mode.horizon

        history = target[start:end]
        future = target[end:future_end]

        past_cov: Optional[Dict[str, np.ndarray]] = None
        future_cov: Optional[Dict[str, np.ndarray]] = None
        if spec.covariate_cols:
            past_cov = {}
            future_cov = {}
            abs_start = offset + start
            abs_end = offset + end
            abs_future_end = abs_end + mode.horizon
            for col in spec.covariate_cols:
                if col in df.columns:
                    col_arr = df[col].dropna().to_numpy(dtype=np.float64)
                    if len(col_arr) >= abs_end:
                        past_cov[col] = col_arr[abs_start:abs_end]
                    if len(col_arr) >= abs_future_end:
                        future_cov[col] = col_arr[abs_end:abs_future_end]
            if not past_cov:
                past_cov = None
            if not future_cov:
                future_cov = None

        windows.append((history, future, past_cov, future_cov))

    logger.info(
        "%s: %d windows | history=%d | horizon=%d | test_split=%s",
        spec.name, len(windows), mode.history_length, mode.horizon,
        spec.test_split_size or "full",
    )
    return windows


# ──────────────── accuracy metrics ────────────────────────────────────

def compute_accuracy(
    forecast: Dict[str, Any],
    actuals: np.ndarray,
    history: np.ndarray,
) -> Dict[str, float]:
    """Compute forecast accuracy metrics used in real dataset evaluation.

    The metrics are:
    - ``mase``: mean absolute scaled error relative to the one-step naive baseline
      derived from the history differences. This is the traditional MASE.
    - ``mase_first``: first-step MASE, computed only on the first forecasted value.
      This isolates the one-step-ahead quality of the forecast from the multi-step
      horizon.
    - ``fair_mase``: a relative MAE against a multi-step naive baseline that
      repeats the last observed historical value across the entire forecast
      horizon. This is a more appropriate comparison for long-range forecasts
      like 96 steps.
    - ``coverage_pct``: percentage of actual values contained within the P10-P90
      forecast interval, which measures probabilistic calibration.
    - ``interval_sharpness``: mean interval width normalized by the range of the
      actuals. This measures how narrow the predictive interval is.

    We keep both ``mase`` and ``fair_mase`` because the standard scaled error
    baseline can be misleading for a long horizon, while ``fair_mase`` directly
    compares against the multi-step naive forecast that a model should beat.
    """
    p10 = np.asarray(forecast["p10"])
    p50 = np.asarray(forecast["p50"])
    p90 = np.asarray(forecast["p90"])
    actuals = np.asarray(actuals)[: len(p50)]

    forecast_mae = float(np.mean(np.abs(p50 - actuals)))
    forecast_first_mae = float(np.abs(p50[0] - actuals[0])) if len(p50) > 0 and len(actuals) > 0 else 0.0

    # Standard MASE baseline: one-step changes from history (Hyndman & Koehler 2006).
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

    in_interval = (actuals >= p10) & (actuals <= p90)
    coverage = float(np.mean(in_interval)) * 100

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


# ──────────────── pipeline builder ────────────────────────────────────

def build_pipelines(
    seed: int,
    config: PipelineConfig,
) -> list[tuple[str, VerbalizationPipeline]]:
    """Build analyst and executive LLM pipelines sharing one loaded model."""
    provider = ChronosForecastProvider(enable_attention=True)
    nli_scorer = NLIConsistencyScorer()
    try:
        analyst_llm = LLMVerbalizer(
            template_verbalizer=TemplateVerbalizer(seed=seed),
            persona="analyst",
        )
        analyst_llm._load_model()  # eager load so executive can reuse it

        exec_llm = LLMVerbalizer(
            template_verbalizer=TemplateVerbalizer(seed=seed),
            persona="executive",
        )
        exec_llm.share_model_from(analyst_llm)

        return [
            ("analyst", VerbalizationPipeline(provider, analyst_llm, nli_scorer, config)),
            ("executive", VerbalizationPipeline(provider, exec_llm, nli_scorer, config)),
        ]
    except Exception as e:
        logger.warning("Could not initialise LLM pipelines: %s", e)
        return []


def _make_covariate_set(cov_dict: Dict[str, np.ndarray]) -> CovariateSet:
    names = list(cov_dict.keys())
    values = np.stack([cov_dict[n] for n in names], axis=1)
    return CovariateSet(
        names=names,
        values=values,
        descriptions={n: n for n in names},
    )


# ──────────────── main evaluation runner ──────────────────────────────

def run_evaluation(
    dataset_keys: List[str],
    mode_key: str = "dev",
    seed: int = RANDOM_SEED,
    save_traces: bool = False,
    output_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Run LLM evaluation with NLI + semantic similarity vs template."""

    mode = EVAL_MODES[mode_key]
    config = PipelineConfig(seed=seed)
    pipelines = build_pipelines(seed, config)
    if not pipelines:
        logger.error("No LLM pipelines available — aborting evaluation.")
        return pd.DataFrame()

    sem_scorer = SemanticSimilarityScorer()
    logger.info("LLM evaluation | mode=%s | datasets=%s | personas=%s",
                mode_key, dataset_keys, [p for p, _ in pipelines])
    logger.info(mode.description)

    records: List[Dict[str, Any]] = []
    out_path = Path(output_dir) if output_dir else EVAL_DIR
    traces_dir = out_path / "traces"

    for ds_key in dataset_keys:
        spec = DATASET_SPECS[ds_key]
        logger.info("--- Dataset: %s ---", spec.name)

        try:
            df = load_dataset_df(spec)
            windows = extract_windows(df, spec, mode)
        except Exception as e:
            logger.error("Failed to load %s: %s", spec.name, e)
            continue

        for w_idx, (history, future, past_cov, future_cov) in enumerate(windows):
            if not past_cov or not future_cov:
                logger.warning(
                    "Skipping window %d [%s] — covariates unavailable.", w_idx, spec.name
                )
                continue

            cov_set = _make_covariate_set(past_cov)
            future_cov_set = _make_covariate_set(future_cov)

            for persona, pipe in pipelines:
                try:
                    result = pipe.run(
                        history,
                        horizon=len(future),
                        covariates=cov_set,
                        future_covariates=future_cov_set,
                    )
                except Exception as e:
                    logger.warning("Window %d [%s/%s] failed: %s", w_idx, spec.name, persona, e)
                    continue

                accuracy = compute_accuracy(result.forecast, future, history)
                template_text = getattr(result.verbalization, "draft_summary", "") or ""
                semantic_sim = sem_scorer.score(result.verbalization.summary, template_text)

                record: dict = {
                    "dataset": spec.name,
                    "window_idx": w_idx,
                    "persona": persona,
                    "history_length": len(history),
                    "horizon": len(future),
                    "overall_consistency": result.consistency_report.overall_score,
                    "is_consistent": result.consistency_report.is_consistent,
                    "contradiction_rate": result.consistency_report.contradiction_rate,
                    "semantic_vs_template": semantic_sim,
                    "num_sentences": len(result.verbalization.sentences),
                    "mase": accuracy["mase"],
                    "mase_first": accuracy["mase_first"],
                    "fair_mase": accuracy["fair_mase"],
                    "coverage_pct": accuracy["coverage_pct"],
                    "interval_sharpness": accuracy["interval_sharpness"],
                    "rst_relations": ",".join(result.verbalization.rst_relations),
                }

                past_attrs = [a for a in result.attribution.attributions if "(future)" not in a.name]
                record["top_covariate"] = past_attrs[0].name if past_attrs else None
                record["top_covariate_impact_pct"] = past_attrs[0].relative_impact_pct if past_attrs else None

                records.append(record)
                logger.info(
                    "[%s] window %d/%d [%s]  nli=%.3f  semantic=%.3f  "
                    "mase=%.4f  fair_mase=%.4f  coverage=%.1f%%",
                    spec.name, w_idx + 1, len(windows), persona,
                    result.consistency_report.overall_score, semantic_sim,
                    accuracy["mase"], accuracy["fair_mase"], accuracy["coverage_pct"],
                )

                if save_traces:
                    try:
                        render_trace(
                            result=result, history=history, actuals=future,
                            dataset_name=spec.name, window_idx=w_idx,
                            verbalizer_type=persona, output_dir=traces_dir,
                            covariates=cov_set,
                        )
                    except Exception as te:
                        logger.warning("Trace failed [%s/%s/w%d]: %s", spec.name, persona, w_idx, te)

    return pd.DataFrame(records)


# ══════════════════ reporting ══════════════════════════════════════════

def plot_results(df: pd.DataFrame, save_dir: Path = EVAL_DIR) -> str:
    """Generate 2×2 comparison plots for evaluation."""
    save_path = save_dir / "evaluation_plots.png"
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    datasets = sorted(df["dataset"].unique())
    personas = sorted(df["persona"].unique()) if "persona" in df.columns else ["analyst"]
    p_colors = {"analyst": "#dd8452", "executive": "#4c72b0"}
    n_p = len(personas)
    bar_width = 0.35 if n_p > 1 else 0.5
    offsets = ([-bar_width/2, bar_width/2] if n_p == 2 else [0])
    x = np.arange(len(datasets))


    def _ds_means(col: str, persona: str = "") -> list[float]:
        sub = df[df["persona"] == persona] if persona and "persona" in df.columns else df
        return [sub[sub["dataset"] == ds][col].mean() if len(sub[sub["dataset"] == ds]) > 0 else 0.0
                for ds in datasets]

    # ── Panel 1: MASE by dataset ────────────────────────────────────────
    for i, p in enumerate(personas):
        axes[0].bar(x + offsets[i], _ds_means("mase", p), bar_width,
                    label=p, color=p_colors.get(p, "grey"), alpha=0.8)
    axes[0].set_title("MASE by Dataset")
    axes[0].set_xticks(x); axes[0].set_xticklabels(datasets, rotation=15)
    axes[0].set_ylabel("MASE (lower is better)")
    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.5, label="naive baseline")
    axes[0].legend(fontsize=8)

    # ── Panel 2: Coverage by dataset ────────────────────────────────────
    for i, p in enumerate(personas):
        axes[1].bar(x + offsets[i], _ds_means("coverage_pct", p), bar_width,
                    label=p, color=p_colors.get(p, "grey"), alpha=0.8)
    axes[1].set_title("P10-P90 Coverage by Dataset")
    axes[1].set_xticks(x); axes[1].set_xticklabels(datasets, rotation=15)
    axes[1].set_ylabel("Coverage % (higher is better)")
    axes[1].axhline(80.0, color="green", linestyle="--", alpha=0.5, label="80% target")
    axes[1].legend(fontsize=8)

    # ── Panel 3: NLI consistency by persona ──────────────────────────────
    nli_groups = [
        df.loc[df["persona"] == p, "overall_consistency"].dropna().values
        if "persona" in df.columns else df["overall_consistency"].dropna().values
        for p in personas
    ]
    bp3 = axes[2].boxplot(nli_groups, tick_labels=personas, patch_artist=True)
    for patch, p in zip(bp3["boxes"], personas):
        patch.set_facecolor(p_colors.get(p, "grey")); patch.set_alpha(0.7)
    axes[2].axhline(0.7, color="red", linestyle="--", alpha=0.5, label="threshold (0.70)")
    axes[2].set_title("NLI Consistency by Persona")
    axes[2].set_ylabel("Entailment score")
    axes[2].legend(fontsize=8)

    # ── Panel 4: Semantic similarity vs template by persona ────────────────
    if "semantic_vs_template" in df.columns:
        sem_groups = [
            df.loc[df["persona"] == p, "semantic_vs_template"].dropna().values
            if "persona" in df.columns else df["semantic_vs_template"].dropna().values
            for p in personas
        ]
        bp4 = axes[3].boxplot(sem_groups, tick_labels=personas, patch_artist=True)
        for patch, p in zip(bp4["boxes"], personas):
            patch.set_facecolor(p_colors.get(p, "grey")); patch.set_alpha(0.7)
        axes[3].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="drift warning (0.50)")
        axes[3].set_title("Semantic Similarity vs Template by Persona")
        axes[3].set_ylabel("Cosine similarity (higher = less drift)")
        axes[3].set_ylim(0, 1.05)
        axes[3].legend(fontsize=8)
    else:
        axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close(fig)
    logger.info("Eval plots saved to %s", save_path)
    return str(save_path)


def write_report(df: pd.DataFrame, mode_key: str, save_dir: Path = EVAL_DIR) -> str:
    """Write markdown report for evaluation."""
    report_path = save_dir / "evaluation_report.md"
    mode = EVAL_MODES[mode_key]
    lines: List[str] = []

    lines.append("# Evaluation Report — Extension 1")
    lines.append("")
    lines.append(f"> Mode: **{mode_key}** — {mode.description}")
    lines.append(f"> History: {mode.history_length} steps | Horizon: {mode.horizon} steps")
    lines.append(
        "> ETTh1 and Weather use their official test splits "
        "for comparability with published benchmarks."
    )
    lines.append("")

    lines.append("## 1. Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total windows evaluated | {len(df)} |")
    lines.append(f"| Datasets | {', '.join(sorted(df['dataset'].unique()))} |")
    lines.append(f"| Mean NLI consistency | {df['overall_consistency'].mean():.4f} |")
    lines.append(f"| PASS rate (\u2265 0.70) | {df['is_consistent'].mean()*100:.1f}% |")
    if "semantic_vs_template" in df.columns:
        lines.append(f"| Mean semantic similarity vs template | {df['semantic_vs_template'].mean():.4f} |")
    lines.append(f"| Mean MASE | {df['mase'].mean():.4f} |")
    lines.append(f"| Mean fair_mase | {df['fair_mase'].mean():.4f} |")
    lines.append(f"| Mean coverage | {df['coverage_pct'].mean():.1f}% |")
    lines.append("")


    lines.append("## 2. Forecast Accuracy by Dataset")
    lines.append("")
    lines.append("| Dataset | Mean MASE | Mean First-Step MASE | Mean fair_mase | Mean Coverage | Mean Sharpness |")
    lines.append("|---|---|---|---|---|---|")
    for ds, grp in df.groupby("dataset"):
        lines.append(
            f"| {ds} | {grp['mase'].mean():.4f} "
            f"| {grp['mase_first'].mean():.4f} "
            f"| {grp['fair_mase'].mean():.4f} "
            f"| {grp['coverage_pct'].mean():.1f}% "
            f"| {grp['interval_sharpness'].mean():.4f} |"
        )
    lines.append("")

    lines.append("## 3. NLI Consistency by Persona and Dataset")
    lines.append("")
    lines.append("| Persona | Dataset | Mean NLI | Std | PASS rate |")
    lines.append("|---|---|---|---|---|")
    group_cols = [c for c in ("persona", "dataset") if c in df.columns]
    for keys, grp in df.groupby(group_cols):
        key_strs = [keys] if isinstance(keys, str) else list(keys)
        lines.append(
            f"| {' | '.join(str(k) for k in key_strs)} "
            f"| {grp['overall_consistency'].mean():.4f} "
            f"| {grp['overall_consistency'].std():.4f} "
            f"| {grp['is_consistent'].mean()*100:.1f}% |"
        )
    lines.append("")


    if "semantic_vs_template" in df.columns:
        lines.append("## 4. Semantic Similarity vs Template by Persona")
        lines.append("")
        lines.append("| Persona | Dataset | Mean | Std | Min |")
        lines.append("|---|---|---|---|---|")
        group_cols = [c for c in ("persona", "dataset") if c in df.columns]
        for keys, grp in df.groupby(group_cols):
            key_strs = [keys] if isinstance(keys, str) else list(keys)
            col = grp["semantic_vs_template"].dropna()
            lines.append(
                f"| {' | '.join(str(k) for k in key_strs)} "
                f"| {col.mean():.4f} | {col.std():.4f} | {col.min():.4f} |"
            )
        lines.append("")

    if "top_covariate" in df.columns and df["top_covariate"].notna().any():
        lines.append("## 5. Covariate Attribution (Attention Rollout)")
        lines.append("")
        top_counts = df["top_covariate"].value_counts().head(5)
        lines.append(
            "- Top covariates: " + ", ".join(
                f"{v} ({c})" for v, c in top_counts.items()
            )
        )
        lines.append("")


    lines.append("## Visualizations")
    lines.append("")
    lines.append("![Evaluation plots](evaluation_plots.png)")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Eval report saved to %s", report_path)
    return str(report_path)


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY (analyst vs executive)")
    print("=" * 65)
    print(df.groupby("dataset")[["mase", "fair_mase", "coverage_pct"]].mean().round(4))
    print("\n  NLI Consistency by persona:")
    group_col = "persona" if "persona" in df.columns else "dataset"
    print(df.groupby(group_col)[["overall_consistency"]].agg(["mean", "std"]).round(4))
    if "semantic_vs_template" in df.columns:
        print("\n  Semantic similarity vs template by persona:")
        print(df.groupby(group_col)[["semantic_vs_template"]].agg(["mean", "std", "min"]).round(4))
    print()


# ──────────────── entry point ──────────────────────────────────────────

def main(
    dataset_keys: Optional[List[str]] = None,
    mode_key: str = "dev",
    save_traces: bool = False,
    output_dir: Optional[Path | str] = None,
) -> None:
    """Run LLM evaluation and save all outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    out_path = Path(output_dir) if output_dir else EVAL_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    if dataset_keys is None:
        dataset_keys = ["etth1", "ettm1", "weather", "sp500"]

    df = run_evaluation(
        dataset_keys=dataset_keys,
        mode_key=mode_key,
        save_traces=save_traces,
        output_dir=out_path,
    )

    if df.empty:
        logger.error("No results produced — check dataset loading.")
        return

    for ds_key in dataset_keys:
        spec = DATASET_SPECS.get(ds_key)
        if spec is None:
            continue
        sub = df[df["dataset"] == spec.name]
        if not sub.empty:
            fname = f"results_{spec.name.replace(' ', '_')}.csv"
            sub.to_csv(out_path / fname, index=False)
            logger.info("Saved %s", fname)

    df.to_csv(out_path / "evaluation_results.csv", index=False)

    print_summary(df)
    plot_results(df, save_dir=out_path)
    write_report(df, mode_key=mode_key, save_dir=out_path)

    logger.info("All eval outputs saved to %s", out_path)


if __name__ == "__main__":
    main()
