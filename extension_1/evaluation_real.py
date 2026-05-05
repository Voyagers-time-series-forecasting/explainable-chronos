"""
Module 7 — Real Dataset Evaluation.

Evaluates Chronos-2 forecast accuracy and NLI verbalization faithfulness
on four public benchmark datasets: ETTh1, ETTm1, Weather (Jena Climate), SP500.

Key design choices:
- ETTh1/ETTm1: official test splits (last 2880 / 23040 rows) for benchmark comparability
- Weather: official test split (last 10539 rows, 10-min intervals, 21 variables)
- SP500: last 756 trading days downloaded via yfinance with OHLCV covariates
- History length 512, horizon 96 — standard in the forecasting literature
- Windows sampled at evenly spaced positions within the test split

All datasets have multiple covariates to exercise covariate attribution (Stage B).

Two evaluation modes:
    dev        — 5 windows, fast iteration (hourly series)
    paper      — 20 windows, suitable for reporting (hourly series)
    dev_daily  — 5 windows, fast iteration (daily series, e.g. SP500)
    paper_daily— 20 windows, reporting mode (daily series)

Usage::

    python run_extensions.py ext1 evaluate-real
    python run_extensions.py ext1 evaluate-real --dataset etth1 --mode dev --verbalizers template
    python run_extensions.py ext1 evaluate-real --dataset sp500 --mode dev_daily --verbalizers template
    python run_extensions.py ext1 evaluate-real --mode paper --verbalizers template llm_guided llm_raw
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PipelineConfig, RANDOM_SEED
from consistency_scorer import NLIConsistencyScorer
from covariate_attribution import CovariateSet
from pipeline import VerbalizationPipeline
from shared.forecast_provider import ChronosForecastProvider
from verbalizer import LLMVerbalizer, TemplateVerbalizer

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent / "eval_results"


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
        history_length=512,   # ~3 weeks hourly — standard in ETT benchmarks
        horizon=96,           # 4 days ahead — standard ETT benchmark horizon
        description="Fast dev mode — 5 windows, 96-step horizon (512 history)",
    ),
    "paper": EvalMode(
        n_windows=20,
        history_length=512,
        horizon=96,
        description="Paper mode — 20 windows, 96-step horizon (512 history)",
    ),
    "dev_daily": EvalMode(
        n_windows=5,
        history_length=252, 
        horizon=5, 
        description="Fast dev mode for daily series — 5 windows, 20-step horizon",
    ),
    "paper_daily": EvalMode(
        n_windows=20,
        history_length=252,   # 1 anno
        horizon=30,           # ~6 settimane
        description="Paper mode for daily series — 20 windows, 30-step horizon",
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
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for SP500. Install with: uv pip install yfinance"
            )
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
    seed: int = RANDOM_SEED,
) -> List[Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]]:
    """Extract (history, future, past_covariates) windows.

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
        if spec.covariate_cols:
            past_cov = {}
            abs_start = offset + start
            abs_end = offset + end
            for col in spec.covariate_cols:
                if col in df.columns:
                    col_arr = df[col].dropna().to_numpy(dtype=np.float64)
                    if len(col_arr) >= abs_end:
                        past_cov[col] = col_arr[abs_start:abs_end]
            if not past_cov:
                past_cov = None

        windows.append((history, future, past_cov))

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
    """Compute MASE, P10-P90 coverage, and interval sharpness."""
    p10 = np.asarray(forecast["p10"])
    p50 = np.asarray(forecast["p50"])
    p90 = np.asarray(forecast["p90"])
    actuals = np.asarray(actuals)[: len(p50)]

    naive_errors = np.abs(np.diff(history))
    naive_mae = float(np.mean(naive_errors)) if len(naive_errors) > 0 else 1.0
    forecast_mae = float(np.mean(np.abs(p50 - actuals)))
    mase = forecast_mae / naive_mae if naive_mae > 1e-9 else 0.0

    in_interval = (actuals >= p10) & (actuals <= p90)
    coverage = float(np.mean(in_interval)) * 100

    mean_width = float(np.mean(p90 - p10))
    actual_range = float(np.ptp(actuals)) if np.ptp(actuals) > 1e-9 else 1.0
    sharpness = mean_width / actual_range

    return {"mase": mase, "coverage_pct": coverage, "interval_sharpness": sharpness}


# ──────────────── pipeline builder ────────────────────────────────────

def build_pipelines(
    verbalizer_names: List[str],
    seed: int,
    config: PipelineConfig,
) -> List[Tuple[str, VerbalizationPipeline]]:
    """Build the requested verbalization pipelines."""
    provider = ChronosForecastProvider()
    scorer = NLIConsistencyScorer()
    pipelines: List[Tuple[str, VerbalizationPipeline]] = []

    if "template" in verbalizer_names:
        tv = TemplateVerbalizer(seed=seed)
        pipelines.append((
            "Template",
            VerbalizationPipeline(
                forecast_provider=provider,
                verbalizer=tv,
                scorer=scorer,
                config=config,
            ),
        ))

    if "llm_guided" in verbalizer_names:
        try:
            lg = LLMVerbalizer(
                template_verbalizer=TemplateVerbalizer(seed=seed),
            )
            pipelines.append((
                "LLM Guided",
                VerbalizationPipeline(
                    forecast_provider=provider,
                    verbalizer=lg,
                    scorer=scorer,
                    config=config,
                ),
            ))
        except Exception as e:
            logger.warning("Could not load LLM Guided: %s", e)

    if "llm_raw" in verbalizer_names:
        try:
            lr = LLMVerbalizer(
                template_verbalizer=TemplateVerbalizer(seed=seed),
            )
            pipelines.append((
                "LLM Raw",
                VerbalizationPipeline(
                    forecast_provider=provider,
                    verbalizer=lr,
                    scorer=scorer,
                    config=config,
                ),
            ))
        except Exception as e:
            logger.warning("Could not load LLM Raw: %s", e)

    return pipelines


def _make_covariate_set(cov_dict: Dict[str, np.ndarray]) -> CovariateSet:
    names = list(cov_dict.keys())
    values = np.stack([cov_dict[n] for n in names], axis=1)
    return CovariateSet(
        names=names,
        values=values,
        descriptions={n: n for n in names},
    )


# ──────────────── main evaluation runner ──────────────────────────────

def run_real_evaluation(
    dataset_keys: List[str],
    mode_key: str = "dev",
    verbalizer_names: Optional[List[str]] = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run evaluation on real datasets."""
    if verbalizer_names is None:
        verbalizer_names = ["template"]

    mode = EVAL_MODES[mode_key]
    config = PipelineConfig(seed=seed)
    pipelines = build_pipelines(verbalizer_names, seed, config)

    logger.info(
        "Real evaluation | mode=%s | datasets=%s | verbalizers=%s",
        mode_key, dataset_keys, [n for n, _ in pipelines],
    )
    logger.info(mode.description)

    records: List[Dict[str, Any]] = []

    for ds_key in dataset_keys:
        spec = DATASET_SPECS[ds_key]
        logger.info("--- Dataset: %s ---", spec.name)

        try:
            df = load_dataset_df(spec)
            windows = extract_windows(df, spec, mode, seed=seed)
        except Exception as e:
            logger.error("Failed to load %s: %s", spec.name, e)
            continue

        for w_idx, (history, future, past_cov) in enumerate(windows):

            covariate_modes: List[Tuple[str, Optional[CovariateSet]]] = [
                ("univariate", None),
            ]
            if past_cov:
                covariate_modes.append(("with_covariates", _make_covariate_set(past_cov)))

            for cov_mode, cov_set in covariate_modes:
                for v_type, pipe in pipelines:
                    try:
                        result = pipe.run(
                            history,
                            horizon=len(future),
                            covariates=cov_set,
                        )
                    except Exception as e:
                        logger.warning(
                            "Window %d [%s/%s/%s] failed: %s",
                            w_idx, spec.name, cov_mode, v_type, e,
                        )
                        continue

                    accuracy = compute_accuracy(result.forecast, future, history)

                    record: Dict[str, Any] = {
                        "dataset": spec.name,
                        "window_idx": w_idx,
                        "covariate_mode": cov_mode,
                        "verbalizer_type": v_type,
                        "history_length": len(history),
                        "horizon": len(future),
                        "overall_consistency": result.consistency_report.overall_score,
                        "is_consistent": result.consistency_report.is_consistent,
                        "num_sentences": len(result.verbalization.sentences),
                        "mase": accuracy["mase"],
                        "coverage_pct": accuracy["coverage_pct"],
                        "interval_sharpness": accuracy["interval_sharpness"],
                        "rst_relations": ",".join(result.verbalization.rst_relations),
                    }

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
                        "[%s] window %d/%d [%s] [%s]  "
                        "consistency=%.4f  mase=%.4f  coverage=%.1f%%",
                        spec.name, w_idx + 1, len(windows),
                        cov_mode, v_type,
                        result.consistency_report.overall_score,
                        accuracy["mase"],
                        accuracy["coverage_pct"],
                    )

    return pd.DataFrame(records)


# ══════════════════ reporting ══════════════════════════════════════════

def plot_real_results(df: pd.DataFrame, save_dir: Path = EVAL_DIR) -> str:
    """Generate comparison plots for real dataset evaluation."""
    save_path = save_dir / "evaluation_real_plots.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    datasets = sorted(df["dataset"].unique())
    cov_colors = {"univariate": "#4c72b0", "with_covariates": "#dd8452"}
    v_colors = {"Template": "#4c72b0", "LLM Guided": "#dd8452", "LLM Raw": "#55a868"}

    n_ds = len(datasets)
    x = np.arange(n_ds)
    cov_modes = sorted(df["covariate_mode"].unique())
    n_modes = len(cov_modes)
    bar_width = 0.35
    offsets = np.linspace(
        -bar_width / 2 * (n_modes - 1),
        bar_width / 2 * (n_modes - 1),
        n_modes,
    )

    # Plot 1 — MASE
    for i, cmode in enumerate(cov_modes):
        means = [
            df[(df["dataset"] == ds) & (df["covariate_mode"] == cmode)]["mase"].mean()
            if len(df[(df["dataset"] == ds) & (df["covariate_mode"] == cmode)]) > 0
            else 0
            for ds in datasets
        ]
        axes[0].bar(x + offsets[i], means, bar_width,
                    label=cmode, color=cov_colors.get(cmode, "grey"), alpha=0.8)
    axes[0].set_title("MASE by Dataset")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("MASE (lower is better)")
    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.5, label="naive baseline")
    axes[0].legend()

    # Plot 2 — Coverage
    for i, cmode in enumerate(cov_modes):
        means = [
            df[(df["dataset"] == ds) & (df["covariate_mode"] == cmode)]["coverage_pct"].mean()
            if len(df[(df["dataset"] == ds) & (df["covariate_mode"] == cmode)]) > 0
            else 0
            for ds in datasets
        ]
        axes[1].bar(x + offsets[i], means, bar_width,
                    label=cmode, color=cov_colors.get(cmode, "grey"), alpha=0.8)
    axes[1].set_title("P10-P90 Coverage by Dataset")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylabel("Coverage % (higher is better)")
    axes[1].axhline(80.0, color="green", linestyle="--", alpha=0.5, label="80% target")
    axes[1].legend()

    # Plot 3 — NLI Consistency boxplot by verbalizer
    v_types = sorted(df["verbalizer_type"].unique())
    data_groups = [
        df.loc[df["verbalizer_type"] == vt, "overall_consistency"].values
        for vt in v_types
    ]
    bp = axes[2].boxplot(data_groups, tick_labels=v_types, patch_artist=True)
    for patch, vt in zip(bp["boxes"], v_types):
        patch.set_facecolor(v_colors.get(vt, "grey"))
        patch.set_alpha(0.7)
    axes[2].axhline(0.7, color="red", linestyle="--", alpha=0.5, label="threshold (0.70)")
    axes[2].set_title("NLI Consistency by Verbalizer")
    axes[2].set_ylabel("Entailment Score")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close(fig)
    logger.info("Real eval plots saved to %s", save_path)
    return str(save_path)


def write_real_report(df: pd.DataFrame, mode_key: str, save_dir: Path = EVAL_DIR) -> str:
    """Write markdown report for real dataset evaluation."""
    report_path = save_dir / "evaluation_report_real.md"
    mode = EVAL_MODES[mode_key]
    lines: List[str] = []

    lines.append("# Real Dataset Evaluation Report — Extension 1")
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
    lines.append(f"| Total rows evaluated | {len(df)} |")
    lines.append(f"| Datasets | {', '.join(sorted(df['dataset'].unique()))} |")
    lines.append(f"| Verbalizers | {', '.join(sorted(df['verbalizer_type'].unique()))} |")
    lines.append(f"| Mean NLI consistency | {df['overall_consistency'].mean():.4f} |")
    lines.append(f"| PASS rate (≥ 0.70) | {df['is_consistent'].mean()*100:.1f}% |")
    lines.append(f"| Mean MASE | {df['mase'].mean():.4f} |")
    lines.append(f"| Mean coverage | {df['coverage_pct'].mean():.1f}% |")
    lines.append("")

    lines.append("## 2. Forecast Accuracy by Dataset and Covariate Mode")
    lines.append("")
    lines.append("| Dataset | Mode | Mean MASE | Mean Coverage | Mean Sharpness |")
    lines.append("|---|---|---|---|---|")
    for (ds, cmode), grp in df.groupby(["dataset", "covariate_mode"]):
        lines.append(
            f"| {ds} | {cmode} | {grp['mase'].mean():.4f} "
            f"| {grp['coverage_pct'].mean():.1f}% "
            f"| {grp['interval_sharpness'].mean():.4f} |"
        )
    lines.append("")

    lines.append("## 3. NLI Consistency by Verbalizer Type")
    lines.append("")
    lines.append("| Verbalizer | Mean | Std | PASS rate |")
    lines.append("|---|---|---|---|")
    for vt, grp in df.groupby("verbalizer_type"):
        lines.append(
            f"| {vt} | {grp['overall_consistency'].mean():.4f} "
            f"| {grp['overall_consistency'].std():.4f} "
            f"| {grp['is_consistent'].mean()*100:.1f}% |"
        )
    lines.append("")

    if "surrogate_r2" in df.columns and df["surrogate_r2"].notna().any():
        lines.append("## 4. Covariate Attribution")
        lines.append("")
        cov_df = df[df["surrogate_r2"].notna()]
        lines.append(f"- Mean surrogate R²: {cov_df['surrogate_r2'].mean():.4f}")
        if "top_covariate" in cov_df.columns:
            top_counts = cov_df["top_covariate"].value_counts().head(5)
            lines.append(
                "- Top covariates: " + ", ".join(
                    f"{v} ({c})" for v, c in top_counts.items()
                )
            )
        lines.append("")

    lines.append("## Visualizations")
    lines.append("")
    lines.append("![Real evaluation plots](evaluation_real_plots.png)")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Real eval report saved to %s", report_path)
    return str(report_path)


def print_real_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  REAL DATASET EVALUATION SUMMARY")
    print("=" * 65)
    print(df.groupby(["dataset", "covariate_mode"])[["mase", "coverage_pct"]].mean().round(4))
    print("\n  NLI Consistency by verbalizer:")
    print(df.groupby("verbalizer_type")["overall_consistency"].agg(["mean", "std"]).round(4))
    print()


# ──────────────── entry point ──────────────────────────────────────────

def main(
    dataset_keys: Optional[List[str]] = None,
    mode_key: str = "dev",
    verbalizer_names: Optional[List[str]] = None,
) -> None:
    """Run real dataset evaluation and save all outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if dataset_keys is None:
        dataset_keys = ["etth1", "ettm1", "weather", "sp500"]
    if verbalizer_names is None:
        verbalizer_names = ["template", "llm_guided", "llm_raw"]

    df = run_real_evaluation(
        dataset_keys=dataset_keys,
        mode_key=mode_key,
        verbalizer_names=verbalizer_names,
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
            fname = f"results_real_{spec.name.replace(' ', '_')}.csv"
            sub.to_csv(EVAL_DIR / fname, index=False)
            logger.info("Saved %s", fname)

    df.to_csv(EVAL_DIR / "evaluation_real_results.csv", index=False)

    print_real_summary(df)
    plot_real_results(df, save_dir=EVAL_DIR)
    write_real_report(df, mode_key=mode_key, save_dir=EVAL_DIR)

    logger.info("All real eval outputs saved to %s", EVAL_DIR)


if __name__ == "__main__":
    main()