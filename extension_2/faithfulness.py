"""
Extension 2 — Attention-Faithfulness experiment.

Tests whether Extension 1's *attention-based* covariate importance (Attention
Rollout; Abnar & Zuidema, 2020) agrees with Extension 2's *interventional*
what-if sensitivity. This is the attention-faithfulness-via-perturbation
methodology (Jain & Wallace, 2019, "Attention is not Explanation"; Serrano &
Smith, 2019, "Is Attention Interpretable?") applied to the Chronos-2
forecaster: if the attention attribution is faithful, the covariate the model
attends to most should also be the one whose perturbation moves the forecast
most.

Per window we obtain two vectors over the covariates:

  * α̂  — attention importance (Attention Rollout, Extension 1);
  * s   — what-if sensitivity (forward-looking perturbation, Extension 2).

and report:

  * Spearman rank correlation ρ(α̂, s);
  * the top-driver hit rate (does argmax α̂ == argmax s?);
  * a random-importance baseline ρ(random, s) for a sanity check.

The faithfulness outcome is not guaranteed a priori — high ρ means the
attention explanation is faithful to the model's interventional sensitivity;
low ρ is itself a valid, reportable finding.

Usage::

    python run_extensions.py ext2 faithfulness --n-windows 20
    python run_extensions.py ext2 faithfulness --n-windows 5 --ablation
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

from extension_1.attribution.attention import AttentionAttributor
from extension_1.attribution.types import CovariateSet
from extension_2.sensitivity import WhatIfAnalyzer, _extract_p50
from shared.forecast_provider import ChronosForecastProvider

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "extension_2"

DEFAULT_MODE = "remove"
DEFAULT_FACTORS: Tuple[float, ...] = (0.5, 1.5)
# Intervention types compared by the robustness ablation. "remove" and
# "negate" produce real signal; positive "scale" is a near-zero baseline
# because Chronos-2 standardises covariates internally (scale-invariant).
ABLATION_MODES = ("remove", "negate", "scale")


def _select_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_windows(
    dataset_key: str,
    n_windows: int,
    n_history: int = 100,
    seed: int = 42,
) -> List[Tuple[np.ndarray, CovariateSet]]:
    """Load ``n_windows`` real windows (target + the dataset's real covariates).

    Works for any key in ``DATASET_SPECS`` (etth1, ettm1, weather, sp500).
    Windows are evenly spaced across the series. Only the dataset's real
    covariates are used (no random padding), so every covariate is a genuine
    signal and the attention↔sensitivity comparison is not contaminated by
    noise columns.
    """
    from extension_1.evaluation.runner import DATASET_SPECS, load_dataset_df

    if dataset_key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {dataset_key!r}; choose from {list(DATASET_SPECS)}.")
    spec = DATASET_SPECS[dataset_key]
    df = load_dataset_df(spec)
    target = df[spec.target_col].values.astype(np.float64)
    cov_cols = list(spec.covariate_cols)
    cov_matrix = df[cov_cols].values.astype(np.float64)

    max_start = len(df) - n_history
    if max_start <= 0:
        raise ValueError(f"Series too short ({len(df)}) for n_history={n_history}.")

    if n_windows == 1:
        starts = [max_start]
    else:
        starts = np.linspace(0, max_start, n_windows, dtype=int).tolist()

    windows: List[Tuple[np.ndarray, CovariateSet]] = []
    for start in starts:
        sl = slice(start, start + n_history)
        history = target[sl]
        values = cov_matrix[sl]
        covariates = CovariateSet(
            names=list(cov_cols),
            values=values,
            descriptions={c: c for c in cov_cols},
        )
        windows.append((history, covariates))
    logger.info("Loaded %d %s windows (history=%d, %d covariates).",
                len(windows), spec.name, n_history, len(cov_cols))
    return windows


def _attention_importance(
    provider: ChronosForecastProvider,
    attributor: AttentionAttributor,
    history: np.ndarray,
    covariates: CovariateSet,
    horizon: int,
) -> dict[str, float]:
    """Extension 1 attention importance α̂_c for every covariate (raw rollout score)."""
    past_cov = {name: covariates.values[:, i] for i, name in enumerate(covariates.names)}
    result = provider.predict(history, horizon=horizon, past_covariates=past_cov)
    if not isinstance(result, tuple):
        raise RuntimeError(
            "Attention weights were not returned. Create the provider with enable_attention=True."
        )
    _, attention_weights = result
    attribution = attributor.explain(covariates, attention_weights=attention_weights)
    # top_k == n_covariates ⇒ all covariates are present; use the raw rollout score.
    return {a.name: float(a.importance_score) for a in attribution.attributions}


def _aligned_vectors(
    names: Sequence[str],
    importance: dict[str, float],
    sensitivity: dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    alpha = np.array([importance.get(n, 0.0) for n in names], dtype=np.float64)
    s = np.array([sensitivity.get(n, 0.0) for n in names], dtype=np.float64)
    return alpha, s


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Spearman ρ guarding against constant vectors (which give NaN)."""
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan"), float("nan")
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def _pooled_correlation(df: pd.DataFrame) -> Tuple[float, float, int]:
    """Pooled rank correlation across all (window × covariate) pairs.

    Within each window, importance and sensitivity are rank-transformed (so
    different windows are comparable), then all ranks are pooled and a single
    Pearson correlation is computed over them. This aggregates far more points
    than the per-window 6-point Spearman, giving a tighter estimate.
    """
    ranks_a, ranks_b = [], []
    for _, row in df.iterrows():
        if pd.isna(row.get("rho")) or not row.get("importance"):
            continue
        imp = json.loads(row["importance"])
        sen = json.loads(row["sensitivity"])
        names = list(imp)
        a = np.array([imp[n] for n in names], dtype=float)
        b = np.array([sen.get(n, 0.0) for n in names], dtype=float)
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        ranks_a.extend(rankdata(a))
        ranks_b.extend(rankdata(b))
    if len(ranks_a) < 3:
        return float("nan"), float("nan"), len(ranks_a)
    rho, p = pearsonr(ranks_a, ranks_b)
    return float(rho), float(p), len(ranks_a)


def run_faithfulness_experiment(
    n_windows: int = 20,
    n_history: int = 100,
    horizon: int = 14,
    dataset: str = "etth1",
    mode: str = DEFAULT_MODE,
    factors: Sequence[float] = DEFAULT_FACTORS,
    seed: int = 42,
    ablation: bool = False,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the attention-faithfulness experiment and write per-window results.

    Returns the per-window DataFrame; also writes CSVs + a JSON summary
    (filenames suffixed with the dataset key so runs don't overwrite).
    """
    rng = np.random.default_rng(seed)
    device = _select_device()
    logger.info("Faithfulness experiment: dataset=%s, %d windows, mode=%s, device=%s",
                dataset, n_windows, mode, device)

    provider = ChronosForecastProvider(device=device, enable_attention=True)
    windows = _load_windows(dataset, n_windows, n_history=n_history, seed=seed)
    n_cov = len(windows[0][1].names)
    attributor = AttentionAttributor(top_k=n_cov)  # top_k == C ⇒ keep all covariates
    analyzer = WhatIfAnalyzer(provider, horizon=horizon)

    records: List[dict] = []
    for w_idx, (history, covariates) in enumerate(windows):
        names = covariates.names
        try:
            importance = _attention_importance(provider, attributor, history, covariates, horizon)
            sensitivity = analyzer.sensitivity(history, covariates, mode=mode, factors=factors, horizon=horizon)

            alpha, s = _aligned_vectors(names, importance, sensitivity)
            rho, p = _safe_spearman(alpha, s)

            # Random-importance baseline (sanity check).
            rand = rng.standard_normal(len(names))
            rho_rand, _ = _safe_spearman(rand, s)

            top_attn = names[int(np.argmax(alpha))]
            top_sens = names[int(np.argmax(s))]
            top_match = top_attn == top_sens

            logger.info("[%d/%d] rho=%.3f (rand=%.3f) top_attn=%s top_sens=%s match=%s",
                        w_idx + 1, len(windows), rho, rho_rand, top_attn, top_sens, top_match)

            records.append({
                "window": w_idx,
                "rho": rho,
                "p_value": p,
                "rho_random": rho_rand,
                "top_attention_covariate": top_attn,
                "top_sensitivity_covariate": top_sens,
                "top_driver_match": top_match,
                "importance": json.dumps({k: round(v, 6) for k, v in importance.items()}),
                "sensitivity": json.dumps({k: round(v, 6) for k, v in sensitivity.items()}),
                "error": None,
            })
        except Exception as exc:  # keep the sweep going on a single-window failure
            logger.warning("Window %d failed: %s", w_idx, exc)
            records.append({
                "window": w_idx, "rho": float("nan"), "p_value": float("nan"),
                "rho_random": float("nan"), "top_attention_covariate": None,
                "top_sensitivity_covariate": None, "top_driver_match": None,
                "importance": None, "sensitivity": None, "error": str(exc),
            })

    df = pd.DataFrame(records)
    out = Path(output_dir) if output_dir else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"faithfulness_results_{dataset}.csv", index=False)

    summary = _summarize(df, dataset, mode, factors, n_history, horizon, seed, device)
    pooled_rho, pooled_p, pooled_n = _pooled_correlation(df)
    summary["pooled_rho"] = pooled_rho
    summary["pooled_p_value"] = pooled_p
    summary["pooled_n_pairs"] = pooled_n
    with open(out / f"faithfulness_summary_{dataset}.json", "w") as f:
        json.dump(summary, f, indent=2)
    _print_summary(summary)

    if ablation:
        abl = run_ablation(analyzer, provider, attributor, windows, horizon, factors)
        abl.to_csv(out / f"faithfulness_ablation_{dataset}.csv", index=False)
        _print_ablation(abl)

    return df


def _summarize(df, dataset, mode, factors, n_history, horizon, seed, device) -> dict:
    valid = df[df["rho"].notna()]
    n = len(valid)
    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "config": {
            "n_windows": int(len(df)),
            "n_valid": int(n),
            "perturbation_mode": mode,
            "factors": list(factors),
            "n_history": n_history,
            "horizon": horizon,
            "seed": seed,
            "device": device,
            "dataset": dataset,
        },
        "mean_rho": float(valid["rho"].mean()) if n else float("nan"),
        "std_rho": float(valid["rho"].std()) if n else float("nan"),
        "frac_rho_positive": float((valid["rho"] > 0).mean()) if n else float("nan"),
        "mean_rho_random": float(valid["rho_random"].mean()) if n else float("nan"),
        "top_driver_hit_rate": float(valid["top_driver_match"].mean()) if n else float("nan"),
    }


def _print_summary(summary: dict) -> None:
    c = summary["config"]
    print("\n" + "=" * 65)
    print("  EXTENSION 2 — ATTENTION FAITHFULNESS (what-if vs attention)")
    print("=" * 65)
    print(f"\n  Dataset / windows     : {c['dataset']} — {c['n_valid']}/{c['n_windows']} valid")
    print(f"  Perturbation          : {c['perturbation_mode']}   (horizon={c['horizon']}, history={c['n_history']})")
    print(f"\n  Mean Spearman rho     : {summary['mean_rho']:.3f} ± {summary['std_rho']:.3f}  (per-window)")
    if "pooled_rho" in summary:
        print(f"  Pooled rank corr.     : {summary['pooled_rho']:.3f}  (p={summary['pooled_p_value']:.3g}, "
              f"{summary['pooled_n_pairs']} pairs)")
    print(f"  Windows with rho>0    : {summary['frac_rho_positive']*100:.1f}%")
    print(f"  Top-driver hit rate   : {summary['top_driver_hit_rate']*100:.1f}%")
    print(f"  Random baseline rho   : {summary['mean_rho_random']:.3f}  (sanity: should be ~0)")
    print()


def run_ablation(
    analyzer: WhatIfAnalyzer,
    provider: ChronosForecastProvider,
    attributor: AttentionAttributor,
    windows: List[Tuple[np.ndarray, CovariateSet]],
    horizon: int,
    factors: Sequence[float],
) -> pd.DataFrame:
    """Robustness ablation: mean rho and mean what-if signal per perturbation type.

    Also reports the mean displacement magnitude, exposing that positive
    scaling is a near-zero (uninformative) intervention while removal/negation
    move the forecast — justifying the erasure-based default.
    """
    rows = []
    # Attention importance does not depend on the perturbation — compute once per window.
    importances = []
    for history, covariates in windows:
        try:
            importances.append(_attention_importance(provider, attributor, history, covariates, horizon))
        except Exception as exc:
            logger.warning("Ablation: attention failed on a window: %s", exc)
            importances.append(None)

    for mode in ABLATION_MODES:
        rhos, signals = [], []
        for (history, covariates), importance in zip(windows, importances):
            if importance is None:
                continue
            try:
                s = analyzer.sensitivity(history, covariates, mode=mode, factors=factors, horizon=horizon)
                signals.append(float(np.mean(list(s.values()))))
                alpha, svec = _aligned_vectors(covariates.names, importance, s)
                rho, _ = _safe_spearman(alpha, svec)
                if not np.isnan(rho):
                    rhos.append(rho)
            except Exception as exc:
                logger.warning("Ablation %s failed on a window: %s", mode, exc)
        rows.append({
            "perturbation": mode,
            "n_valid_rho": len(rhos),
            "mean_signal": float(np.mean(signals)) if signals else float("nan"),
            "mean_rho": float(np.mean(rhos)) if rhos else float("nan"),
            "std_rho": float(np.std(rhos)) if rhos else float("nan"),
        })
    return pd.DataFrame(rows)


def _print_ablation(df: pd.DataFrame) -> None:
    print("\n  --- Robustness ablation (perturbation type) ---")
    print("  perturbation   mean_signal   mean_rho ± std   (n)")
    for _, r in df.iterrows():
        print(f"  {r['perturbation']:12s}  {r['mean_signal']:.5f}     "
              f"{r['mean_rho']:+.3f} ± {r['std_rho']:.3f}   ({r['n_valid_rho']})")
    print("  (positive scaling ≈ 0 signal: Chronos-2 standardises covariates ⇒ scale-invariant)")
    print()


# ──────────────── grouped-removal variant (redundancy-corrected) ──────

def _correlation_groups(values: np.ndarray, threshold: float = 0.7) -> List[List[int]]:
    """Cluster covariates whose pairwise |correlation| ≥ threshold (union-find).

    Highly correlated covariates are redundant: removing one alone barely moves
    the forecast (the others compensate), which makes leave-one-out importance
    underestimate them. Removing them as a group fixes that confound.
    """
    n = values.shape[1]
    C = np.corrcoef(values.T)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(C[i, j]) and abs(C[i, j]) >= threshold:
                parent[find(i)] = find(j)

    groups: dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def run_grouped_faithfulness(
    n_windows: int = 50,
    n_history: int = 100,
    horizon: int = 14,
    dataset: str = "etth1",
    threshold: float = 0.7,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Redundancy-corrected faithfulness: remove correlated covariates as groups.

    Attention importance is summed within each correlation group and compared
    (rank correlation) against the group's interventional what-if effect
    (removing the whole group). This removes the leave-one-out redundancy
    confound that affects the per-covariate experiment on correlated datasets.
    """
    device = _select_device()
    logger.info("Grouped faithfulness: dataset=%s, %d windows, corr-threshold=%.2f, device=%s",
                dataset, n_windows, threshold, device)

    provider = ChronosForecastProvider(device=device, enable_attention=True)
    windows = _load_windows(dataset, n_windows, n_history=n_history, seed=seed)
    n_cov = len(windows[0][1].names)
    attributor = AttentionAttributor(top_k=n_cov)
    analyzer = WhatIfAnalyzer(provider, horizon=horizon)

    rng = np.random.default_rng(seed)
    records: List[dict] = []
    pooled_a, pooled_b = [], []
    for w_idx, (history, covariates) in enumerate(windows):
        try:
            importance = _attention_importance(provider, attributor, history, covariates, horizon)
            groups = _correlation_groups(covariates.values, threshold)
            base_p50 = analyzer._predict_p50(history, covariates, horizon)
            level = float(np.mean(np.abs(base_p50))) + 1e-9

            # Random per-covariate scores summed by the SAME groups: a baseline
            # that ALSO benefits from group size, so it isolates whether the real
            # attention adds signal beyond the size/redundancy structure.
            rand_imp = {name: float(rng.random()) for name in covariates.names}

            g_attn, g_sens, g_rand = [], [], []
            for g in groups:
                g_attn.append(float(sum(importance[covariates.names[i]] for i in g)))
                g_rand.append(float(sum(rand_imp[covariates.names[i]] for i in g)))
                modc = CovariateSet(
                    names=list(covariates.names),
                    values=covariates.values.copy(),
                    descriptions=dict(covariates.descriptions),
                )
                for i in g:
                    modc.values[:, i] = 0.0
                mod_p50 = analyzer._predict_p50(history, modc, horizon)
                m = min(len(base_p50), len(mod_p50))
                g_sens.append(float(np.mean(np.abs(mod_p50[:m] - base_p50[:m]))) / level)

            a, b = np.array(g_attn), np.array(g_sens)
            rho, p = _safe_spearman(a, b)
            rho_rand, _ = _safe_spearman(np.array(g_rand), b)
            if not np.isnan(rho):
                pooled_a.extend(rankdata(a)); pooled_b.extend(rankdata(b))
            top_match = (int(np.argmax(a)) == int(np.argmax(b))) if len(a) else None
            logger.info("[%d/%d] groups=%d rho=%.3f (rand=%.3f) top_match=%s",
                        w_idx + 1, len(windows), len(groups), rho, rho_rand, top_match)
            records.append({
                "window": w_idx, "n_groups": len(groups), "rho": rho, "p_value": p,
                "rho_random": rho_rand, "top_group_match": top_match,
                "group_attn": json.dumps([round(x, 4) for x in g_attn]),
                "group_sens": json.dumps([round(x, 6) for x in g_sens]),
                "error": None,
            })
        except Exception as exc:
            logger.warning("Grouped window %d failed: %s", w_idx, exc)
            records.append({
                "window": w_idx, "n_groups": None, "rho": float("nan"), "p_value": float("nan"),
                "rho_random": float("nan"), "top_group_match": None,
                "group_attn": None, "group_sens": None, "error": str(exc),
            })

    df = pd.DataFrame(records)
    out = Path(output_dir) if output_dir else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"faithfulness_grouped_{dataset}.csv", index=False)

    valid = df[df["rho"].notna()]
    pooled_rho, pooled_p = (pearsonr(pooled_a, pooled_b) if len(pooled_a) >= 3 else (float("nan"), float("nan")))
    summary = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "variant": "grouped_removal",
        "dataset": dataset, "n_windows": int(len(df)), "n_valid": int(len(valid)),
        "corr_threshold": threshold, "horizon": horizon, "n_history": n_history, "seed": seed,
        "mean_n_groups": float(valid["n_groups"].mean()) if len(valid) else float("nan"),
        "mean_rho": float(valid["rho"].mean()) if len(valid) else float("nan"),
        "std_rho": float(valid["rho"].std()) if len(valid) else float("nan"),
        "pooled_rho": float(pooled_rho), "pooled_p_value": float(pooled_p),
        "pooled_n_pairs": int(len(pooled_a)),
        "mean_rho_random": float(valid["rho_random"].mean()) if len(valid) else float("nan"),
        "top_group_hit_rate": float(valid["top_group_match"].mean()) if len(valid) else float("nan"),
    }
    with open(out / f"faithfulness_grouped_summary_{dataset}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("  EXT 2 — GROUPED FAITHFULNESS (redundancy-corrected)")
    print("=" * 65)
    print(f"\n  Dataset               : {dataset}  ({summary['n_valid']}/{summary['n_windows']} windows)")
    print(f"  Corr. threshold       : {threshold}  →  mean {summary['mean_n_groups']:.1f} groups/window")
    print(f"  Mean Spearman rho     : {summary['mean_rho']:.3f} ± {summary['std_rho']:.3f}")
    print(f"  Pooled rank corr.     : {summary['pooled_rho']:.3f}  (p={summary['pooled_p_value']:.3g}, "
          f"{summary['pooled_n_pairs']} pairs)")
    print(f"  Top-group hit rate    : {summary['top_group_hit_rate']*100:.1f}%")
    print(f"  Random-attn baseline  : {summary['mean_rho_random']:.3f}  (size-confound control: should be ~0)")
    print()
    return df


def main(
    n_windows: int = 20,
    n_history: int = 100,
    horizon: int = 14,
    dataset: str = "etth1",
    mode: str = DEFAULT_MODE,
    factors: Optional[Sequence[float]] = None,
    seed: int = 42,
    ablation: bool = False,
    grouped: bool = False,
    corr_threshold: float = 0.7,
    output_dir: Optional[Path] = None,
) -> None:
    """Entry point for the faithfulness experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if grouped:
        run_grouped_faithfulness(
            n_windows=n_windows,
            n_history=n_history,
            horizon=horizon,
            dataset=dataset,
            threshold=corr_threshold,
            seed=seed,
            output_dir=output_dir,
        )
        return
    run_faithfulness_experiment(
        n_windows=n_windows,
        n_history=n_history,
        horizon=horizon,
        dataset=dataset,
        mode=mode,
        factors=tuple(factors) if factors else DEFAULT_FACTORS,
        seed=seed,
        ablation=ablation,
        output_dir=output_dir,
    )
