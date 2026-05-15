"""Per-scenario trace visualization for the Extension 1 pipeline."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from extension_1.attribution.types import CovariateSet
from extension_1.pipeline import PipelineResult

logger = logging.getLogger(__name__)

_HISTORY_TAIL = 64   # max history points shown in forecast panel
_CMAP_COV = plt.colormaps.get_cmap("tab10")
_NLI_GREEN = "#2ecc71"
_NLI_YELLOW = "#f39c12"
_NLI_RED = "#e74c3c"


def _nli_color(score: float) -> str:
    if score >= 0.70:
        return _NLI_GREEN
    if score >= 0.50:
        return _NLI_YELLOW
    return _NLI_RED


def _extract_attention_signal(attention_weights: dict) -> Optional[np.ndarray]:
    """Return a 1-D per-timestep importance signal from raw attention weights.

    Averages across layers and heads, then sums over query dimension to
    produce a ``(seq_len,)`` array showing which history positions were
    most attended to.  Returns ``None`` on any structural mismatch.
    """
    try:
        attn_layers = attention_weights["attentions"]["enc_time_self_attn_weights"]
        if not attn_layers:
            return None
        accumulated = None
        for layer in attn_layers:
            # layer: (batch, heads, seq, seq) — could be torch.Tensor or np.ndarray
            try:
                arr = layer.detach().cpu().numpy()
            except AttributeError:
                arr = np.asarray(layer)
            avg = arr[0].mean(axis=0)  # (seq, seq) — average over heads
            accumulated = avg if accumulated is None else accumulated + avg
        if accumulated is None:
            return None
        signal = accumulated.mean(axis=0)  # (seq,) — mean query contribution per key
        total = signal.sum()
        return signal / total if total > 1e-12 else signal
    except (KeyError, IndexError, TypeError):
        return None


def _plot_forecast(
    ax: plt.Axes,
    history: np.ndarray,
    result: PipelineResult,
    actuals: Optional[np.ndarray],
    covariates: Optional[CovariateSet],
) -> None:
    tail = history[-_HISTORY_TAIL:]
    h_steps = np.arange(-len(tail), 0)
    f_steps = np.arange(len(np.asarray(result.forecast["p50"])))

    ax.plot(h_steps, tail, color="#7f8c8d", linewidth=1.5, label="History")
    ax.axvline(x=0, color="#7f8c8d", linestyle=":", linewidth=0.8)

    p10 = np.asarray(result.forecast["p10"])
    p50 = np.asarray(result.forecast["p50"])
    p90 = np.asarray(result.forecast["p90"])
    ax.fill_between(f_steps, p10, p90, alpha=0.20, color="#2980b9", label="P10–P90")
    ax.plot(f_steps, p50, color="#2980b9", linewidth=2.0, label="P50 (median)")

    if actuals is not None and len(actuals) > 0:
        act = np.asarray(actuals)[: len(p50)]
        ax.plot(
            f_steps[: len(act)], act,
            color="#e67e22", linewidth=1.5, linestyle="--", label="Actual",
        )

    if covariates is not None and covariates.n_covariates > 0:  # defensive: always expected
        ax2 = ax.twinx()
        cov_tail = covariates.values[-_HISTORY_TAIL:] if len(covariates.values) >= _HISTORY_TAIL else covariates.values
        for i, name in enumerate(covariates.names):
            ax2.plot(
                h_steps[-len(cov_tail):],
                cov_tail[:, i],
                color=_CMAP_COV(i % 10),
                linewidth=0.9,
                alpha=0.6,
                linestyle="-.",
                label=name,
            )
        ax2.set_ylabel("Covariates", fontsize=8, color="#555")
        ax2.tick_params(labelsize=7)
        ax2.legend(loc="upper left", fontsize=7, framealpha=0.5)

    ax.set_title("Forecast", fontsize=10, fontweight="bold")
    ax.set_xlabel("Steps (0 = forecast start)", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(labelsize=8)


def _plot_attention(
    ax: plt.Axes,
    result: PipelineResult,
    history: np.ndarray,
) -> None:
    signal = None
    if result.attention_weights:
        signal = _extract_attention_signal(result.attention_weights)

    if signal is None or len(signal) == 0:
        ax.text(
            0.5, 0.5, "No attention weights available",
            ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#888",
        )
        ax.set_title("Attention Weights", fontsize=10, fontweight="bold")
        ax.axis("off")
        return

    # Align signal with history tail
    seq = len(signal)
    tail_len = min(seq, _HISTORY_TAIL)
    sig = signal[-tail_len:]
    steps = np.arange(-tail_len, 0)

    ax.bar(steps, sig, width=0.8, color="#8e44ad", alpha=0.75)
    ax.set_title("Attention Rollout (per history timestep)", fontsize=10, fontweight="bold")
    ax.set_xlabel("History offset (0 = last observed)", fontsize=8)
    ax.set_ylabel("Normalised attention", fontsize=8)
    ax.tick_params(labelsize=8)


def _plot_attribution(ax: plt.Axes, result: PipelineResult) -> None:
    if result.attribution is None or not result.attribution.attributions:
        ax.text(
            0.5, 0.5, "No attribution data",
            ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#888",
        )
        ax.set_title("Attribution", fontsize=10, fontweight="bold")
        ax.axis("off")
        return

    attrs = result.attribution.attributions[: result.attribution.top_k]
    names = [a.name.replace("_", " ").title() for a in attrs]
    impacts = [a.relative_impact_pct for a in attrs]
    colors = [_NLI_GREEN if a.direction == "positive" else _NLI_RED for a in attrs]

    y = np.arange(len(names))
    ax.barh(y, impacts, color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Relative impact (%)", fontsize=8)
    ax.tick_params(labelsize=8)

    method_label = "SHAP" if not (
        result.attention_weights and result.attribution
    ) else "Attention Rollout"
    r2 = result.attribution.surrogate_r2
    r2_str = f"R²={r2:.3f}" if r2 == r2 else "R²=N/A"  # nan check
    ax.set_title(f"Attribution ({method_label}, {r2_str})", fontsize=10, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_NLI_GREEN, alpha=0.8, label="Positive"),
        Patch(facecolor=_NLI_RED, alpha=0.8, label="Negative"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")


def _plot_nli(ax: plt.Axes, result: PipelineResult) -> None:
    report = result.consistency_report
    sentences = report.sentence_scores

    if not sentences:
        ax.text(0.5, 0.5, "No NLI scores", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        ax.axis("off")
        return

    ax.axis("off")
    overall = report.overall_score
    status = "PASS" if report.is_consistent else "FAIL"
    status_color = _NLI_GREEN if report.is_consistent else _NLI_RED

    ax.text(
        0.0, 1.0,
        f"NLI Consistency — Overall: {overall:.3f}  [{status}]",
        transform=ax.transAxes, fontsize=9, fontweight="bold",
        color=status_color, va="top",
    )

    line_height = 0.85 / max(len(sentences), 1)
    for i, ss in enumerate(sentences):
        y = 0.95 - (i + 1) * line_height
        score = ss.entailment_prob
        color = _nli_color(score)
        # Score badge
        ax.text(0.0, y, f"{score:.2f}", transform=ax.transAxes, fontsize=7,
                color=color, fontweight="bold", va="top")
        # Sentence text (wrapped)
        wrapped = textwrap.shorten(ss.sentence, width=120, placeholder="…")
        ax.text(0.06, y, wrapped, transform=ax.transAxes, fontsize=7,
                color="#2c3e50", va="top")

    ax.set_title("NLI Sentence Scores", fontsize=10, fontweight="bold")


def render_trace(
    result: PipelineResult,
    history: np.ndarray,
    actuals: Optional[np.ndarray],
    dataset_name: str,
    window_idx: int,
    attribution_method: str,
    verbalizer_type: str,
    output_dir: Path,
    covariates: Optional[CovariateSet] = None,
) -> Path:
    """Render and save a multi-panel trace figure for one pipeline scenario.

    Parameters
    ----------
    result : PipelineResult
    history : np.ndarray
        Full history window fed to the model.
    actuals : np.ndarray or None
        Ground-truth values for the forecast horizon.
    dataset_name : str
    window_idx : int
    attribution_method : str
        "shap" or "attention" — used in the filename.
    verbalizer_type : str
        "template" or "llm_guided" — used in the filename.
    output_dir : Path
        Directory where the PNG will be saved.
    covariates : CovariateSet, optional

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    fig.suptitle(
        f"Trace — {dataset_name}  window {window_idx:02d}  "
        f"[{attribution_method} + {verbalizer_type}]",
        fontsize=11, fontweight="bold",
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[3, 2.2, 1.8],
        hspace=0.35,
        wspace=0.30,
    )

    ax_forecast = fig.add_subplot(gs[0, :])
    ax_attention = fig.add_subplot(gs[1, 0])
    ax_attribution = fig.add_subplot(gs[1, 1])
    ax_nli = fig.add_subplot(gs[2, :])

    _plot_forecast(ax_forecast, history, result, actuals, covariates)
    _plot_attention(ax_attention, result, history)
    _plot_attribution(ax_attribution, result)
    _plot_nli(ax_nli, result)

    fname = f"{dataset_name}_w{window_idx:02d}_{attribution_method}_{verbalizer_type}.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trace saved: %s", out_path)
    return out_path
