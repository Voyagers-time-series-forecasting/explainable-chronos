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
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from extension_1.attribution.types import CovariateSet
from extension_1.evaluation.qa_scorer import QAFaithfulnessReport
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
    future_covariates: Optional[CovariateSet] = None,
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

    has_past = covariates is not None and covariates.n_covariates > 0

    if has_past:
        ax2 = ax.twinx()
        cov_tail = (
            covariates.values[-_HISTORY_TAIL:]
            if len(covariates.values) >= _HISTORY_TAIL
            else covariates.values
        )
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

    ax.set_title("Forecast + Covariates", fontsize=10, fontweight="bold")
    ax.set_xlabel("Steps (0 = forecast start)", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(labelsize=8)


def _plot_attention(
    ax: plt.Axes,
    result: PipelineResult,
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

    # Plot signal against Patch Index instead of raw time steps
    # since we don't have the exact tokenizer patch stride/padding here.
    seq = len(signal)
    patches = np.arange(seq)

    ax.bar(patches, signal, width=0.8, color="#8e44ad", alpha=0.75)
    ax.set_title("Temporal Attention (per model patch)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Patch Index (0 = oldest context)", fontsize=8)
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
    for j, (impact, color) in enumerate(zip(impacts, colors)):
        ax.barh(y[j], impact, color=color, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Relative impact (%)", fontsize=8)
    ax.tick_params(labelsize=8)

    ax.set_title("Attribution (Attention Rollout)", fontsize=10, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_NLI_GREEN, alpha=0.8, label="Positive impact"),
        Patch(facecolor=_NLI_RED, alpha=0.8, label="Negative impact"),
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


_QA_GREEN  = "#27ae60"
_QA_YELLOW = "#f39c12"
_QA_RED    = "#c0392b"


def _qa_color(score: float) -> str:
    if score >= 0.8:
        return _QA_GREEN
    if score >= 0.5:
        return _QA_YELLOW
    return _QA_RED


def _plot_qa(ax: plt.Axes, qa_report: QAFaithfulnessReport | None) -> None:
    """Render a per-slot QA faithfulness table."""
    ax.axis("off")
    if qa_report is None:
        ax.text(0.5, 0.5, "No QA faithfulness data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888")
        ax.set_title("QA Faithfulness", fontsize=10, fontweight="bold")
        return

    overall = qa_report.coverage_score
    status  = "PASS" if qa_report.is_faithful else "FAIL"
    status_color = _QA_GREEN if qa_report.is_faithful else _QA_RED

    ax.text(
        0.0, 1.0,
        f"QA Coverage — Overall: {overall:.3f}  [{status}]  "
        f"({qa_report.correct_slots}/{qa_report.total_slots} slots correct)",
        transform=ax.transAxes, fontsize=9, fontweight="bold",
        color=status_color, va="top",
    )

    n = len(qa_report.slot_scores)
    line_height = 0.88 / max(n, 1)
    for i, ss in enumerate(qa_report.slot_scores):
        y = 0.93 - (i + 1) * line_height
        color = _qa_color(ss.score)
        # Score badge
        ax.text(0.0, y, f"{ss.score:.2f}", transform=ax.transAxes,
                fontsize=7, color=color, fontweight="bold", va="top")
        # Slot name
        ax.text(0.06, y, ss.slot_name, transform=ax.transAxes,
                fontsize=7, color="#555", va="top", style="italic")
        # Expected → extracted
        detail = textwrap.shorten(
            f"expect: {ss.expected_answer!r}  got: {ss.extracted_answer!r}",
            width=90, placeholder="…",
        )
        ax.text(0.22, y, detail, transform=ax.transAxes,
                fontsize=7, color="#2c3e50", va="top")

    ax.set_title("QA Slot Faithfulness", fontsize=10, fontweight="bold")



def render_trace(
    result: PipelineResult,
    history: np.ndarray,
    actuals: Optional[np.ndarray],
    dataset_name: str,
    window_idx: int,
    verbalizer_type: str,
    output_dir: Path,
    covariates: Optional[CovariateSet] = None,
    future_covariates: Optional[CovariateSet] = None,
    qa_report: Optional[QAFaithfulnessReport] = None,
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
    verbalizer_type : str
        Used in the filename.
    output_dir : Path
        Directory where the PNG will be saved.
    covariates : CovariateSet, optional
        Past covariates — overlaid on history portion of the forecast panel.
    future_covariates : CovariateSet, optional
        Future covariates — overlaid on forecast portion (dotted lines).

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(16, 15), constrained_layout=True)
    canvas = FigureCanvasAgg(fig)
    fig.suptitle(
        f"Trace — {dataset_name}  window {window_idx:02d}  [{verbalizer_type}]",
        fontsize=11, fontweight="bold",
    )

    gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[3, 2.2, 1.8, 1.8],
        hspace=0.40,
        wspace=0.30,
    )

    ax_forecast    = fig.add_subplot(gs[0, :])
    ax_attention   = fig.add_subplot(gs[1, 0])
    ax_attribution = fig.add_subplot(gs[1, 1])
    ax_nli         = fig.add_subplot(gs[2, :])
    ax_qa          = fig.add_subplot(gs[3, :])

    _plot_forecast(ax_forecast, history, result, actuals, covariates, future_covariates)
    _plot_attention(ax_attention, result)
    _plot_attribution(ax_attribution, result)
    _plot_nli(ax_nli, result)
    _plot_qa(ax_qa, qa_report)

    fname = f"{dataset_name}_w{window_idx:02d}_{verbalizer_type}.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Trace saved: %s", out_path)
    
    # Save accompanying text trace
    txt_fname = f"{dataset_name}_w{window_idx:02d}_{verbalizer_type}.txt"
    txt_out_path = output_dir / txt_fname
    try:
        with open(txt_out_path, "w", encoding="utf-8") as f:
            f.write(f"=== TRACE CONTEXT: {dataset_name} Window {window_idx:02d} [{verbalizer_type}] ===\n\n")
            
            f.write("--- 1. EXTRACTED FEATURES (CONTEXT) ---\n")
            if hasattr(result, "features") and result.features:
                for k, v in result.features.to_dict().items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write("No features extracted.\n")
            f.write("\n")
            
            f.write("--- 2. ATTRIBUTION ---\n")
            if hasattr(result, "attribution") and result.attribution and result.attribution.attributions:
                for a in result.attribution.attributions[:result.attribution.top_k]:
                    f.write(f"{a.name}: {a.relative_impact_pct:.1f}% ({a.direction})\n")
            else:
                f.write("No attribution data.\n")
            f.write("\n")
            
            f.write("--- 3. RST TEMPLATE / DRAFT ---\n")
            verb_res = result.verbalization
            if getattr(verb_res, "draft_summary", None):
                f.write(f"DRAFT SUMMARY:\n{verb_res.draft_summary}\n\n")
            if getattr(verb_res, "rst_relations", None):
                f.write("RST RELATIONS USED:\n")
                for r in verb_res.rst_relations:
                    f.write(f"- {r}\n")
            else:
                f.write("No RST relations triggered.\n")
            f.write("\n")
            
            f.write("--- 4. LLM PROMPT ---\n")
            if getattr(verb_res, "prompt", None):
                f.write(verb_res.prompt)
            else:
                f.write("No LLM Prompt used.\n")
            f.write("\n\n")
            
            f.write("--- 5. FINAL RESULT ---\n")
            f.write(verb_res.summary)
            f.write("\n\n")

            f.write("--- 6. QA FAITHFULNESS ---\n")
            if qa_report is not None:
                f.write(
                    f"Coverage score : {qa_report.coverage_score:.4f} "
                    f"({'PASS' if qa_report.is_faithful else 'FAIL'})\n"
                    f"Correct slots  : {qa_report.correct_slots}/{qa_report.total_slots}\n"
                )
                if qa_report.missing_slots:
                    f.write(f"Missing slots  : {', '.join(qa_report.missing_slots)}\n")
                f.write("\nSlot-by-slot breakdown:\n")
                for ss in qa_report.slot_scores:
                    ok = "OK" if ss.is_correct else "MISS"
                    f.write(
                        f"  [{ok}] {ss.slot_name:<28} "
                        f"score={ss.score:.3f}  conf={ss.qa_confidence:.3f}\n"
                        f"         expect: {ss.expected_answer!r}\n"
                        f"         got   : {ss.extracted_answer!r}\n"
                    )
            else:
                f.write("QA scorer was not run for this window.\n")
            f.write("\n")
    except Exception as e:
        logger.warning(f"Failed to write text trace: {e}")

    return out_path
