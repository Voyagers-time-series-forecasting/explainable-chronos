"""
Trajectory verbalization — converts raw trajectory data into prose.

The data (segments, turning points) is computed by
``extension_1.features.extractor.extract_trajectory``.
This module handles text generation only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from extension_1.attribution.types import TemporalAttribution


def verbalize_trajectory(trajectory: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Generate a prose sentence describing how the P50 curve moves.

    Returns a sentence and a grounding dict for NLI consistency scoring.
    All parts are joined into a single sentence (not period-separated) so
    the grounding slot stays 1-to-1 with the NLI sentence list.
    """
    segs = trajectory.get("segments", [])
    tps = trajectory.get("turning_points", [])
    start = trajectory["start_value"]
    end_val = trajectory["end_value"]

    if not segs:
        grounding = {
            "type": "trajectory",
            "start_value": start,
            "end_value": end_val,
            "turning_points": [],
        }
        return (
            f"The median forecast holds near {start:.2f} throughout the horizon.",
            grounding,
        )

    parts: list[str] = [f"Starting from {start:.2f}"]

    for tp in tps[:3]:  # cap at 3 turning points for conciseness
        verb = "peaks" if tp.kind == "peak" else "troughs"
        parts.append(f"the series {verb} near {tp.value:.2f} around step {tp.step}")

    end_direction = "above" if end_val > start else "below"
    pct_change = abs((end_val - start) / (abs(start) + 1e-9) * 100)
    parts.append(
        f"before settling near {end_val:.2f} at the horizon "
        f"({pct_change:.1f}% {end_direction} the starting level)"
    )

    sentence = ", ".join(parts) + "."

    grounding = {
        "type": "trajectory",
        "start_value": start,
        "end_value": end_val,
        "pct_change": pct_change,
        "end_direction": end_direction,
        "turning_points": [(tp.step, tp.value, tp.kind) for tp in tps],
    }
    return sentence, grounding


def verbalize_temporal_focus(
    temporal: list[TemporalAttribution],
    history_length: int,
) -> tuple[str, dict[str, Any]]:
    """Describe where in the history window the model's attention peaked for each covariate.

    Example output:
        "The model focused most on temp min in the final 15% of the history window
        (around history step 410), while wind attention peaked near step 280."
    """
    if not temporal:
        return "", {}

    # Sort by peak saliency value so the most concentrated signal comes first
    ranked = sorted(temporal, key=lambda t: t.saliency[t.peak_step], reverse=True)

    def _describe_position(peak: int, n: int) -> str:
        frac = peak / max(n - 1, 1)
        if frac >= 0.75:
            pct = int((1.0 - frac) * 100)
            return f"the final {max(pct, 1)}% of the history window"
        if frac >= 0.5:
            return "the latter half of the history window"
        if frac <= 0.25:
            pct = int(frac * 100)
            return f"the first {max(pct, 1)}% of the history window"
        return "the middle of the history window"

    groundings: list[dict[str, Any]] = []
    parts: list[str] = []
    for ta in ranked[:2]:
        name = ta.covariate_name.replace("_", " ")
        position = _describe_position(ta.peak_step, history_length)
        parts.append((name, position, ta.peak_step))
        groundings.append({
            "covariate_name": ta.covariate_name,
            "peak_step": ta.peak_step,
            "focus_breadth": ta.focus_breadth,
            "position_label": position,
        })

    if len(parts) == 1:
        name, position, step = parts[0]
        sentence = (
            f"The model focused most on {name} in {position} "
            f"(around history step {step})."
        )
    else:
        n1, p1, s1 = parts[0]
        n2, _p2, s2 = parts[1]
        sentence = (
            f"The model focused most on {n1} in {p1} (step {s1}), "
            f"while {n2} attention peaked near step {s2}."
        )

    grounding: dict[str, Any] = {
        "type": "temporal_focus",
        "covariates": groundings,
        "history_length": history_length,
    }
    return sentence, grounding
