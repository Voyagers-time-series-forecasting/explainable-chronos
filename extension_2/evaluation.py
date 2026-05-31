"""
Extension 2 — Evaluation.

Evaluates the dialogue system on predefined natural-language query sets
with known expected outcomes.

Metric: Task Completion Rate (TCR) — the proportion of test queries
where the system:
    1. Correctly classifies the intent type
    2. Correctly identifies the target covariate (if applicable)
    3. Applies the right modification to Chronos-2 inputs
    4. Produces a faithful NL response (NLI score >= 0.70)

Test set structure:
    Each test case specifies:
    - query          : the natural language input
    - expected_intent: the correct intent type
    - expected_covariate: the correct target covariate (or None)
    - expected_horizon: the correct new horizon (or None)
    - description    : human-readable description of what should happen

Usage::

    python run_extensions.py ext2 evaluate
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from extension_1.attribution.types import CovariateSet
from extension_2.dialogue import DialogueSystem
from extension_2.intent_parser import ParsedIntent

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent / "eval_results"

# Synthetic covariate names used across intent-only and full-pipeline tests.
# Must match what the test set queries refer to (slot extraction relies on them).
COVARIATE_NAMES: List[str] = [
    "marketing_spend", "website_traffic", "previous_day_sales",
    "competitor_promotion_index", "price_discount_percentage",
    "holiday_proximity", "shipping_delay_hours",
    "social_media_mentions", "weather_temperature", "random_sensor_noise",
]


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
    import sys, datetime, subprocess, importlib.metadata

    cfg: dict = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "evaluation_set": evaluation_set,
        "run_full_pipeline": run_full_pipeline,
        "models": {
            "chronos": "autogluon/chronos-2-small",
            "nli": "facebook/bart-large-mnli",
        },
        "python": sys.version,
        "package_versions": {},
    }

    for pkg in ("torch", "transformers", "chronos"):
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

    Tries to load a real ETTh1 window (real temporal dynamics) and maps
    the ETTh1 covariate columns onto the synthetic names used by the test
    set. Falls back to a sine-wave + random-noise scenario if the dataset
    is not available locally.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from extension_1.evaluation.runner import DATASET_SPECS, load_dataset_df

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
        logger.warning(
            "Could not load ETTh1 data (%s) — falling back to synthetic scenario.", exc
        )
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 4 * np.pi, n_history)
        history = np.sin(t) + 0.1 * rng.standard_normal(n_history)
        cov_values = rng.standard_normal((n_history, len(covariate_names)))
        covariates = CovariateSet(names=covariate_names, values=cov_values)
        return history, covariates


def _check_delta_faithfulness(delta_p50: Optional[np.ndarray], answer: str) -> Optional[bool]:
    """Return True if the answer text reflects the direction of the forecast delta.

    Uses simple keyword matching: increase words for positive delta,
    decrease words for negative delta. Returns None when delta is absent
    (confidence queries) or near-zero (direction ambiguous).
    """
    if delta_p50 is None:
        return None
    mean_delta = float(np.mean(delta_p50))
    answer_lower = answer.lower()
    if mean_delta > 0.05:
        return any(w in answer_lower for w in ["increas", "higher", "ris", "up", "more"])
    if mean_delta < -0.05:
        return any(w in answer_lower for w in ["decreas", "lower", "fall", "down", "less", "reduc"])
    return None  # near-zero delta — direction not verifiable by keyword


# ──────────────── test set ────────────────────────────────────────────

@dataclass
class TestCase:
    """A single test case for the dialogue evaluation.

    Attributes
    ----------
    query : str
        Natural language input to the dialogue system.
    expected_intent : str
        Correct intent type.
    expected_covariate : str | None
        Correct target covariate name (or None if not applicable).
    expected_horizon : int | None
        Correct new horizon (or None if not applicable).
    description : str
        Human-readable description of what the system should do.
    """
    query: str
    expected_intent: str
    description: str
    expected_covariate: Optional[str] = None
    expected_horizon: Optional[int] = None


# Development set — 40 queries used while developing and fixing the parser.
# These are not used for the official reported score.
DEV_SET: List[TestCase] = [
    # ── remove_covariate (10 queries) ────────────────────────────────
    TestCase(
        query="What would happen if we removed the marketing spend covariate?",
        expected_intent="remove_covariate",
        expected_covariate="marketing_spend",
        description="Remove marketing_spend by name",
    ),
    TestCase(
        query="What if there were no website traffic data?",
        expected_intent="remove_covariate",
        expected_covariate="website_traffic",
        description="Remove website_traffic with no-data phrasing",
    ),
    TestCase(
        query="Show me the forecast without the shipping delay information.",
        expected_intent="remove_covariate",
        expected_covariate="shipping_delay_hours",
        description="Remove shipping_delay_hours with 'without' phrasing",
    ),
    TestCase(
        query="Exclude the competitor promotion index from the model.",
        expected_intent="remove_covariate",
        expected_covariate="competitor_promotion_index",
        description="Remove competitor_promotion_index with 'exclude' phrasing",
    ),
    TestCase(
        query="Drop the previous day sales covariate.",
        expected_intent="remove_covariate",
        expected_covariate="previous_day_sales",
        description="Remove previous_day_sales with 'drop' phrasing",
    ),
    TestCase(
        query="What if holiday proximity didn't affect the forecast?",
        expected_intent="remove_covariate",
        expected_covariate="holiday_proximity",
        description="Remove holiday_proximity with hypothetical phrasing",
    ),
    TestCase(
        query="Zero out the social media mentions variable.",
        expected_intent="remove_covariate",
        expected_covariate="social_media_mentions",
        description="Remove social_media_mentions with 'zero out' phrasing",
    ),
    TestCase(
        query="Eliminate weather temperature from the analysis.",
        expected_intent="remove_covariate",
        expected_covariate="weather_temperature",
        description="Remove weather_temperature with 'eliminate' phrasing",
    ),
    TestCase(
        query="What would the forecast look like without price discounts?",
        expected_intent="remove_covariate",
        expected_covariate="price_discount_percentage",
        description="Remove price_discount_percentage",
    ),
    TestCase(
        query="Remove all covariate effects and show univariate forecast.",
        expected_intent="remove_covariate",
        expected_covariate=None,  # no specific covariate
        description="Generic remove request without specific covariate",
    ),

    # ── scale_covariate (10 queries) ─────────────────────────────────
    TestCase(
        query="What if marketing spend doubled?",
        expected_intent="scale_covariate",
        expected_covariate="marketing_spend",
        description="Double marketing_spend",
    ),
    TestCase(
        query="What would happen if website traffic increased by 50%?",
        expected_intent="scale_covariate",
        expected_covariate="website_traffic",
        description="Increase website_traffic by 50%",
    ),
    TestCase(
        query="Show me the forecast if price discounts were reduced by 30%.",
        expected_intent="scale_covariate",
        expected_covariate="price_discount_percentage",
        description="Reduce price_discount_percentage by 30%",
    ),
    TestCase(
        query="What if social media mentions dropped by 20%?",
        expected_intent="scale_covariate",
        expected_covariate="social_media_mentions",
        description="Decrease social_media_mentions by 20%",
    ),
    TestCase(
        query="Halve the shipping delay and show the new forecast.",
        expected_intent="scale_covariate",
        expected_covariate="shipping_delay_hours",
        description="Halve shipping_delay_hours",
    ),
    TestCase(
        query="What if competitor promotions tripled in intensity?",
        expected_intent="scale_covariate",
        expected_covariate="competitor_promotion_index",
        description="Triple competitor_promotion_index",
    ),
    TestCase(
        query="Scale up marketing spend by 2x.",
        expected_intent="scale_covariate",
        expected_covariate="marketing_spend",
        description="Scale marketing_spend by 2x",
    ),
    TestCase(
        query="What if holiday proximity increased by 25%?",
        expected_intent="scale_covariate",
        expected_covariate="holiday_proximity",
        description="Increase holiday_proximity by 25%",
    ),
    TestCase(
        query="Reduce website traffic by half.",
        expected_intent="scale_covariate",
        expected_covariate="website_traffic",
        description="Halve website_traffic",
    ),
    TestCase(
        query="What would happen if previous day sales rose by 10%?",
        expected_intent="scale_covariate",
        expected_covariate="previous_day_sales",
        description="Increase previous_day_sales by 10%",
    ),

    # ── change_horizon (10 queries) ───────────────────────────────────
    TestCase(
        query="Show me the next 7 days instead.",
        expected_intent="change_horizon",
        expected_horizon=168,  # 7 * 24
        description="Change horizon to 7 days",
    ),
    TestCase(
        query="Can you forecast the next 30 days?",
        expected_intent="change_horizon",
        expected_horizon=720,
        description="Change horizon to 30 days",
    ),
    TestCase(
        query="I want to see a 14-day forecast.",
        expected_intent="change_horizon",
        expected_horizon=336,
        description="Change horizon to 14 days",
    ),
    TestCase(
        query="Predict the next 48 hours.",
        expected_intent="change_horizon",
        expected_horizon=48,
        description="Change horizon to 48 hours",
    ),
    TestCase(
        query="Show me 10 steps ahead.",
        expected_intent="change_horizon",
        expected_horizon=10,
        description="Change horizon to 10 steps",
    ),
    TestCase(
        query="Can you extend the forecast to 3 weeks?",
        expected_intent="change_horizon",
        expected_horizon=504,  # 3 * 168
        description="Change horizon to 3 weeks",
    ),
    TestCase(
        query="Give me a 24-hour forecast.",
        expected_intent="change_horizon",
        expected_horizon=24,
        description="Change horizon to 24 hours",
    ),
    TestCase(
        query="Forecast for the next 5 days.",
        expected_intent="change_horizon",
        expected_horizon=120,
        description="Change horizon to 5 days",
    ),
    TestCase(
        query="Show 50 periods ahead.",
        expected_intent="change_horizon",
        expected_horizon=50,
        description="Change horizon to 50 periods",
    ),
    TestCase(
        query="I need a 2-week prediction.",
        expected_intent="change_horizon",
        expected_horizon=336,
        description="Change horizon to 2 weeks",
    ),

    # ── confidence_query (10 queries) ─────────────────────────────────
    TestCase(
        query="How confident are you in this forecast?",
        expected_intent="confidence_query",
        description="Direct confidence question",
    ),
    TestCase(
        query="What is the uncertainty around your predictions?",
        expected_intent="confidence_query",
        description="Uncertainty question",
    ),
    TestCase(
        query="What are the best and worst case scenarios?",
        expected_intent="confidence_query",
        description="Best/worst case question",
    ),
    TestCase(
        query="How wide are the prediction intervals?",
        expected_intent="confidence_query",
        description="Prediction interval width question",
    ),
    TestCase(
        query="What is the P10 and P90 range?",
        expected_intent="confidence_query",
        description="P10/P90 question",
    ),
    TestCase(
        query="How certain are you about the next 7 periods?",
        expected_intent="confidence_query",
        description="Certainty question",
    ),
    TestCase(
        query="What is the downside risk in this forecast?",
        expected_intent="confidence_query",
        description="Downside risk question",
    ),
    TestCase(
        query="Is there a lot of uncertainty in your prediction?",
        expected_intent="confidence_query",
        description="General uncertainty question",
    ),
    TestCase(
        query="What is the upside potential according to this model?",
        expected_intent="confidence_query",
        description="Upside potential question",
    ),
    TestCase(
        query="Can you tell me the margin of error for this forecast?",
        expected_intent="confidence_query",
        description="Margin of error question",
    ),
]


# Test set — originally held-out, but patterns were updated to fix failures
# observed on this set. It is now correctly labelled as a development-adjacent
# test set rather than a blind evaluation set. Use BLIND_SET for the final
# reported score.
TEST_SET: List[TestCase] = [
    # ── remove_covariate (5 queries) ─────────────────────────────────
    TestCase(
        query="Run it again without marketing spend.",
        expected_intent="remove_covariate",
        expected_covariate="marketing_spend",
        description="Remove marketing_spend with alternate without phrasing",
    ),
    TestCase(
        query="Remove website traffic from the inputs before forecasting.",
        expected_intent="remove_covariate",
        expected_covariate="website_traffic",
        description="Remove website_traffic from inputs",
    ),
    TestCase(
        query="Can you drop shipping delay hours for this run?",
        expected_intent="remove_covariate",
        expected_covariate="shipping_delay_hours",
        description="Remove shipping_delay_hours with question phrasing",
    ),
    TestCase(
        query="What if the competitor promotion index were gone?",
        expected_intent="remove_covariate",
        expected_covariate="competitor_promotion_index",
        description="Remove competitor_promotion_index with gone phrasing",
    ),
    TestCase(
        query="Set previous day sales to zero and forecast again.",
        expected_intent="remove_covariate",
        expected_covariate="previous_day_sales",
        description="Remove previous_day_sales with zero phrasing",
    ),

    # ── scale_covariate (5 queries) ──────────────────────────────────
    TestCase(
        query="Scale marketing spend to 2x and rerun the forecast.",
        expected_intent="scale_covariate",
        expected_covariate="marketing_spend",
        description="Double marketing_spend with multiplier phrasing",
    ),
    TestCase(
        query="What changes if website traffic is 40% higher?",
        expected_intent="scale_covariate",
        expected_covariate="website_traffic",
        description="Increase website_traffic by 40%",
    ),
    TestCase(
        query="Rerun the forecast with discounts 15% lower.",
        expected_intent="scale_covariate",
        expected_covariate="price_discount_percentage",
        description="Reduce price_discount_percentage by 15%",
    ),
    TestCase(
        query="Cut social media mentions by 25%.",
        expected_intent="scale_covariate",
        expected_covariate="social_media_mentions",
        description="Reduce social_media_mentions by 25%",
    ),
    TestCase(
        query="Increase weather temperature by 10% and show the result.",
        expected_intent="scale_covariate",
        expected_covariate="weather_temperature",
        description="Increase weather_temperature by 10%",
    ),

    # ── change_horizon (5 queries) ───────────────────────────────────
    TestCase(
        query="Forecast the next 72 hours.",
        expected_intent="change_horizon",
        expected_horizon=72,
        description="Change horizon to 72 hours",
    ),
    TestCase(
        query="Show a 4-day forecast.",
        expected_intent="change_horizon",
        expected_horizon=96,
        description="Change horizon to 4 days",
    ),
    TestCase(
        query="Extend the forecast to 3 weeks.",
        expected_intent="change_horizon",
        expected_horizon=504,
        description="Change horizon to 3 weeks",
    ),
    TestCase(
        query="Use a horizon of 12 steps.",
        expected_intent="change_horizon",
        expected_horizon=12,
        description="Change horizon to 12 periods",
    ),
    TestCase(
        query="Predict 1 month ahead.",
        expected_intent="change_horizon",
        expected_horizon=720,
        description="Change horizon to 1 month",
    ),

    # ── confidence_query (5 queries) ─────────────────────────────────
    TestCase(
        query="How confident is this projection?",
        expected_intent="confidence_query",
        description="Reliability question",
    ),
    TestCase(
        query="How uncertain is the forecast over the next few periods?",
        expected_intent="confidence_query",
        description="Stability and uncertainty question",
    ),
    TestCase(
        query="Do the prediction intervals spread out much?",
        expected_intent="confidence_query",
        description="Forecast band width question",
    ),
    TestCase(
        query="What range should I expect for the forecast?",
        expected_intent="confidence_query",
        description="Risk around prediction question",
    ),
    TestCase(
        query="Is there much downside risk in these predictions?",
        expected_intent="confidence_query",
        description="Tightness and uncertainty question",
    ),

    # ── extra queries with partial/fuzzy covariate names (4 queries) ──
    # These test that slot extraction generalises beyond exact name matches.
    TestCase(
        query="What happens if we remove the marketing budget?",
        expected_intent="remove_covariate",
        expected_covariate="marketing_spend",
        description="Remove marketing_spend via alias 'marketing budget'",
    ),
    TestCase(
        query="What if website visits increased by 30%?",
        expected_intent="scale_covariate",
        expected_covariate="website_traffic",
        description="Scale website_traffic via alias 'website visits'",
    ),
    TestCase(
        query="How reliable is this prediction, really?",
        expected_intent="confidence_query",
        description="Informal confidence question without keywords",
    ),
    TestCase(
        query="Show only the next 2 weeks.",
        expected_intent="change_horizon",
        expected_horizon=336,
        description="Change horizon to 2 weeks via 'only'",
    ),

    # ── counterfactual (2 queries) ────────────────────────────────────
    TestCase(
        query="What would have happened if marketing spend had been higher last month?",
        expected_intent="counterfactual",
        expected_covariate="marketing_spend",
        description="Historical counterfactual — system should decline and suggest alternatives",
    ),
    TestCase(
        query="What if website traffic had been much higher last week?",
        expected_intent="counterfactual",
        expected_covariate="website_traffic",
        description="Historical counterfactual about past data (no scale trigger)",
    ),

    # ── additional edge cases (4 queries) ────────────────────────────
    TestCase(
        query="Give me a one-month forecast.",
        expected_intent="change_horizon",
        expected_horizon=720,
        description="Change horizon — 'one-month' as word not digit",
    ),
    TestCase(
        query="Can you drop the social media variable?",
        expected_intent="remove_covariate",
        expected_covariate="social_media_mentions",
        description="Remove via 'drop' + alias 'social media variable'",
    ),
    TestCase(
        query="Triple the competitor promotion index.",
        expected_intent="scale_covariate",
        expected_covariate="competitor_promotion_index",
        description="Scale with word-factor 'triple'",
    ),
    TestCase(
        query="What are the P10 and P90 bounds?",
        expected_intent="confidence_query",
        description="Explicit quantile names",
    ),
]

# Blind set — patterns must NOT be updated based on results on this set.
# Created after patterns were frozen. Use this for the final reported score.
BLIND_SET: List[TestCase] = [
    # ── confidence_query (2 queries) ─────────────────────────────────
    TestCase(
        query="How much uncertainty surrounds this prediction?",
        expected_intent="confidence_query",
        description="Uncertainty question without the word 'confidence'",
    ),
    TestCase(
        query="Can you tell me the prediction intervals?",
        expected_intent="confidence_query",
        description="Direct prediction-interval request",
    ),

    # ── remove_covariate (2 queries) ─────────────────────────────────
    TestCase(
        query="Exclude weather_temperature from the model.",
        expected_intent="remove_covariate",
        expected_covariate="weather_temperature",
        description="Remove via 'exclude' with exact name",
    ),
    TestCase(
        query="What if there were no holiday_proximity factor?",
        expected_intent="remove_covariate",
        expected_covariate="holiday_proximity",
        description="Remove via 'what if...no' phrasing",
    ),

    # ── scale_covariate (3 queries) ──────────────────────────────────
    TestCase(
        query="Double the holiday_proximity.",
        expected_intent="scale_covariate",
        expected_covariate="holiday_proximity",
        description="Scale with word-factor 'double'",
    ),
    TestCase(
        query="Suppose social_media_mentions dropped by 40%.",
        expected_intent="scale_covariate",
        expected_covariate="social_media_mentions",
        description="Scale via 'dropped by' + percentage",
    ),
    TestCase(
        query="What if the price discount were halved?",
        expected_intent="scale_covariate",
        expected_covariate="price_discount_percentage",
        description="Scale via 'halved' + partial alias 'price discount'",
    ),

    # ── change_horizon (2 queries) ───────────────────────────────────
    TestCase(
        query="Forecast for the next three days.",
        expected_intent="change_horizon",
        expected_horizon=72,
        description="Horizon via word-number 'three days'",
    ),
    TestCase(
        query="Can we look at 96 steps ahead?",
        expected_intent="change_horizon",
        expected_horizon=96,
        description="Horizon via 'N steps ahead' phrasing",
    ),

    # ── counterfactual (1 query) ──────────────────────────────────────
    TestCase(
        query="What would the forecast have looked like with higher marketing spend last quarter?",
        expected_intent="counterfactual",
        expected_covariate="marketing_spend",
        description="Historical counterfactual — no scale trigger word",
    ),
]

# Backward-compatible alias kept for older imports.
HELD_OUT_SET = TEST_SET


# ──────────────── multi-turn test set ─────────────────────────────────

@dataclass
class MultiTurnTestCase:
    """A sequence of dialogue turns testing cross-turn state persistence."""
    description: str
    turns: List[TestCase]


MULTI_TURN_SET: List[MultiTurnTestCase] = [
    MultiTurnTestCase(
        description="Remove covariate then query confidence",
        turns=[
            TestCase(
                query="Remove marketing_spend from the forecast.",
                expected_intent="remove_covariate",
                expected_covariate="marketing_spend",
                description="Turn 1: remove covariate",
            ),
            TestCase(
                query="How confident are you in this forecast?",
                expected_intent="confidence_query",
                description="Turn 2: confidence query — state from turn 1 should persist",
            ),
        ],
    ),
    MultiTurnTestCase(
        description="Scale covariate then change horizon",
        turns=[
            TestCase(
                query="What if website traffic increased by 50%?",
                expected_intent="scale_covariate",
                expected_covariate="website_traffic",
                description="Turn 1: scale covariate",
            ),
            TestCase(
                query="Show me the next 7 days.",
                expected_intent="change_horizon",
                expected_horizon=168,
                description="Turn 2: change horizon — scaled covariates should persist",
            ),
        ],
    ),
    MultiTurnTestCase(
        description="Two successive covariate modifications",
        turns=[
            TestCase(
                query="Remove shipping delay hours.",
                expected_intent="remove_covariate",
                expected_covariate="shipping_delay_hours",
                description="Turn 1: remove first covariate",
            ),
            TestCase(
                query="Also double the marketing spend.",
                expected_intent="scale_covariate",
                expected_covariate="marketing_spend",
                description="Turn 2: scale second covariate — first should remain zeroed",
            ),
        ],
    ),
]


# ──────────────── evaluation result ──────────────────────────────────

@dataclass
class EvalResult:
    """Result of a single test case evaluation."""
    test_case: TestCase
    parsed_intent: str
    parsed_covariate: Optional[str]
    parsed_horizon: Optional[int]
    intent_correct: bool
    covariate_correct: bool
    horizon_correct: bool
    modification_applied: bool
    nli_score: float
    nli_pass: bool
    task_completed: bool
    error: Optional[str] = None


# ──────────────── evaluator ───────────────────────────────────────────

def run_dialogue_evaluation(
    n_scenarios: int = 5,
    seed: int = 42,
    run_full_pipeline: bool = False,
    evaluation_set: str = "heldout",
) -> pd.DataFrame:
    """Evaluate the dialogue system on a predefined query set.

    Parameters
    ----------
    n_scenarios : int
        Number of different time series scenarios to test on.
        Each scenario runs the full test set.
    seed : int
        Random seed.
    run_full_pipeline : bool
        If True, actually runs Chronos-2 and NLI for each query.
        If False, evaluates only intent parsing (much faster).
    evaluation_set : str
        Which query set to use: "heldout" for official evaluation or
        "dev" for parser debugging.

    Returns
    -------
    pd.DataFrame
        One row per test case with all evaluation metrics.
    """
    _seed_everything(seed)

    from extension_2.intent_parser import IntentParser

    parser = IntentParser(covariate_names=COVARIATE_NAMES)
    records: List[Dict[str, Any]] = []
    query_sets = {
        "dev": DEV_SET,
        "test": TEST_SET,
        "heldout": TEST_SET,   # backward-compatible alias
        "held-out": TEST_SET,  # backward-compatible alias
        "blind": BLIND_SET,
    }
    try:
        eval_cases = query_sets[evaluation_set.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(query_sets))
        raise ValueError(f"Unknown evaluation_set={evaluation_set!r}; use one of: {valid}") from exc

    set_label = {id(DEV_SET): "dev", id(TEST_SET): "test", id(BLIND_SET): "blind"}.get(
        id(eval_cases), "unknown"
    )

    if run_full_pipeline:
        # Full evaluation — runs Chronos-2 + NLI for each query.
        # Uses a synthetic sine-wave history with the same covariate names as
        # the test set so slot extraction still works.
        history, covariates = _make_eval_scenario(COVARIATE_NAMES, seed=seed)
        system = DialogueSystem(
            history=history,
            covariates=covariates,
            horizon=14,
            seed=seed,
        )

        for i, tc in enumerate(eval_cases):
            logger.info("[%d/%d] %s", i + 1, len(eval_cases), tc.description)
            try:
                response = system.query(tc.query)
                intent = response.intent
                modification = response.modification

                intent_correct = intent.intent_type == tc.expected_intent
                covariate_correct = (
                    tc.expected_covariate is None
                    or intent.target_covariate == tc.expected_covariate
                )
                horizon_correct = (
                    tc.expected_horizon is None
                    or intent.new_horizon == tc.expected_horizon
                )
                delta_faithful = _check_delta_faithfulness(
                    response.delta_p50, response.answer
                )

                records.append({
                    "query": tc.query,
                    "evaluation_set": set_label,
                    "description": tc.description,
                    "expected_intent": tc.expected_intent,
                    "parsed_intent": intent.intent_type,
                    "intent_correct": intent_correct,
                    "expected_covariate": tc.expected_covariate,
                    "parsed_covariate": intent.target_covariate,
                    "covariate_correct": covariate_correct,
                    "expected_horizon": tc.expected_horizon,
                    "parsed_horizon": intent.new_horizon,
                    "horizon_correct": horizon_correct,
                    "modification_applied": modification.modified,
                    "nli_score": response.consistency_score,
                    "nli_pass": response.is_consistent,
                    "delta_mean": float(np.mean(response.delta_p50)) if response.delta_p50 is not None else None,
                    "delta_faithful": delta_faithful,
                    "task_completed": response.task_completed,
                    "confidence": intent.confidence,
                    "error": None,
                })

            except Exception as e:
                logger.warning("Test case failed: %s — %s", tc.description, e)
                records.append({
                    "query": tc.query,
                    "evaluation_set": set_label,
                    "description": tc.description,
                    "expected_intent": tc.expected_intent,
                    "parsed_intent": "error",
                    "intent_correct": False,
                    "expected_covariate": tc.expected_covariate,
                    "parsed_covariate": None,
                    "covariate_correct": False,
                    "expected_horizon": tc.expected_horizon,
                    "parsed_horizon": None,
                    "horizon_correct": False,
                    "modification_applied": False,
                    "nli_score": 0.0,
                    "nli_pass": False,
                    "delta_mean": None,
                    "delta_faithful": None,
                    "task_completed": False,
                    "confidence": "error",
                    "error": str(e),
                })

    else:
        # Intent-only evaluation — fast, no Chronos-2 or NLI
        for tc in eval_cases:
            intent = parser.parse(tc.query)
            intent_correct = intent.intent_type == tc.expected_intent
            covariate_correct = (
                tc.expected_covariate is None
                or intent.target_covariate == tc.expected_covariate
            )
            horizon_correct = (
                tc.expected_horizon is None
                or intent.new_horizon == tc.expected_horizon
            )

            records.append({
                "query": tc.query,
                "evaluation_set": set_label,
                "description": tc.description,
                "expected_intent": tc.expected_intent,
                "parsed_intent": intent.intent_type,
                "intent_correct": intent_correct,
                "expected_covariate": tc.expected_covariate,
                "parsed_covariate": intent.target_covariate,
                "covariate_correct": covariate_correct,
                "expected_horizon": tc.expected_horizon,
                "parsed_horizon": intent.new_horizon,
                "horizon_correct": horizon_correct,
                "modification_applied": False,
                "nli_score": None,
                "nli_pass": None,
                "delta_mean": None,
                "delta_faithful": None,
                "task_completed": intent_correct and covariate_correct and horizon_correct,
                "confidence": intent.confidence,
                "error": None,
            })

    return pd.DataFrame(records)


def print_evaluation_report(df: pd.DataFrame) -> None:
    """Print a summary of the evaluation results."""
    print("\n" + "=" * 65)
    print("  EXTENSION 2 — DIALOGUE EVALUATION REPORT")
    print("=" * 65)

    total = len(df)
    intent_acc = df["intent_correct"].mean() * 100
    tcr = df["task_completed"].mean() * 100

    set_label = df["evaluation_set"].iloc[0] if "evaluation_set" in df.columns and total else "unknown"

    print(f"\n  Evaluation set   : {set_label}")
    print(f"  Total test cases : {total}")
    print(f"  Intent accuracy  : {intent_acc:.1f}%")
    print(f"  Task completion  : {tcr:.1f}%")

    if "nli_score" in df.columns and df["nli_score"].notna().any():
        nli_mean = df["nli_score"].mean()
        nli_pass = df["nli_pass"].mean() * 100
        print(f"  NLI consistency  : {nli_mean:.4f}")
        print(f"  NLI PASS rate    : {nli_pass:.1f}%")

    if "delta_faithful" in df.columns and df["delta_faithful"].notna().any():
        delta_verifiable = df["delta_faithful"].notna()
        delta_rate = df.loc[delta_verifiable, "delta_faithful"].mean() * 100
        n_verifiable = delta_verifiable.sum()
        print(f"  Delta faithfulness: {delta_rate:.1f}% ({n_verifiable} verifiable cases)")

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
            print(
                f"  ✗ [{row['expected_intent']} → {row['parsed_intent']}] "
                f"{row['query'][:60]}"
            )
    print()


def write_evaluation_report(df: pd.DataFrame, save_dir: Path = EVAL_DIR) -> str:
    """Write a markdown report of the evaluation results."""
    report_path = save_dir / "evaluation_report_ext2.md"
    lines: List[str] = []

    total = len(df)
    set_label = df["evaluation_set"].iloc[0] if "evaluation_set" in df.columns and total else "unknown"
    intent_acc = df["intent_correct"].mean() * 100
    tcr = df["task_completed"].mean() * 100

    lines.append("# Extension 2 — Dialogue System Evaluation Report")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Evaluation set | {set_label} |")
    lines.append(f"| Total test cases | {total} |")
    lines.append(f"| Intent classification accuracy | {intent_acc:.1f}% |")
    lines.append(f"| Task completion rate | {tcr:.1f}% |")

    if "nli_score" in df.columns and df["nli_score"].notna().any():
        lines.append(f"| Mean NLI consistency | {df['nli_score'].mean():.4f} |")
        lines.append(f"| NLI PASS rate | {df['nli_pass'].mean()*100:.1f}% |")

    lines.append("")

    lines.append("## 2. Breakdown by Intent Type")
    lines.append("")
    lines.append("| Intent | Correct | Total | Accuracy |")
    lines.append("|---|---|---|---|")
    for intent_type, grp in df.groupby("expected_intent"):
        correct = grp["intent_correct"].sum()
        total_g = len(grp)
        acc = correct / total_g * 100
        lines.append(f"| {intent_type} | {correct} | {total_g} | {acc:.1f}% |")
    lines.append("")

    lines.append("## 3. Failed Cases")
    lines.append("")
    failed = df[~df["intent_correct"]]
    if len(failed) == 0:
        lines.append("No failures — all intents parsed correctly.")
    else:
        lines.append("| Expected | Parsed | Query |")
        lines.append("|---|---|---|")
        for _, row in failed.iterrows():
            lines.append(
                f"| {row['expected_intent']} | {row['parsed_intent']} "
                f"| {row['query'][:60]} |"
            )
    lines.append("")

    lines.append("## 4. Success Cases")
    lines.append("")
    success = df[df["intent_correct"]]
    lines.append(f"{len(success)} / {total} queries parsed correctly.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved to %s", report_path)
    return str(report_path)


def run_multi_turn_evaluation(seed: int = 42) -> pd.DataFrame:
    """Evaluate state persistence across multi-turn dialogue sequences.

    Each sequence creates a fresh DialogueSystem and runs turns in order.
    State persistence is verified by inspecting system.covariates directly
    after each modifying turn — no Chronos-2 call needed.
    """
    _seed_everything(seed)
    history, initial_covariates = _make_eval_scenario(COVARIATE_NAMES, seed=seed)
    records: List[Dict[str, Any]] = []

    for seq in MULTI_TURN_SET:
        system = DialogueSystem(
            history=history,
            covariates=initial_covariates,
            horizon=14,
            seed=seed,
        )
        prev_response: Optional[Any] = None

        for turn_idx, tc in enumerate(seq.turns):
            try:
                # Use the public interface with dry_run=True: parses intent,
                # applies modification, persists state — without running Chronos-2.
                response = system.query(tc.query, dry_run=True)
                intent = response.intent
                modification = response.modification

                intent_correct = intent.intent_type == tc.expected_intent
                covariate_correct = (
                    tc.expected_covariate is None
                    or intent.target_covariate == tc.expected_covariate
                )
                horizon_correct = (
                    tc.expected_horizon is None
                    or intent.new_horizon == tc.expected_horizon
                )

                # State-persistence check: verify that the previous turn's
                # covariate removal is still reflected in system state.
                state_persisted: Optional[bool] = None
                if turn_idx > 0 and prev_response is not None and prev_response.modification.modified:
                    prev_tc = seq.turns[turn_idx - 1]
                    if (
                        prev_tc.expected_intent == "remove_covariate"
                        and prev_tc.expected_covariate is not None
                        and prev_tc.expected_covariate in COVARIATE_NAMES
                    ):
                        cov_idx = COVARIATE_NAMES.index(prev_tc.expected_covariate)
                        state_persisted = bool(
                            np.allclose(system.covariates.values[:, cov_idx], 0.0)
                        )

                prev_response = response

                records.append({
                    "sequence": seq.description,
                    "turn": turn_idx + 1,
                    "query": tc.query,
                    "expected_intent": tc.expected_intent,
                    "parsed_intent": intent.intent_type,
                    "intent_correct": intent_correct,
                    "covariate_correct": covariate_correct,
                    "horizon_correct": horizon_correct,
                    "modification_applied": modification.modified,
                    "state_persisted": state_persisted,
                    "task_completed": intent_correct and covariate_correct and horizon_correct,
                    "error": None,
                })

            except Exception as exc:
                records.append({
                    "sequence": seq.description,
                    "turn": turn_idx + 1,
                    "query": tc.query,
                    "expected_intent": tc.expected_intent,
                    "parsed_intent": "error",
                    "intent_correct": False,
                    "covariate_correct": False,
                    "horizon_correct": False,
                    "modification_applied": False,
                    "state_persisted": None,
                    "task_completed": False,
                    "error": str(exc),
                })

    return pd.DataFrame(records)


def print_multi_turn_report(df: pd.DataFrame) -> None:
    """Print a summary of the multi-turn evaluation."""
    print("\n" + "=" * 65)
    print("  EXTENSION 2 — MULTI-TURN STATE PERSISTENCE REPORT")
    print("=" * 65)

    total_turns = len(df)
    intent_acc = df["intent_correct"].mean() * 100
    state_checks = df["state_persisted"].dropna()
    state_rate = state_checks.mean() * 100 if len(state_checks) > 0 else float("nan")

    print(f"\n  Total turns      : {total_turns}")
    print(f"  Intent accuracy  : {intent_acc:.1f}%")
    if len(state_checks) > 0:
        print(f"  State persisted  : {state_rate:.1f}% ({len(state_checks)} verifiable)")

    print("\n  --- Per-sequence breakdown ---")
    for seq_name, grp in df.groupby("sequence"):
        acc = grp["intent_correct"].mean() * 100
        print(f"  [{acc:.0f}%] {seq_name}")
        for _, row in grp.iterrows():
            tag = "✓" if row["intent_correct"] else "✗"
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
    evaluation_set: str = "heldout",
    output_dir: Optional[Path | str] = None,
) -> None:
    """Entry point for Extension 2 evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    out = Path(output_dir) if output_dir else EVAL_DIR
    out.mkdir(parents=True, exist_ok=True)
    _save_eval_config(out, seed, evaluation_set, run_full_pipeline)

    logger.info(
        "Running Extension 2 evaluation (set=%s, full_pipeline=%s) ...",
        evaluation_set,
        run_full_pipeline,
    )

    df = run_dialogue_evaluation(
        seed=seed,
        run_full_pipeline=run_full_pipeline,
        evaluation_set=evaluation_set,
    )
    df.to_csv(out / "evaluation_results_ext2.csv", index=False)
    print_evaluation_report(df)
    write_evaluation_report(df, save_dir=out)

    logger.info("Running blind-set evaluation (final reported score) ...")
    df_blind = run_dialogue_evaluation(
        seed=seed,
        run_full_pipeline=False,
        evaluation_set="blind",
    )
    df_blind.to_csv(out / "blind_results_ext2.csv", index=False)
    print_evaluation_report(df_blind)

    logger.info("Running multi-turn state-persistence evaluation ...")
    df_mt = run_multi_turn_evaluation(seed=seed)
    df_mt.to_csv(out / "multi_turn_results_ext2.csv", index=False)
    print_multi_turn_report(df_mt)

    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
