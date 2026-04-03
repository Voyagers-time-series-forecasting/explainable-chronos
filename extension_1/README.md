# Extension 5.1 — Post-Hoc Verbalization of Forecast Intervals

Convert Chronos-2 probabilistic quantile forecasts (P10 / P50 / P90) into
human-readable natural-language summaries and verify their factual
consistency using a Natural Language Inference (NLI) model.

## Architecture

```
  ┌─────────────────────────────┐
  │  Chronos-2 Quantile Forecast │  ← forecast_provider.py
  │  (or MockForecastProvider)   │
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │  Numerical Feature Extraction│  ← feature_extractor.py
  │  (numpy / scipy — no ML)     │
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │  Template-Based Verbalization│  ← verbalizer.py (Approach A)
  │  + Optional LLM Refinement  │    (Approach B — prompt only)
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │  NLI Factual Consistency     │  ← consistency_scorer.py
  │  Scoring (MNLI / BART-MNLI)  │
  └─────────────────────────────┘
```

Orchestration via **pipeline.py**; bulk evaluation via **evaluation.py**.

## Installation

```bash
cd extension_1
pip install -r requirements.txt
```

> **Note:** `chronos-forecasting` is optional.  The pipeline ships with a
> `MockForecastProvider` that generates realistic synthetic forecasts for
> offline development and testing.

## Quick Start

```python
from pipeline import run_demo

# Run with synthetic data (no GPU needed)
run_demo(scenario="trending")
run_demo(scenario="volatile")
run_demo(scenario="flat")
```

Or from the command line:

```bash
python pipeline.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Evaluation

```bash
python evaluation.py
```

Generates:

* **evaluation_results.csv** — per-scenario metrics.
* **evaluation_plots.png** — box plot + scatter visualisation.

## Key Modules

| Module                  | Purpose                                                |
| ----------------------- | ------------------------------------------------------ |
| `config.py`             | All thresholds, seeds, model names                     |
| `forecast_provider.py`  | Chronos-2 interface + mock provider                    |
| `feature_extractor.py`  | Derive trend, uncertainty, risk, regime-shift features |
| `verbalizer.py`         | Rule-based sentence planner + optional LLM prompt      |
| `consistency_scorer.py` | NLI entailment scoring per sentence                    |
| `pipeline.py`           | End-to-end orchestration                               |
| `evaluation.py`         | Systematic evaluation over 50 synthetic scenarios      |

## Consistency Metric

For **each** verbalized sentence the scorer constructs a *(premise,
hypothesis)* pair:

* **premise** – structured English rendering of the numerical features
  that generated the sentence (e.g. "The median slope is +0.034 …").
* **hypothesis** – the verbalized sentence itself.

An NLI model (default: `facebook/bart-large-mnli`) estimates the
probability that the premise **entails** the hypothesis.  The overall
consistency score is the mean entailment probability across all
sentences.  A score ≥ 0.70 is considered **consistent**.

## File Structure

```
extension_1/
├── README.md
├── requirements.txt
├── config.py
├── forecast_provider.py
├── feature_extractor.py
├── verbalizer.py
├── consistency_scorer.py
├── pipeline.py
├── evaluation.py
└── tests/
    ├── test_feature_extractor.py
    ├── test_verbalizer.py
    └── test_consistency_scorer.py
```
