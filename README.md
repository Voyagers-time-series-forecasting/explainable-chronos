# Explainable Chronos

A modular NLP framework that sits on top of [Chronos-2](https://github.com/amazon-science/chronos-forecasting) at inference time to produce natural-language explanations of probabilistic time-series forecasts.

## Overview

**Extension 1 — Verbalization and Consistency Scoring**

Given a Chronos-2 quantile forecast (P10/P50/P90), the pipeline:
1. Extracts interpretable numerical features (trend, uncertainty, regime shift, trajectory, covariate attribution)
2. Computes covariate importance and temporal saliency via Attention Rollout
3. Generates a natural-language summary via a Template or LLM verbalizer
4. Scores each sentence for factual consistency using NLI (BART-large-MNLI)

## Repository Layout

```
explainable-chronos/
├── run_extensions.py          # CLI entry point
├── requirements.txt
├── notebooks/
│   ├── demo.ipynb             # Framework demo — three synthetic scenarios with visualisations
│   └── run_evaluation.ipynb   # Full benchmark evaluation (Colab-ready)
├── shared/
│   └── forecast_provider.py   # Chronos-2 inference wrapper
├── extension_1/
│   ├── config.py              # Thresholds and constants
│   ├── pipeline.py            # End-to-end orchestration
│   ├── features/
│   │   └── extractor.py       # Forecast feature extraction
│   ├── verbalization/
│   │   ├── template.py        # Template verbalizer
│   │   ├── llm.py             # LLM verbalizer (Qwen2.5-7B-Instruct)
│   │   └── trajectory.py      # Trajectory narration helpers
│   ├── attribution/
│   │   ├── attention.py       # Attention Rollout attributor
│   │   └── types.py           # Attribution data types
│   └── evaluation/
│       ├── runner.py          # Benchmark evaluation loop
│       ├── scoring.py         # NLI consistency scorer
│       ├── factuality.py      # Fact recall and feature completeness
│       ├── judge.py           # LLM-as-judge pairwise comparison
│       └── trace.py           # Per-scenario trace visualisation
└── results/
    └── extension_1/           # All evaluation outputs
```

## Quick Start

**► Try the demo first:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb) runs the full pipeline on three synthetic sales scenarios (growth, decline, volatile market) and shows forecast plots, covariate attribution, temporal saliency, the generated verbalization, and per-sentence NLI consistency scores — no dataset download required.

**Benchmark evaluation:** [`notebooks/run_evaluation.ipynb`](notebooks/run_evaluation.ipynb) runs the full evaluation on real datasets (Seattle Weather, ETTh1, ETTm1, S&P 500) and produces CSV results, summary plots, and a markdown report.
