# Evaluation Report — Extension 1

> Mode: **full** — Full mode — 200 windows, exhaustive evaluation
> History: 512 steps | Horizon: 96 steps
> ETTh1 and Weather use their official test splits for comparability with published benchmarks.

## 1. Overview

| Metric | Value |
|---|---|
| Total windows evaluated | 400 |
| Datasets | Weather |
| Mean NLI consistency | 0.7095 |
| PASS rate (≥ 0.70) | 59.8% |
| Mean semantic similarity vs template | 0.8267 |
| Mean MASE | 0.9981 |
| Mean fair_mase | 0.4251 |
| Mean coverage | 91.4% |

## 2. Forecast Accuracy by Dataset

| Dataset | Mean MASE | Mean First-Step MASE | Mean fair_mase | Mean Coverage | Mean Sharpness |
|---|---|---|---|---|---|
| Weather | 0.9981 | 0.8782 | 0.4251 | 91.4% | 0.4771 |

## 3. NLI Consistency by Persona and Dataset

| Persona | Dataset | Mean NLI | Std | PASS rate |
|---|---|---|---|---|
| analyst | Weather | 0.7333 | 0.0754 | 71.5% |
| executive | Weather | 0.6858 | 0.0868 | 48.0% |

## 4. Semantic Similarity vs Template by Persona

| Persona | Dataset | Mean | Std | Min |
|---|---|---|---|---|
| analyst | Weather | 0.8640 | 0.0351 | 0.6778 |
| executive | Weather | 0.7893 | 0.0537 | 0.6265 |

## 5. Covariate Attribution (Attention Rollout)

- Top covariates: temp_min (398), wind (2)

## Visualizations

![Evaluation plots](evaluation_plots.png)
