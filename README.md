# explainable-chronos

Run commands from the repository root:

```powershell
python run_extensions.py ext1 <experiment> [options]
```

## Repository Layout

```text
explainable-chronos/
├── run_extensions.py          # CLI entry point
├── requirements.txt
├── shared/
│   ├── forecast_provider.py   # Chronos-2 inference wrapper
│   └── __init__.py
├── extension_1/
│   ├── config.py              # All thresholds and constants
│   ├── pipeline.py            # End-to-end orchestration
│   ├── features/
│   │   └── extractor.py       # Forecast feature extraction
│   ├── verbalization/
│   │   ├── template.py        # RST-based TemplateVerbalizer (Approach A)
│   │   └── llm.py             # LLM-refined LLMVerbalizer (Approach B)
│   ├── attribution/
│   │   ├── base.py            # CovariateSet, AttributionResult, factory
│   │   ├── shap.py            # SurrogateExplainer (XGBoost + SHAP)
│   │   └── attention.py       # AttentionAttributor (Attention Rollout)
│   └── evaluation/
│       ├── runner.py          # Benchmark evaluation loop
│       ├── scoring.py         # NLI consistency scorer
│       ├── trace.py           # Per-scenario trace visualisation
│       └── judge.py           # LLM-as-judge pairwise comparison
└── results/
    └── extension_1/           # All evaluation outputs
```

## Experiment Tree

```text
ext1
`-- evaluate
    |-- Purpose: benchmark evaluation on real datasets
    |-- Default datasets: etth1, ettm1, weather, sp500
    |-- Default mode: dev
    |-- Default verbalizers: Template, LLM
    |-- Default attribution method: shap
    `-- Options: --dataset, --mode, --verbalizers, --attribution-method, --save-traces, --judge
```

## Commands

```powershell
python run_extensions.py ext1 evaluate
python run_extensions.py ext1 evaluate --dataset etth1
python run_extensions.py ext1 evaluate --dataset etth1 ettm1 weather sp500
python run_extensions.py ext1 evaluate --mode dev
python run_extensions.py ext1 evaluate --mode full
python run_extensions.py ext1 evaluate --verbalizers template
python run_extensions.py ext1 evaluate --verbalizers template llm
python run_extensions.py ext1 evaluate --attribution-method shap
python run_extensions.py ext1 evaluate --attribution-method attention
```

Evaluation modes:

| `--mode` | Windows | History length | Forecast horizon | Intended use |
|---|---:|---:|---:|---|
| `dev` | 5 | 512 | 96 | Fast hourly-series evaluation |
| `paper` | 20 | 512 | 96 | Fuller hourly-series reporting |
| `dev_daily` | 5 | 252 | 5 | Fast daily-series evaluation |
| `paper_daily` | 20 | 252 | 30 | Fuller daily-series reporting |

Outputs are saved under:

```text
results/extension_1/
├── evaluation_results.csv
├── results_ETTh1.csv
├── results_ETTm1.csv
├── results_Weather.csv
├── results_SP500.csv
├── evaluation_plots.png
└── evaluation_report.md
```

## CLI Options

| Option | Values | Default | Used by |
|---|---|---|---|
| `action` | `evaluate` | required | all |
| `--dataset` | `etth1`, `ettm1`, `weather`, `sp500` | all datasets | `evaluate` |
| `--mode` | `dev`, `full` | `dev` | `evaluate` |
| `--verbalizers` | `template`, `llm` | both | `evaluate` |
| `--attribution-method` | `shap`, `attention` | `shap` | `evaluate` |

## Notes

- Both past and future covariates are required per window; windows where either cannot be extracted are skipped.
- `--seed` is not exposed through the CLI; the default seed `42` is used.
