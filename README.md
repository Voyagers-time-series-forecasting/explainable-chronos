# explainable-chronos

Run commands from the repository root:

```powershell
python run_extensions.py ext1 <experiment> [options]
```

## Experiment Tree

```text
ext1
|-- demo-uni
|   |-- Purpose: one quick univariate forecast verbalization demo
|   |-- Default history: 50 synthetic points
|   |-- Default horizon: 14 forecast steps
|   |-- Default seed: 42
|   |-- Default verbalizer: Template
|   |-- Default attribution: none, because there are no covariates
|   `-- Options: --seed
|
|-- demo-cov
|   |-- Purpose: one quick demo with synthetic covariates and attribution
|   |-- Default history: 50 synthetic points
|   |-- Default horizon: 14 forecast steps
|   |-- Default seed: 42
|   |-- Default covariates: 10 synthetic covariates
|   |-- Default verbalizer: Template
|   |-- Default attribution method: shap
|   `-- Options: --seed, --attribution-method
|
|-- evaluate
|   |-- Purpose: synthetic benchmark evaluation
|   |-- Default scenarios: 10 without covariates + 10 with covariates
|   |-- Default seed: 42
|   |-- Scenario history lengths: sampled from 30, 50, 100, 200
|   |-- Scenario forecast horizons: sampled from 7, 14, 30
|   |-- Scenario types: trending, volatile, flat, seasonal, noisy
|   |-- Default covariates: second pass uses 10 synthetic covariates
|   |-- Default verbalizers: Template, LLM Guided, LLM Raw
|   |-- Fallback: Template only if LLMs cannot be loaded
|   `-- Options: none wired through the top-level CLI
|
`-- evaluate-real
    |-- Purpose: real-dataset benchmark evaluation
    |-- Default datasets: etth1, ettm1, weather, sp500
    |-- Default mode: dev
    |-- Default seed: 42
    |-- Default verbalizers: Template, LLM Guided, LLM Raw
    |-- Default attribution method: shap
    |-- Evaluates each window twice when covariates exist:
    |   |-- univariate
    |   `-- with_covariates
    `-- Options: --dataset, --mode, --verbalizers, --attribution-method
```

## Commands

### Demo Experiments

```powershell
python run_extensions.py ext1 demo-uni
python run_extensions.py ext1 demo-uni --seed 42

python run_extensions.py ext1 demo-cov
python run_extensions.py ext1 demo-cov --seed 42
python run_extensions.py ext1 demo-cov --attribution-method shap
python run_extensions.py ext1 demo-cov --attribution-method attention
```

`demo-uni` and `demo-cov` use the general pipeline default horizon from
`PipelineConfig`: 14 forecast steps.

What the demo outputs mean:

| Command | Output explanation |
|---|---|
| `python run_extensions.py ext1 demo-uni` | Prints a univariate forecast report: Chronos predicts P10/P50/P90 intervals, `feature_extractor.py` derives trend, uncertainty, downside/upside risk, and regime-shift features, then `TemplateVerbalizer` turns those features into the printed `Summary`. |
| `python run_extensions.py ext1 demo-cov` | Prints the same forecast report plus `Top covariates`; by default, covariate importance is computed with SHAP over the synthetic covariates and added to the summary text. |
| `python run_extensions.py ext1 demo-cov --attribution-method shap` | Same as `demo-cov`, but explicitly selects SHAP attribution, so the output includes covariate direction, relative impact percentages, and surrogate `R^2`. |
| `python run_extensions.py ext1 demo-cov --attribution-method attention` | Uses attention-based attribution instead of SHAP, so `Top covariates` comes from attention-derived importance scores while the summary is still produced from forecast features plus attribution. |

### Synthetic Evaluation

```powershell
python run_extensions.py ext1 evaluate
```

The output compares multiple synthetic scenarios and reports consistency,
faithfulness, attribution, and forecast metrics before saving CSV, plot, and
Markdown report files.

The synthetic evaluation does not use `--mode`. It generates synthetic
history/future pairs and passes each scenario's own future length as the
forecast horizon. Those horizons are sampled from `7`, `14`, and `30`.

Outputs are saved under:

```text
extension_1/eval_results/
|-- results_univariate.csv
|-- results_covariates.csv
|-- evaluation_results.csv
|-- evaluation_plots.png
`-- evaluation_report.md
```

### Real-Data Evaluation

```powershell
python run_extensions.py ext1 evaluate-real
python run_extensions.py ext1 evaluate-real --dataset etth1
python run_extensions.py ext1 evaluate-real --dataset etth1 ettm1 weather sp500
python run_extensions.py ext1 evaluate-real --mode dev
python run_extensions.py ext1 evaluate-real --mode paper
python run_extensions.py ext1 evaluate-real --mode dev_daily
python run_extensions.py ext1 evaluate-real --mode paper_daily
python run_extensions.py ext1 evaluate-real --verbalizers template
python run_extensions.py ext1 evaluate-real --verbalizers template llm_guided llm_raw
python run_extensions.py ext1 evaluate-real --attribution-method shap
python run_extensions.py ext1 evaluate-real --attribution-method attention
```

What the real-data output means:

| Command | Output explanation |
|---|---|
| `python run_extensions.py ext1 evaluate-real --verbalizers template` | Runs the real-dataset benchmark using only the deterministic template verbalizer and prints grouped metrics such as `mase`, `fair_mase`, `coverage_pct`, and NLI consistency by verbalizer. |
| `python run_extensions.py ext1 evaluate-real` | Runs the same real-data benchmark with all default verbalizers, evaluating univariate and covariate versions when covariates are available. |
| `python run_extensions.py ext1 evaluate-real --dataset etth1` | Limits the real-data output to one dataset, making the saved CSV and report easier to inspect for that dataset. |
| `python run_extensions.py ext1 evaluate-real --mode dev` | Uses the fast real-data setting with fewer windows, so it is mainly for checking that the pipeline runs end to end. |
| `python run_extensions.py ext1 evaluate-real --mode paper` | Uses more windows for fuller reporting, so the metric tables are more stable but the run takes longer. |
| `python run_extensions.py ext1 evaluate-real --attribution-method shap` | Uses SHAP for covariate runs, so the report includes top covariates and surrogate attribution quality. |
| `python run_extensions.py ext1 evaluate-real --attribution-method attention` | Uses attention-derived covariate importance for covariate runs instead of SHAP. |

Real-data modes:

| `--mode` | Windows | History length | Forecast horizon | Intended use |
|---|---:|---:|---:|---|
| `dev` | 5 | 512 | 96 | Fast hourly-series evaluation |
| `paper` | 20 | 512 | 96 | Fuller hourly-series reporting |
| `dev_daily` | 5 | 252 | 5 | Fast daily-series evaluation |
| `paper_daily` | 20 | 252 | 30 | Fuller daily-series reporting |

Outputs are saved under:

```text
extension_1/eval_results/
|-- evaluation_real_results.csv
|-- results_real_ETTh1.csv
|-- results_real_ETTm1.csv
|-- results_real_Weather.csv
|-- results_real_SP500.csv
|-- evaluation_real_plots.png
`-- evaluation_report_real.md
```

## CLI Options

| Option | Values | Default | Used by |
|---|---|---|---|
| `action` | `demo-uni`, `demo-cov`, `evaluate`, `evaluate-real` | required | all |
| `--seed` | integer | `42` | demos only |
| `--dataset` | `etth1`, `ettm1`, `weather`, `sp500` | all datasets | `evaluate-real` |
| `--mode` | `dev`, `paper`, `dev_daily`, `paper_daily` | `dev` | `evaluate-real` |
| `--verbalizers` | `template`, `llm_guided`, `llm_raw` | all three | `evaluate-real` |
| `--attribution-method` | `shap`, `attention` | `shap` | `demo-cov`, `evaluate-real` |

## Notes

- `--mode` only affects `evaluate-real`.
- `evaluate` has its own synthetic scenario generator and does not read `--mode`.
- `--seed` is currently wired through the top-level runner only for demo commands.
- Attention attribution requires covariates, so it is valid for `demo-cov` and
  real-data covariate runs.
