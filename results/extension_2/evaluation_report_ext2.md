# Extension 2 — Dialogue System Evaluation Report

## 1. Overview

| Metric | Value |
|---|---|
| Evaluation set | test |
| Total test cases | 60 |
| Intent classification accuracy | 93.3% |
| Task completion rate | 83.3% |

## 2. Accuracy by Tier

| Tier | Queries | Accuracy |
|---|---|---|
| rule | 40 | 92.5% |
| bert | 19 | 94.7% |
| llm | 1 | 100.0% |

## 3. Breakdown by Intent Type

| Intent | Correct | Total | Accuracy |
|---|---|---|---|
| change_horizon | 12 | 12 | 100.0% |
| confidence_query | 12 | 12 | 100.0% |
| counterfactual | 5 | 6 | 83.3% |
| remove_covariate | 11 | 14 | 78.6% |
| scale_covariate | 16 | 16 | 100.0% |

## 4. Failed Cases

| Expected | Parsed | Tier | Query |
|---|---|---|---|
| counterfactual | scale_covariate | rule | What if sales yesterday had been 20% higher? |
| remove_covariate | counterfactual | rule | What if we paused all advertising? |
| remove_covariate | scale_covariate | bert | Kill the shipping delays. |
| remove_covariate | counterfactual | rule | What happens if we ditch the weather input? |

## 5. Success Cases

56 / 60 queries parsed correctly.
