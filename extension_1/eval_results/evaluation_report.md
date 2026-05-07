# Evaluation Report — Extension 1

> This evaluation separates two questions: (1) Is the explanation faithful to the forecast? (2) Is the forecast itself accurate? A high-scoring explanation of an inaccurate forecast is still a correct explanation.

## 1. Overview

| Metric | Value |
|---|---|
| Scenarios evaluated | 20 |
| Forecast model | `autogluon/chronos-2-small` |
| NLI model | `facebook/bart-large-mnli` |
| Consistency threshold | 0.70 |
| Mean consistency | 0.9341 |
| Std consistency | 0.1184 |
| Min consistency | 0.6697 |
| Max consistency | 0.9989 |
| % consistent (≥ 0.7) | 85.0% |

## 2. Breakdown by Verbalizer Type

| Verbalizer | Mean | Std | Min | Max | Count |
|---|---|---|---|---|---|
| Template | 0.9341 | 0.1184 | 0.6697 | 0.9989 | 20 |

## 3. Breakdown by Scenario Type

| Scenario | Mean | Std | Min | Max | Count |
|---|---|---|---|---|---|
| flat | 0.7798 | 0.1268 | 0.6699 | 0.8898 | 4 |
| noisy | 0.9928 | 0.0041 | 0.9879 | 0.9963 | 4 |
| seasonal | 0.9864 | 0.0176 | 0.9610 | 0.9987 | 4 |
| trending | 0.9162 | 0.1643 | 0.6697 | 0.9989 | 4 |
| volatile | 0.9955 | 0.0032 | 0.9908 | 0.9980 | 4 |

## 4. Feature Distribution

**trend_direction**: flat (20)
**trend_magnitude**: slightly (20)
**uncertainty_level**: moderate (7), low (7), high (6)
**uncertainty_trend**: widening (18), stable (2)
**asymmetry_label**: symmetric (18), left_skewed (2)

- Downside risk flagged: 10 / 20
- Upside potential flagged: 6 / 20
- Regime shift detected: 11 / 20

## 5. Covariate Attribution Summary

- Mean surrogate R²: 0.3222
- Top covariate distribution: competitor_promotion_index (4), marketing_spend (4), previous_day_sales (1), holiday_proximity (1)
- Mean top covariate impact: 45.2%

## 6. Explanation Faithfulness (NLI consistency + quantile round-trip)

- Quantile round-trip mismatches: 0 / 20

## 7. Forecast Accuracy (model vs. synthetic actuals)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| mase | 2.6196 | 1.8835 | 0.1569 | 6.8296 |
| coverage_pct | 76.2143 | 24.6040 | 20.0000 | 100.0000 |
| interval_sharpness | 1.2868 | 0.6446 | 0.2001 | 2.2413 |

## 8. 5 Lowest-Scoring Sentences

| Score | Verbalizer | Scenario | Index |
|---|---|---|---|
| 0.010 | Template | flat | 2 |
| 0.011 | Template | trending | 5 |
| 0.011 | Template | flat | 2 |
| 0.012 | Template | flat | 7 |
| 0.012 | Template | flat | 7 |

## 9. 5 Highest-Scoring Sentences

| Score | Verbalizer | Scenario | Index |
|---|---|---|---|
| 1.000 | Template | seasonal | 8 |
| 1.000 | Template | noisy | 9 |
| 1.000 | Template | flat | 2 |
| 1.000 | Template | trending | 0 |
| 1.000 | Template | flat | 2 |

## 10. Failure Analysis

**3 scenarios** scored below the 0.70 threshold:

### Scenario 2 (flat / Template)
- Overall score: 0.6699
- sent_0_score: 0.9996
- sent_1_score: 0.0111
- sent_2_score: 0.9990

### Scenario 5 (trending / Template)
- Overall score: 0.6697
- sent_0_score: 0.9995
- sent_1_score: 0.0106
- sent_2_score: 0.9990

### Scenario 7 (flat / Template)
- Overall score: 0.6701
- sent_0_score: 0.9995
- sent_1_score: 0.0117
- sent_2_score: 0.9990


## 11. RST Relation Distribution

| Relation | Count |
|---|---|
| elaboration | 11 |
| cause | 10 |
| contrast | 9 |

## Visualizations

![Evaluation plots](evaluation_plots.png)
