# Explainable Chronos

A NLP framework for [Chronos-2](https://github.com/amazon-science/chronos-forecasting) at inference time to produce natural-language explanations of probabilistic time-series forecasts.

## Extension 1: Verbalization

Chronos-2 only outputs numbers: a P10, P50, and P90 curve over the forecast horizon. Extension 1 turns those numbers into a short paragraph that explains what is being predicted and why.

- Extracts simple facts from the forecast itself: trend direction, predicted magnitude, uncertainty width
- Traces the model's attention weights back to the original inputs with attention rollout, to find which covariate mattered most and which past time window it focused on
- Feeds these extracted facts, into a language model that writes a fluent paragraph grounded strictly in those facts
- Scores the generated sentences for consistency, checking that each one is actually entailed by the fact it describes

```
   P10 / P50 / P90        attention weights
        |                       |
        v                       v
 [ feature extractor ]   [ attention rollout ]
        |                       |
        v                       v
   trend, magnitude,     covariate ranking,
   uncertainty width     peak time, focus breadth
        \                       /
         \                     /
          v                   v
         [ structured facts ]
                  |
                  v
         [ language model ]
                  |
                  v
       natural-language explanation
                  |
                  v
        [ consistency scoring ]
```

## Reproducibility

All results and demos in this project can be reproduced from the notebooks in [notebooks/](notebooks/):

- [notebooks/ex_1_run_evaluation.ipynb](notebooks/ex_1_run_evaluation.ipynb) — Extension 1 evaluation
- [notebooks/ex_2_run_evaluation.ipynb](notebooks/ex_2_run_evaluation.ipynb) — Extension 2 evaluation

