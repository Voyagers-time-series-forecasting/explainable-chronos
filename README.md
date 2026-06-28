# Explainable Chronos

A NLP framework for [Chronos-2](https://github.com/amazon-science/chronos-forecasting) at inference time to produce natural-language explanations of probabilistic time-series forecasts.

## Extension 1: Verbalization

Chronos-2 only outputs numbers: a P10, P50, and P90 curve over the forecast horizon. Extension 1 turns those numbers into a short paragraph that explains what is being predicted and why.

It works in three steps. First, it reads the forecast itself and pulls out simple facts like the trend direction, the predicted magnitude, and how wide the uncertainty band is. Second, it looks inside the model's attention weights and traces them back to the original inputs using attention rollout, so it can tell which covariate the model relied on most and which point in the past it focused on. Third, it feeds these extracted facts, not the raw model internals, into a language model that writes a fluent paragraph grounded strictly in those facts. A consistency check then verifies that the generated sentences are actually entailed by the facts they are supposed to describe.

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

