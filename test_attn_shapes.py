import torch
from extension_1.config import PipelineConfig
from extension_1.pipeline import VerbalizationPipeline
from extension_1.features.extractor import extract_features
from shared.forecast_provider import ChronosForecastProvider
from extension_1.attribution.types import CovariateSet
import numpy as np

def test_shapes():
    provider = ChronosForecastProvider(enable_attention=True)
    history = np.random.randn(100)
    past_cov = {
        "temp": np.random.randn(100),
        "rain": np.random.randn(100)
    }
    future_cov = {
        "temp": np.random.randn(14),
        "rain": np.random.randn(14)
    }
    
    result, attns = provider.predict(
        history, 
        horizon=14, 
        past_covariates=past_cov, 
        future_covariates=future_cov
    )
    
    print("Time Attn Shape:", [a.shape for a in attns['attentions']['enc_time_self_attn_weights']])
    
    group_attn = attns['attentions'].get('enc_group_self_attn_weights')
    if group_attn is not None:
        print("Group Attn Shape:", [a.shape for a in group_attn])
    else:
        print("No group attn")

if __name__ == "__main__":
    test_shapes()
