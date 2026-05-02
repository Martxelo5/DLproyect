import torch.nn as nn

# Here we made three models: Under_parameterized, baseline model and over-parameterized.
def get_tiny_model():
    """Under-parameterized model (High Bias)"""
    return nn.Sequential(
        nn.Linear(9, 2),
        nn.ReLU(),
        nn.Linear(2, 1)
    )

def get_Baseline_model():
    """Baseline model"""
    return nn.Sequential(
        nn.Linear(9, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

def get_massive_model():
    """Over-parameterized model (High Variance)"""
    return nn.Sequential(
        nn.Linear(9, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )