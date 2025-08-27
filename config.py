"""
Configuration file for all model parameters
Centralized configuration for easy parameter management
"""

# Deep Learning Models Configuration
DEEP_MODELS_CONFIG = {
    "timevae": {
        "epochs": 200,
        "latent_dim": 8,
        "hidden_layer_sizes": [50, 100, 200],
        "trend_poly": 0,
        "custom_seas": None,
        "use_residual_conn": True,
        "learning_rate": 0.001,
        "batch_size": 16,
        "verbose": 1,
        "reconstruction_wt": 3.0,
    },
    "timegan": {
        "epochs": 1000,
        "hidden_dim": 24,
        "num_layers": 3,
        "z_dim": None,  # Will be set to feat_dim automatically
        "batch_size": 128,
        "learning_rate": 0.001,
    },
    "fourierflow": {
        "epochs": 300,
        "hidden": 200,  # hidden units in networks
        "fft_size": 125,  # FFT size (use sequence length)
        "n_flows": 10,  # Number of flow transformations
        "FFT": True,  # Use FFT transformation
        "flip": True,  # Alternate flip patterns
        "normalize": False,  # Normalize spectral components
        "batch_size": 128,
        "learning_rate": 1e-3,
        "display_step": 100,
    },
    "timetransformer": {
        "epochs": 50,
        "latent_dim": 16,
        "hidden_layer_sizes": [64, 128, 256],
        "dilations": [1, 2, 4],  # Controls the number of TimeSformer layers
        "k_size": 4,
        "head_size": 64,  # Must be divisible by num_heads (96/8 = 12)
        "num_heads": 4,  # Changed from 3 to 8
        "dropout": 0.1,
        "reconstruction_wt": 3.0,
        "batch_size": 32,
        "learning_rate": 0.001,
    },
}

# Time Series Models Configuration
TIMESERIES_MODELS_CONFIG = {
    "arima": {
        "order": (1, 1, 1),  # (p, d, q) parameters
        "seasonal_order": (0, 0, 0, 0),  # (P, D, Q, s) - no seasonality
        "trend": "n",  # 'n' for no trend (required when d=1, as differencing eliminates constant trend)
        # Note: 'method' and 'maxiter' are not valid ARIMA parameters
    },
    "exp_smooth": {
        "seasonal_periods": 7,
        "trend": "add",  # 'add', 'mul', 'additive', 'multiplicative'
        "seasonal": "add",  # 'add', 'mul', 'additive', 'multiplicative'
        "damped_trend": False,
        "use_boxcox": False,
    },
}

# Model Selection Configuration
MODEL_SELECTION_CONFIG = {
    "available_models": [
        "timevae",
        "timegan",
        "fourierflow",
        "timetransformer",
        "arima",
        "exp_smooth",
    ],
    "deep_models": ["timevae", "timegan", "fourierflow", "timetransformer"],
    "timeseries_models": ["arima", "exp_smooth"],
}


def get_model_config(model_type: str) -> dict:
    """Get configuration for a specific model type"""
    if model_type in DEEP_MODELS_CONFIG:
        return DEEP_MODELS_CONFIG[model_type]
    elif model_type in TIMESERIES_MODELS_CONFIG:
        return TIMESERIES_MODELS_CONFIG[model_type]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
