**This codebase is part of the Master Thesis by Matej Kutirov with title "Automated Model Selection for Time Series Data Generation".**

# Time Series Generation Pipeline

A comprehensive pipeline for generating synthetic time series data using various deep learning and statistical models, with automated model selection capabilities using FLAML AutoML.

## Installation

```bash
# Clone repo
git clone https://github.com/matejkutirov/AutoTSG.git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

## Quick Start

### Standard Pipeline

```bash
# Single model
python main.py --ds google --model timevae

# Multiple models
python main.py --ds google --model timevae timegan arima

# All models
python main.py --ds google --model all
```

### AutoML Pipeline

```bash
# AutoML with time budget (in seconds)
python main.py --ds google --model all --ms automl --time_budget 300

# AutoML with specific metric for model selection
python main.py --ds google --model all --ms automl --time_budget 300 --metric RMSE
python main.py --ds air --model all --ms automl --time_budget 600 --metric ED
```

## Available Datasets

- `google`: Google stock prices
- `air`: Air quality data
- `appliances`: Energy consumption
- `eeg`: Brain wave measurements

## Available Models

- **Deep Learning**: TimeVAE, TimeGAN, FourierFlow, TimeTransformer
- **Statistical**: ARIMA, Exponential Smoothing

## Available Metrics (AutoML)

The following metrics can be specified for AutoML model selection using the `--metric` parameter:

- `RMSE`: Root Mean Square Error
- `MAE`: Mean Absolute Error
- `ED`: Euclidean Distance
- `DTW`: Dynamic Time Warping
- `MDD`: Marginal Distribution Difference
- `ACD`: Autocorrelation Difference
- `SD`: Skewness Difference
- `KD`: Kurtosis Difference
- `DS`: Discriminative Score
- `PS`: Predictive Score
- `C-FID`: Contextual-FID

**Note**: All comprehensive metrics are calculated for each model, but AutoML selects the best model based on the specified metric.

## Generated Data Characteristics

The pipeline generates synthetic time series data with the following characteristics:

- **Sequence Length**: The sequence length is automatically determined using autocorrelation analysis on the input data. The optimal length is bounded between 30-100 time steps (rounded to the nearest multiple of 5), with a default fallback of 125 if autocorrelation analysis fails. This ensures sequences capture meaningful temporal patterns in the data.

- **Number of Generated Time Series**: The number of generated time series samples equals the number of training sequences, which is 90% of the total sequences created from the input dataset (10% is reserved for testing). For the Air dataset with domain adaptation, the split is 90% training, 10% validation, with target domain data used for testing.

- **Data Shape**: Generated data has shape `(num_samples, sequence_length, num_features)`, where:
  - `num_samples`: Number of generated time series (equal to number of training sequences)
  - `sequence_length`: Determined by autocorrelation analysis (typically 30-100, default 125)
  - `num_features`: Number of features in the original dataset

## Output

- `results.json`: Evaluation metrics
- `visualizations/`: t-SNE and distribution plots
- `models/`: Saved trained models

## Examples

```bash
# Standard pipeline examples
python main.py --ds air --model arima
python main.py --ds eeg --model timevae timegan
python main.py --ds google --model all

# AutoML examples
python main.py --ds google --model all --ms automl --time_budget 600
python main.py --ds air --model timevae timegan --ms automl --time_budget 180
python main.py --ds eeg --model all --ms automl --time_budget 300 --metric DTW
python main.py --ds appliances --model all --ms automl --time_budget 600 --metric C-FID
```

## Configuration

Model parameters can be modified in `config/config.py` (epochs, hidden dimensions, learning rates, etc.).

## Extending the Codebase

### Adding a New Dataset

To add a new dataset, modify the following file:

1. **`src/data_loading.py`**: Add a new conditional branch in the `data_loading()` function to load your dataset. Return a `pd.DataFrame` with numerical columns only.
2. **`main.py`** (optional): Update the help text for the `--ds` argument to include your new dataset name.

For domain adaptation datasets (like "air"), return a tuple `(source_data, target_data)` and update `src/pipeline.py` and `src/data_preprocessing.py` if special handling is needed.

### Adding a New Model

To add a new model, modify the following files:

1. **`config/config.py`**:

   - Add model configuration to `DEEP_MODELS_CONFIG` (for deep learning models) or `TIMESERIES_MODELS_CONFIG` (for statistical models)
   - Register the model in `MODEL_SELECTION_CONFIG["available_models"]` and add to either `"deep_models"` or `"timeseries_models"` list

2. **`src/model_generation.py`**: Add a new conditional branch in the `_train_deep_model()` function to handle model initialization, training, data generation, and saving. Ensure the model returns generated data with shape `(num_samples, seq_len, num_features)`.

3. **`src/automl.py`** (optional, for AutoML support): Create an estimator class inheriting from `BaseEstimator` and register it in the `automl_model_selection()` function.

4. **`models/your_new_model.py`** (if needed): Create a separate implementation file for complex models and import it in `src/model_generation.py`.
