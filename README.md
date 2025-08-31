# Time Series Generation Pipeline

A comprehensive pipeline for generating synthetic time series data using various deep learning and statistical models.

## Installation

```bash
# Clone repo
git clone https://github.com/matejkutirov/AutoTSG.git

# Create virtual environment
python -m venv venv
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

## Output

- `results.json`: Evaluation metrics
- `visualizations/`: t-SNE and distribution plots
- `models/`: Saved trained models
- `data/gen/`: Generated synthetic data

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

Model parameters can be modified in `config.py` (epochs, hidden dimensions, learning rates, etc.).
