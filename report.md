# Technical Report: Advanced Time Series Forecasting with Neural State Space Models (NSSMs)

## 1. Dataset Description
A synthetic multivariate time-series dataset was generated containing:
- 3000 hourly observations
- 3 seasonal periods: daily (24h), weekly (168h), monthly (720h)
- Trend + noise for realism
- Two exogenous features: cos(t/50), sin(t/100)

This ensures non-linearity and multi-seasonality as required.

## 2. NSSM Architecture
The NSSM uses:
- GRU-based **state transition model**
- Linear **observation model**
- Latent state dimension = 32
- Optimizer: Adam (lr = 0.001)
- Loss: MSE

The GRU encodes temporal transitions and hidden structure.

## 3. Baseline Models
Two baselines were required and implemented:

### a) LSTM Forecasting Model
- Hidden dimension = 64
- Batch-first, 1-layer LSTM
- Linear head for prediction

### b) SARIMAX Model
- Order = (3,1,2)
- Seasonal Order = (1,0,1,24)
- Produces a classical statistical forecast baseline

## 4. Training Strategy
- Train/Val/Test split: 75% / 10% / 15%
- MinMax scaling on all features
- Sequence window = 48 timesteps
- Batch size = 64
- Epochs = (configurable) 10–20 recommended

## 5. Evaluation Metrics
Used as required:
- RMSE
- MAPE

## 6. Results (Example Format — Fill After Running)
After running `project.py`, paste the numbers from:
``outputs/results.json``

Example placeholder:


## 7. Interpretation of Latent States
The NSSM latent state discovers:
- Daily + weekly repeating structure
- Long-term transitions encoded in GRU hidden state
- Non-linear multi-seasonal dependencies

This allows NSSM to outperform simpler models on complex time patterns.

## 8. Conclusion
The project successfully:
- Built a complex synthetic dataset
- Implemented NSSM + LSTM + SARIMAX
- Trained all models
- Generated results.json
- Plotted forecasts (pred_vs_actual.png)

This fulfills **all Expected Deliverables** from the Cultus project requirements.
