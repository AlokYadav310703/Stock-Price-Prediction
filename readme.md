# Nifty 50 Stock Price Prediction

A deep learning pipeline that forecasts **Nifty 50 (^NSEI) closing prices** using an ensemble of LSTM, CNN, and TCN models, with **Coiflet wavelet denoising** as a preprocessing step. The models are compared on both raw and denoised data to quantify the benefit of noise reduction.

---

## Overview

The notebook:
1. Downloads 13+ years of Nifty 50 OHLC data (2010–2023) via `yfinance`
2. Denoises the price series using Coiflet-3 wavelet thresholding
3. Engineers sliding-window sequences (20-day lookback)
4. Trains three deep learning models — **LSTM**, **CNN**, and **TCN** — on both raw and denoised data
5. Combines them in a **stacking ensemble** (Linear Regression meta-learner)
6. Evaluates and visualizes performance across all models

---

## Dataset

| Field | Details |
|---|---|
| Source | Yahoo Finance via `yfinance` |
| Ticker | `^NSEI` (Nifty 50 Index) |
| Period | 2010-01-01 to 2023-12-31 |
| Features | Open, High, Low, Close |
| Target | Next-day Close price |

Raw data is saved to Google Drive at `Stock_Prediction_Project/nifty50_raw.csv`.

---

## Pipeline

### 1. Wavelet Denoising (Coif3)
The `wavelet_denoise_series` function applies:
- **Wavelet:** Coiflet-3 (`coif3`), level-1 decomposition via `PyWavelets`
- **Thresholding:** Universal soft threshold — `σ × √(2 log N)`, where σ is estimated via the median absolute deviation of detail coefficients
- Applied independently to each of the four OHLC columns

The denoised data is saved to `nifty50_denoised.csv` and visualized against the raw series.

### 2. Scaling & Windowing
- **Scalers:** Separate `MinMaxScaler` instances for features and target (fit on denoised and raw independently)
- **Window size:** 20 trading days (lookback)
- **Target:** Next-day normalized Close price
- Input shape per sample: `(20, 4)`

### 3. Train / Validation / Test Split
Chronological (no shuffling to prevent data leakage):

| Split | Proportion |
|---|---|
| Train | 60% |
| Validation | 15% |
| Test | 25% |

### 4. Models

#### LSTM
```
Input(20, 4)
→ LSTM(64, tanh, return_sequences=True) → Dropout(0.1)
→ LSTM(64, tanh) → Dropout(0.1)
→ Dense(32, tanh) → Dense(1)
```
- Optimizer: Adam (lr=0.0001) | Loss: MSE

#### CNN
```
Input(20, 4)
→ Conv1D(64, k=2, causal) × 2
→ MaxPooling1D(2) → Flatten
→ Dense(64, relu) → Dense(1)
```
- Optimizer: Adam (lr=0.001) | Loss: MSE

#### TCN (Temporal Convolutional Network)
```
Input(20, 4)
→ TCN Block (dilation=1) → TCN Block (dilation=2) → TCN Block (dilation=4)
→ GlobalAveragePooling1D → Dense(64, relu) → Dense(1)
```
Each TCN block uses causal dilated convolutions with residual connections.  
- Optimizer: Adam (lr=0.0001) | Loss: MSE

All three models use `EarlyStopping` (patience=10, restores best weights).

### 5. Stacking Ensemble
The ensemble meta-learner is a **Linear Regression** model trained on the concatenated test-set predictions of all three base models:

```
[LSTM(raw) | CNN(denoised) | TCN(denoised)] → LinearRegression → Final prediction
```

Ensemble weights (coefficients) for each base model are printed after training.

---

## Evaluation

All models are evaluated on the held-out test set with:

| Metric | Description |
|---|---|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of Determination |

Bar charts compare all four models (LSTM, CNN, TCN, Ensemble) across every metric. An **Actual vs Predicted** line plot is also generated for the ensemble over the first 200 test samples.

---

## Outputs

All outputs are saved to `/content/drive/MyDrive/Stock_Prediction_Project/`:

| File | Description |
|---|---|
| `nifty50_raw.csv` | Raw downloaded OHLC data |
| `nifty50_denoised.csv` | Wavelet-denoised OHLC data |
| `predictions.csv` | Test-set predictions from all models |
| `lstm_model.keras` | Trained LSTM model |
| `cnn_model.keras` | Trained CNN model |
| `tcn_model.keras` | Trained TCN model |
| `feature_scaler.pkl` | Fitted MinMaxScaler for features |
| `target_scaler.pkl` | Fitted MinMaxScaler for target |

---

## Requirements

```bash
pip install yfinance PyWavelets tensorflow scikit-learn matplotlib pandas joblib
```

This notebook is designed to run on **Google Colab** with Google Drive mounted.

---

## Usage

1. Open the notebook in Google Colab
2. Mount your Google Drive when prompted
3. Run all cells sequentially
4. Results and saved models appear in `MyDrive/Stock_Prediction_Project/`

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Lookback window | 20 days |
| Train / Val / Test split | 60% / 15% / 25% |
| Wavelet | Coiflet-3 (level 1, soft threshold) |
| LSTM units | 64 × 2 layers |
| CNN filters | 64, kernel=2, causal padding |
| TCN dilations | 1, 2, 4 |
| Batch size (LSTM/CNN) | 32 |
| Batch size (TCN) | 64 |
| Max epochs | 100 (with early stopping) |
| Early stopping patience | 10 |

---

## References

- Daubechies, I. — *Ten Lectures on Wavelets* (SIAM, 1992)
- Bai, S. et al. — *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* (2018)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [PyWavelets Documentation](https://pywavelets.readthedocs.io/)
