# Time-Series Anomaly Detection using Stock Market Data

MSc. in Data Science – Semester Project  
Course: **COMP-592DL – Project in Data Science**

## Project Overview

This repository contains my semester project on **time-series anomaly detection using stock market data**.

The project studies how different anomaly-detection methods behave on **financial time series**, using selected stocks together with benchmark market indices. The goal is to detect unusual market behavior, compare model outputs, and evaluate the trade-offs between simple, tree-based, and deep-learning approaches.

The project compares three methods:

- **Baseline statistical detector**
- **Isolation Forest**
- **LSTM Autoencoder**

---

## Research Objective

The main objective of this project is to investigate anomaly detection in **financial time series** and compare whether different methods identify similar or different types of unusual behavior.

More specifically, the project aims to:

- collect historical market data for selected stocks and indices
- engineer anomaly-relevant features from OHLCV data
- apply multiple anomaly-detection methods
- analyze anomaly dates and event peaks
- compare the methods in terms of sensitivity, interpretability, and behavior
- connect the implementation to relevant academic literature

---

## Selected Instruments

### Indices
- **S&P 500**
- **NASDAQ**
- **VIX**

### Stocks
A selected subset of stocks is used in the project and can be adjusted inside the scripts and notebooks.

---

## Methods

### 1. Baseline Detector
A simple and interpretable benchmark based on statistical anomaly logic and thresholding.

### 2. Isolation Forest
An unsupervised tree-based anomaly-detection model applied to engineered financial features.

### 3. LSTM Autoencoder
A sequence-based deep-learning model that detects anomalies through reconstruction error on rolling time windows.

---

## Feature Engineering

The project transforms raw **OHLCV** market data into anomaly-detection inputs.

The main feature groups are:

- **Return features**  
  simple returns, absolute returns, and multi-horizon returns

- **Trend features**  
  moving averages and distances from moving averages

- **Volatility features**  
  rolling volatility over different windows

- **Range / candle features**  
  high-low range, gaps, candle-body behavior

- **Volume features**  
  volume changes, relative volume, and rolling volume anomalies

- **Market context features**  
  SP500, NASDAQ, and VIX series used as contextual benchmarks

- **Relative features**  
  stock behavior relative to market benchmarks such as SP500 and NASDAQ

These features are used as inputs to the baseline detector, Isolation Forest, and LSTM autoencoder.

For implementation details, see:
- `src/features.py`
- `src/Feature_Engineering.docx`

---

## Project Pipeline

The project pipeline is:

1. **Problem definition**
2. **Data collection**
3. **Preprocessing and feature engineering**
4. **Baseline anomaly detection**
5. **Isolation Forest anomaly detection**
6. **LSTM autoencoder anomaly detection**
7. **Per-method notebook analysis**
8. **Cross-method comparison**
9. **Report writing and presentation**

---

## Repository Structure

```text
.
├── src/
│   ├── download_data.py
│   ├── features.py
│   ├── baseline_detector.py
│   ├── isolation_forest_detector.py
│   ├── lstm_autoencoder_detector.py
│   └── Feature_Engineering.docx
├── baseline_anomaly_analysis.ipynb
├── isolation_forest_analysis.ipynb
├── lstm_autoencoder_analysis.ipynb
├── method_comparison_analysis.ipynb
├── data.zip
├── results.zip
├── README.md
└── LICENSE