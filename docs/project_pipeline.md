# Project Pipeline

This document summarizes the end-to-end workflow of the project **Time-Series Anomaly Detection using Stock Market Data**.

## 1. Problem Definition

The project investigates anomaly detection in **financial time series** using stock market data. The objective is to detect unusual market behavior and compare the behavior of three method families:

- baseline statistical detector
- Isolation Forest
- LSTM autoencoder

The project focuses on selected stocks together with benchmark indices such as:

- S&P 500
- NASDAQ
- VIX

## 2. Data Collection

Historical market data is collected using the data download script.

**Script**
- `src/download_data.py`

**Data collected**
- OHLCV data
- benchmark indices
- selected stocks
- metadata such as sector, industry, exchange, and currency

**Output**
- raw market dataset in the data pipeline

## 3. Preprocessing and Feature Engineering

The raw market data is transformed into anomaly-detection inputs.

**Script**
- `src/features.py`

**Feature groups**
- return features
- multi-horizon return features
- moving averages and trend distance
- volatility features
- range and candle features
- volume anomaly features
- market context features
- relative performance features

**Output**
- processed feature dataset used by all downstream models

## 4. Baseline Anomaly Detection

A simple statistical benchmark is applied first.

**Script**
- `src/baseline_detector.py`

**Purpose**
- provide an interpretable reference model
- establish a benchmark for comparison against more advanced methods

**Outputs**
- anomaly scores
- anomaly flags
- summary tables
- figures for notebook analysis

## 5. Isolation Forest Detection

A tree-based unsupervised anomaly detector is applied to the engineered feature space.

**Script**
- `src/isolation_forest_detector.py`

**Purpose**
- detect multivariate anomalies without deep learning
- compare against the baseline and sequence-based methods

**Outputs**
- Isolation Forest scores
- anomaly flags
- summary tables
- figures for notebook analysis

## 6. LSTM Autoencoder Detection

A sequence-based deep-learning model is applied to the engineered features.

**Script**
- `src/lstm_autoencoder_detector.py`

**Purpose**
- detect anomalies from unusual temporal sequences
- evaluate whether a sequence model captures different anomaly behavior from the other two methods

**Outputs**
- LSTM anomaly scores
- anomaly flags
- event peaks
- summary tables
- figures for notebook analysis

## 7. Per-Method Notebook Analysis

Each method is analyzed in its own notebook.

**Notebooks**
- `notebooks/baseline_anomaly_analysis.ipynb`
- `notebooks/isolation_forest_analysis.ipynb`
- `notebooks/lstm_autoencoder_analysis.ipynb`

**Purpose**
- inspect anomaly dates
- inspect score behavior
- visualize price and volume
- interpret event peaks
- understand method-specific strengths and weaknesses

## 8. Cross-Method Comparison

The final notebook compares all three methods.

**Notebook**
- `notebooks/method_comparison_analysis.ipynb`

**Purpose**
- compare anomaly counts
- compare event peak counts
- examine overlap and disagreement
- analyze differences in sensitivity and interpretability

## 9. Report and Presentation

The final outputs of the project are:

- code implementation in Python
- notebooks for method analysis
- final report / PDF
- references to supporting literature
- presentation slides

## 10. Pipeline Summary

The overall workflow is:

1. define the problem
2. collect market data
3. engineer features
4. run baseline detector
5. run Isolation Forest
6. run LSTM autoencoder
7. analyze each method in notebooks
8. compare methods
9. write the report and prepare the presentation
