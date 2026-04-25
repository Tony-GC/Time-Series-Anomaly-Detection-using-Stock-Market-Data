# Time-Series Anomaly Detection using Stock Market Data

MSc. in Data Science – Semester Project  
Course: **COMP-592DL – Project in Data Science**

## Project Overview

This project studies **anomaly detection in financial time series** using stock market data.  
The goal is to identify unusual market behavior in stock prices and related market indicators, and to compare three different anomaly-detection approaches:

- a **baseline statistical detector**
- **Isolation Forest**
- an **LSTM autoencoder**

The project is built around publicly available historical market data and focuses on both **interpretability** and **method comparison**.

---

## Research Objective

The main objective is to investigate how different anomaly-detection methods behave on **financial time series**, and whether they capture similar or different types of unusual market behavior.

In particular, the project aims to:

- engineer anomaly-relevant features from OHLCV market data
- detect unusual behavior in selected stocks and benchmark indices
- compare simple, tree-based, and deep-learning methods
- analyze anomaly dates, anomaly clusters, and event peaks
- connect the implementation with relevant academic literature

---

## Selected Instruments

### Indices
- S&P 500
- NASDAQ
- VIX

### Stocks
The project uses a selected subset of stocks for analysis.  
The exact stock list can be adjusted inside the scripts and notebooks.

---

## Methods

### 1. Baseline Detector
A simple and interpretable benchmark based on statistical anomaly logic and thresholding.

### 2. Isolation Forest
An unsupervised tree-based anomaly-detection model applied to engineered financial features.

### 3. LSTM Autoencoder
A sequence-based deep-learning model that detects anomalies using reconstruction error on rolling time windows.

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