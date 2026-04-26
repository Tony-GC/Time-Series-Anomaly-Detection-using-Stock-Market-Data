# Methodology

This document summarizes the methodology used in the project **Time-Series Anomaly Detection using Stock Market Data**.

## 1. Research Setting

The project studies anomaly detection in financial time series using selected stocks and benchmark indices.

The methods were chosen to represent three different levels of complexity:

- **Baseline detector**: simple and interpretable benchmark
- **Isolation Forest**: unsupervised machine-learning detector
- **LSTM autoencoder**: sequence-based deep-learning detector

This structure makes it possible to compare simple, classical, and deep approaches in the same experimental setting.

## 2. Data

The project uses historical market data with daily frequency. The dataset includes:

- open
- high
- low
- close
- adjusted close
- volume

In addition to selected stocks, benchmark market context is included through:

- S&P 500
- NASDAQ
- VIX

## 3. Feature Engineering

The project does not rely directly on raw prices alone. Instead, the models are trained on engineered features that capture abnormal market behavior more clearly.

The main feature categories are:

### 3.1 Return Features
Examples:
- simple returns
- absolute returns
- 5-day and 10-day returns

### 3.2 Trend Features
Examples:
- moving averages
- distance from moving averages

### 3.3 Volatility Features
Examples:
- rolling volatility
- volatility ratios

### 3.4 Range and Candle Features
Examples:
- high-low range
- candle body
- price gaps

### 3.5 Volume Features
Examples:
- relative volume
- volume z-scores

### 3.6 Market Context Features
Examples:
- SP500 simple return
- NASDAQ simple return
- VIX simple return
- VIX z-score

### 3.7 Relative Features
Examples:
- excess return relative to SP500
- excess return relative to NASDAQ

## 4. Methods

## 4.1 Baseline Detector

The baseline method is an interpretable benchmark. It is based on statistical anomaly logic and thresholding.

**Purpose**
- provide a transparent reference
- establish a benchmark for comparison

**Advantages**
- simple
- easy to explain
- useful for sanity checking the data and features

**Limitations**
- limited ability to capture complex multivariate structure
- less flexible than machine-learning methods

## 4.2 Isolation Forest

Isolation Forest is an unsupervised anomaly-detection method based on random partitioning.

**Purpose**
- detect anomalies in multivariate feature space
- provide a stronger benchmark than simple thresholding

**Advantages**
- works well in high-dimensional settings
- unsupervised
- relatively efficient

**Limitations**
- less directly interpretable than the baseline
- may still treat certain financial regime changes as anomalies even if they are not rare in economic terms

## 4.3 LSTM Autoencoder

The LSTM autoencoder is a sequence-based deep-learning method.

**Purpose**
- detect anomalies through reconstruction error on rolling temporal windows
- model sequential structure rather than isolated points only

**Advantages**
- sequence-aware
- capable of detecting regime-like anomalies
- useful for time-dependent patterns

**Limitations**
- more computationally expensive
- harder to tune
- less interpretable than the baseline

## 5. Event Peaks

To reduce clutter in anomaly outputs, nearby anomaly dates are grouped into **event peaks**.

An event peak is defined as the highest-scoring anomaly inside a local cluster of nearby anomaly dates.

This makes the results easier to interpret, especially in the LSTM and Isolation Forest workflows.

## 6. Analysis Strategy

The project analyzes results at two levels:

### 6.1 Per-method analysis
Each method has its own notebook for:
- anomaly plots
- score plots
- event peak review
- selected feature plots

### 6.2 Cross-method comparison
A final notebook compares:
- anomaly counts
- event peak counts
- score behavior
- overlap and disagreement across methods

## 7. Evaluation Logic

The project does not assume the existence of perfect ground-truth anomaly labels. Instead, it evaluates the methods by:

- interpretability of detected episodes
- consistency with visible market events
- relative sensitivity across instruments
- overlap and disagreement across methods

This is appropriate for a financial anomaly-detection setting where anomaly labels are often ambiguous.

## 8. Final Deliverable Logic

The final methodology integrates:

- reproducible scripts
- engineered financial features
- three anomaly-detection methods
- per-method notebook analysis
- comparison notebook
- academic discussion linked to literature
