# Results Summary

This document provides a concise summary of the project outputs and the role of each method.

## 1. Project Output Types

The project produces several categories of outputs:

- processed feature datasets
- detector result files
- summary tables
- analysis notebooks
- cross-method comparison notebook
- final report and presentation materials

## 2. Baseline Detector

The baseline detector acts as the most interpretable benchmark.

### Main role
- identify obvious abnormal behavior
- provide a statistical reference
- offer a transparent comparison point for the more advanced methods

### Typical behavior
- tends to highlight large individual abnormal points
- useful for clear visual sanity checking
- easiest to explain in the final report

## 3. Isolation Forest

Isolation Forest provides the multivariate machine-learning benchmark.

### Main role
- detect anomalies from the engineered feature space
- identify abnormal combinations of behavior, not just single-threshold events

### Typical behavior
- more flexible than the baseline
- often balances sensitivity and interpretability reasonably well
- useful as the central non-deep-learning comparison method

## 4. LSTM Autoencoder

The LSTM autoencoder represents the sequence-based deep-learning approach.

### Main role
- detect anomalies through unusual temporal sequences
- capture regime-like behavior rather than isolated points only

### Typical behavior
- more sensitive to sequence structure
- can produce anomaly clusters and event peaks
- often less intuitive visually, but valuable for comparison

## 5. Event Peaks

Event peaks are used to summarize nearby anomaly detections into a single representative event.

### Why they are useful
- reduce clutter in visualizations
- simplify the interpretation of anomaly clusters
- make comparison across methods easier

## 6. Comparison Logic

The final comparison notebook allows the project to compare the methods through:

- number of anomalies detected
- number of event peaks detected
- overlap and disagreement across methods
- visual comparison on the same assets

## 7. Expected High-Level Findings

The exact results depend on the selected assets and parameter settings, but the analysis is designed to highlight the following themes:

- the baseline method is the most interpretable
- Isolation Forest provides stronger multivariate detection than the baseline
- the LSTM autoencoder captures sequence behavior but is harder to tune and interpret
- different methods may identify different kinds of anomaly episodes

## 8. How to Use This Summary

This file is intended as a lightweight project overview.

For detailed results, refer to:
- the per-method notebooks in `notebooks/`
- the comparison notebook
- the final report / PDF
