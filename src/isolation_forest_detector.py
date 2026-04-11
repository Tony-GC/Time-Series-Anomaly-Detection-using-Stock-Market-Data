

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest


"""
Documentation - What this script does

This script applies Isolation Forest to the engineered financial features.

Model design
------------
- One Isolation Forest model is trained per ticker / series.
- The model is trained on an earlier time window only.
- It then scores the full series.

Why this is useful
------------------
Compared with the baseline detector, Isolation Forest can detect unusual
combinations of features rather than only large return or volume z-scores.

Feature groups used
-------------------
This first version uses a practical subset of the engineered features:

- log_return
- abs_log_return
- return_5d
- return_20d
- hl_range_pct
- dist_ma_20
- vol_5
- vol_20
- z_return_20
- volume_z_20
- SP500_log_return
- NASDAQ_log_return
- VIX_level
- VIX_log_return
- excess_return_sp500

Missing values
--------------
Isolation Forest cannot handle NaN values directly.
This script fills missing values using the median of the training split
for each ticker and feature.

Training split
--------------
Train period:
- 2021-01-01 to 2024-12-31

The model is then used to score all rows for that ticker.

Outputs
-------
1. Detailed scored dataset:
   data/processed/isolation_forest_results.csv

2. Summary table:
   results/tables/isolation_forest_summary.csv

3. Figures:
   results/figures/isolation_forest/

Plot style
----------
- price line
- anomalies shown as red dots
- vertical grey dashed lines at anomaly dates
"""


# ---------------------------
# Paths
# ---------------------------
INPUT_PATH = Path("data/processed/market_data_features_2021_2026.csv")
OUTPUT_DATA_PATH = Path("data/processed/isolation_forest_results.csv")

FIGURES_DIR = Path("results/figures/isolation_forest")
TABLES_DIR = Path("results/tables")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = TABLES_DIR / "isolation_forest_summary.csv"


# ---------------------------
# Configuration
# ---------------------------
TRAIN_END_DATE = pd.Timestamp("2024-12-31")

CONTAMINATION = 0.01
N_ESTIMATORS = 300
RANDOM_STATE = 42

CANDIDATE_FEATURES = [
    "log_return",
    "abs_log_return",
    "return_5d",
    "return_20d",
    "hl_range_pct",
    "dist_ma_20",
    "vol_5",
    "vol_20",
    "z_return_20",
    "volume_z_20",
    "SP500_log_return",
    "NASDAQ_log_return",
    "VIX_level",
    "VIX_log_return",
    "excess_return_sp500",
]


def prepare_features(group: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Prepare train and full feature matrices for one ticker.
    Missing values are filled using train medians only.
    """
    group = group.sort_values("Date").copy()

    available_features = [c for c in feature_cols if c in group.columns]
    if not available_features:
        raise ValueError("No requested features are available in the input dataframe.")

    train_mask = group["Date"] <= TRAIN_END_DATE
    train_df = group.loc[train_mask].copy()

    if train_df.empty:
        raise ValueError("Training split is empty for this ticker.")

    X_train = train_df[available_features].replace([np.inf, -np.inf], np.nan).copy()
    X_full = group[available_features].replace([np.inf, -np.inf], np.nan).copy()

    # median fill from training data only
    train_medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(train_medians)
    X_full = X_full.fillna(train_medians)

    # if some columns are still all-NaN after median fill, drop them
    valid_cols = [c for c in available_features if X_train[c].notna().any() and X_full[c].notna().any()]
    X_train = X_train[valid_cols]
    X_full = X_full[valid_cols]

    if X_train.shape[1] == 0:
        raise ValueError("No valid features remain after NaN handling.")

    return X_train, X_full, valid_cols


def fit_and_score_group(group: pd.DataFrame) -> pd.DataFrame:
    """
    Fit one Isolation Forest per ticker and score the full series.
    """
    group = group.sort_values("Date").copy()

    X_train, X_full, used_features = prepare_features(group, CANDIDATE_FEATURES)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train)

    # decision_function: higher = more normal
    # We invert it so higher = more anomalous
    decision = model.decision_function(X_full)
    pred = model.predict(X_full)   # -1 anomaly, 1 normal

    group["iforest_score_raw"] = decision
    group["iforest_score"] = -decision
    group["iforest_is_anomaly"] = pred == -1
    group["iforest_used_n_features"] = len(used_features)
    group["iforest_features_used"] = ", ".join(used_features)

    # Useful for transparency
    group["iforest_train_end_date"] = TRAIN_END_DATE
    group["iforest_contamination"] = CONTAMINATION

    return group


def summarize_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact per-ticker summary table.
    """
    summary = (
        df.groupby(["SeriesName", "Ticker"], as_index=False)
        .agg(
            n_rows=("Date", "count"),
            n_anomalies=("iforest_is_anomaly", "sum"),
            anomaly_rate=("iforest_is_anomaly", "mean"),
            avg_score=("iforest_score", "mean"),
            max_score=("iforest_score", "max"),
            n_features=("iforest_used_n_features", "max"),
        )
        .sort_values(["SeriesName", "Ticker"])
        .reset_index(drop=True)
    )
    return summary


def plot_anomalies(group: pd.DataFrame, outdir: Path) -> None:
    """
    Save a price plot with Isolation Forest anomaly points highlighted.
    """
    group = group.sort_values("Date").copy()
    series_name = group["SeriesName"].iloc[0]
    ticker = group["Ticker"].iloc[0]

    anomalies = group[group["iforest_is_anomaly"]]

    plt.figure(figsize=(12, 5))
    plt.plot(group["Date"], group["Price"], label="Price", linewidth=1.5)

    # Vertical grey lines at anomaly dates
    for anomaly_date in anomalies["Date"]:
        plt.axvline(
            x=anomaly_date,
            color="grey",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            zorder=1
        )

    # Anomalies as red dots
    plt.scatter(
        anomalies["Date"],
        anomalies["Price"],
        color="red",
        label="Anomaly",
        s=28,
        zorder=3
    )

    plt.title(f"{series_name} ({ticker}) - Isolation Forest Anomalies", pad=20)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    filename = f"{ticker}_iforest_anomalies.png"
    plt.savefig(outdir / filename, dpi=150)
    plt.close()


def main() -> None:
    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"], low_memory=False)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    required_cols = ["Date", "SeriesName", "Ticker", "Price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    parts = []
    for ticker, group in df.groupby("Ticker", sort=False):
        result = fit_and_score_group(group)
        parts.append(result)

    result_df = pd.concat(parts, ignore_index=True)

    # Save detailed results
    result_df.to_csv(OUTPUT_DATA_PATH, index=False)

    # Save summary table
    summary = summarize_by_ticker(result_df)
    summary.to_csv(SUMMARY_PATH, index=False)

    # Save one plot per ticker
    for ticker, group in result_df.groupby("Ticker", sort=False):
        plot_anomalies(group, FIGURES_DIR)

    print(f"Saved detailed results to: {OUTPUT_DATA_PATH}")
    print(f"Saved summary table to: {SUMMARY_PATH}")
    print(f"Saved plots to: {FIGURES_DIR}")
    print()
    print(summary)


if __name__ == "__main__":
    main()