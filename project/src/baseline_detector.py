from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Documentation - What this script does

This script builds a simple baseline anomaly detector using the engineered
features from `market_data_features_2021_2026.csv`.

The detector is intentionally simple and interpretable.

Baseline idea
-------------
For each ticker / series, define:
- return anomaly component = abs(z_return_20)
- volume anomaly component = abs(volume_z_20)

Then combine them into one baseline anomaly score:
    baseline_score = return_weight * return_component
                   + volume_weight * volume_component

If volume data is missing or unusable, the score falls back to the return
component only.

Thresholding
------------
The script computes anomaly thresholds separately for each ticker using a
high quantile of the baseline score.

Example:
- 0.99 quantile = top 1% highest-score days are flagged as anomalies

Outputs
-------
1. A CSV file with the baseline score and anomaly flags:
   data/processed/baseline_anomaly_results.csv

2. A summary table by ticker:
   results/tables/baseline_anomaly_summary.csv

3. A price plot for each ticker:
   results/figures/baseline/

Plot style
----------
- Price series shown as a line
- Anomalies shown as red dots
- Vertical grey dashed lines at anomaly dates

Why this version avoids the DtypeWarning
----------------------------------------
The warning came from reading the full CSV, where some metadata columns
(e.g. Sector, Industry) may contain mixed types. This script avoids that by:
- reading only the columns it actually needs
- explicitly setting dtypes for those columns
"""


# ---------------------------
# Paths
# ---------------------------
INPUT_PATH = Path("data/processed/market_data_features_2021_2026.csv")
OUTPUT_DATA_PATH = Path("data/processed/baseline_anomaly_results.csv")

FIGURES_DIR = Path("results/figures/baseline")
TABLES_DIR = Path("results/tables")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = TABLES_DIR / "baseline_anomaly_summary.csv"


# ---------------------------
# Configuration
# ---------------------------
RETURN_WEIGHT = 0.7
VOLUME_WEIGHT = 0.3
ANOMALY_QUANTILE = 0.99


def safe_abs(series: pd.Series) -> pd.Series:
    """
    Absolute value with inf converted to NaN.
    """
    return series.abs().replace([np.inf, -np.inf], np.nan)


def build_baseline_score(group: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple interpretable anomaly score for one ticker.
    """
    group = group.sort_values("Date").copy()

    group["return_component"] = safe_abs(group["z_return_20"])
    group["volume_component"] = safe_abs(group["volume_z_20"])

    has_volume = group["volume_component"].notna()

    group["baseline_score"] = np.where(
        has_volume,
        RETURN_WEIGHT * group["return_component"] + VOLUME_WEIGHT * group["volume_component"],
        group["return_component"]
    )

    group["flag_return_only"] = group["return_component"] >= 3.0
    group["flag_volume_only"] = group["volume_component"] >= 3.0

    threshold = group["baseline_score"].quantile(ANOMALY_QUANTILE)
    group["baseline_threshold"] = threshold
    group["is_anomaly"] = group["baseline_score"] >= threshold

    return group


def summarize_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact per-ticker summary table.
    """
    summary = (
        df.groupby(["SeriesName", "Ticker"], as_index=False)
        .agg(
            n_rows=("Date", "count"),
            n_anomalies=("is_anomaly", "sum"),
            anomaly_rate=("is_anomaly", "mean"),
            avg_score=("baseline_score", "mean"),
            max_score=("baseline_score", "max"),
            threshold=("baseline_threshold", "max"),
        )
        .sort_values(["SeriesName", "Ticker"])
        .reset_index(drop=True)
    )

    return summary


def plot_anomalies(group: pd.DataFrame, outdir: Path) -> None:
    """
    Save a price plot with anomaly points highlighted.
    """
    group = group.sort_values("Date").copy()
    series_name = group["SeriesName"].iloc[0]
    ticker = group["Ticker"].iloc[0]

    anomalies = group[group["is_anomaly"]]

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

    plt.title(f"{series_name} ({ticker}) - Baseline Anomalies")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    filename = f"{ticker}_baseline_anomalies.png"
    plt.savefig(outdir / filename, dpi=150)
    plt.close()


def main() -> None:
    # Read only the columns needed for the baseline model.
    # This avoids mixed-type warnings from unrelated metadata columns.
    usecols = [
        "Date",
        "SeriesName",
        "Ticker",
        "Price",
        "z_return_20",
        "volume_z_20",
    ]

    dtype_map = {
        "SeriesName": "string",
        "Ticker": "string",
        "Price": "float64",
        "z_return_20": "float64",
        "volume_z_20": "float64",
    }

    df = pd.read_csv(
        INPUT_PATH,
        usecols=usecols,
        dtype=dtype_map,
        parse_dates=["Date"],
        low_memory=False,
    )

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    required_cols = ["Date", "SeriesName", "Ticker", "Price", "z_return_20", "volume_z_20"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    parts = []
    for ticker, group in df.groupby("Ticker", sort=False):
        parts.append(build_baseline_score(group))

    result = pd.concat(parts, ignore_index=True)

    result.to_csv(OUTPUT_DATA_PATH, index=False)

    summary = summarize_by_ticker(result)
    summary.to_csv(SUMMARY_PATH, index=False)

    for ticker, group in result.groupby("Ticker", sort=False):
        plot_anomalies(group, FIGURES_DIR)

    print(f"Saved detailed results to: {OUTPUT_DATA_PATH}")
    print(f"Saved summary table to: {SUMMARY_PATH}")
    print(f"Saved plots to: {FIGURES_DIR}")
    print()
    print(summary)


if __name__ == "__main__":
    main()


    