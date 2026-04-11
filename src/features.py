from pathlib import Path
import numpy as np
import pandas as pd


"""
Documentation - What this script does

This script creates anomaly-detection features from the raw market dataset.

It builds features in three layers:

1. Per-series behavior
Computed separately for each index or stock:
 - returns
 - rolling volatility
 - moving averages
 - rolling z-scores
 - range features
 - volume anomalies

2. Market context
It brings in, for every date:
 - S&P 500 level and return
 - NASDAQ level and return
 - VIX level and return

So every stock-day can be interpreted relative to what the market was doing.

3. Relative behavior
For stocks only:
 - excess_return_sp500
 - excess_return_nasdaq

These are very useful because a stock can be anomalous even if the whole
market is moving, and these features help isolate that.

Important notes

1. VIX volume
VIX often does not behave like a stock and may have missing or less useful
volume information. That is fine.

2. Missing values at the beginning
The first few rows per ticker will have NaN values because:
 - returns need previous prices
 - 20-day and 60-day features need enough history

Do not fill those yet. They are expected.

3. Newer stocks
Some tickers may not have data for the entire period. That is also normal.
The feature code will still work.

Initial feature subset for modeling

[
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

That is a strong first feature set for Isolation Forest later.
"""


# ---------------------------
# Paths
# ---------------------------
RAW_PATH = Path("data/raw/market_data_2021_2026_with_metadata.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "market_data_features_2021_2026.csv"

MARKET_SERIES = {"SP500", "NASDAQ", "VIX"}


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score with stable handling of zero standard deviation.
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)
    rolling_std = rolling_std.replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


def choose_price_column(df: pd.DataFrame) -> str:
    """
    Prefer adjusted close if available, otherwise close.
    """
    if "Adj Close" in df.columns:
        return "Adj Close"
    return "Close"


def compute_single_series_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features independently for one ticker / series.
    """
    group = group.sort_values("Date").copy()

    price_col = choose_price_column(group)
    group["Price"] = group[price_col].astype(float)

    # Basic returns
    group["simple_return"] = group["Price"].pct_change()
    group["log_return"] = np.log(group["Price"]).diff()
    group["abs_log_return"] = group["log_return"].abs()

    # Multi-horizon returns
    group["return_5d"] = group["Price"].pct_change(5)
    group["return_20d"] = group["Price"].pct_change(20)

    # Range / candlestick style features
    group["hl_range_pct"] = (group["High"] - group["Low"]) / group["Price"]
    group["oc_change_pct"] = (group["Close"] - group["Open"]) / group["Open"]

    # Moving averages
    group["ma_5"] = group["Price"].rolling(5, min_periods=5).mean()
    group["ma_20"] = group["Price"].rolling(20, min_periods=20).mean()
    group["ma_60"] = group["Price"].rolling(60, min_periods=60).mean()

    group["dist_ma_20"] = (group["Price"] - group["ma_20"]) / group["ma_20"]
    group["dist_ma_60"] = (group["Price"] - group["ma_60"]) / group["ma_60"]

    # Rolling volatility based on log returns
    group["vol_5"] = group["log_return"].rolling(5, min_periods=5).std(ddof=0)
    group["vol_20"] = group["log_return"].rolling(20, min_periods=20).std(ddof=0)
    group["vol_60"] = group["log_return"].rolling(60, min_periods=60).std(ddof=0)

    # Return anomaly score
    group["z_return_20"] = rolling_zscore(group["log_return"], 20)

    # Volume features
    if "Volume" in group.columns:
        group["log_volume"] = np.log1p(group["Volume"].astype(float))
        group["volume_change"] = group["Volume"].pct_change()
        group["volume_z_20"] = rolling_zscore(group["log_volume"], 20)
    else:
        group["log_volume"] = np.nan
        group["volume_change"] = np.nan
        group["volume_z_20"] = np.nan

    return group


def build_market_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract daily market context from SP500, NASDAQ, and VIX
    and merge it back into the whole dataset.
    """
    context_cols = ["Date", "SeriesName", "Price", "log_return", "vol_20", "z_return_20"]

    market_df = df[df["SeriesName"].isin(MARKET_SERIES)][context_cols].copy()

    # Pivot to wide format
    wide = market_df.pivot(index="Date", columns="SeriesName")

    # Flatten MultiIndex columns
    wide.columns = [f"{series}_{feature}" for feature, series in wide.columns]
    wide = wide.reset_index()

    # Rename for readability
    rename_map = {
        "SP500_Price": "SP500_level",
        "SP500_log_return": "SP500_log_return",
        "SP500_vol_20": "SP500_vol_20",
        "SP500_z_return_20": "SP500_z_20",
        "NASDAQ_Price": "NASDAQ_level",
        "NASDAQ_log_return": "NASDAQ_log_return",
        "NASDAQ_vol_20": "NASDAQ_vol_20",
        "NASDAQ_z_return_20": "NASDAQ_z_20",
        "VIX_Price": "VIX_level",
        "VIX_log_return": "VIX_log_return",
        "VIX_vol_20": "VIX_vol_20",
        "VIX_z_return_20": "VIX_z_20",
    }
    wide = wide.rename(columns=rename_map)

    df = df.merge(wide, on="Date", how="left")

    # Excess returns only make sense for stocks, not for the market series themselves
    stock_mask = ~df["SeriesName"].isin(MARKET_SERIES)

    df["excess_return_sp500"] = np.nan
    df["excess_return_nasdaq"] = np.nan

    df.loc[stock_mask, "excess_return_sp500"] = (
        df.loc[stock_mask, "log_return"] - df.loc[stock_mask, "SP500_log_return"]
    )
    df.loc[stock_mask, "excess_return_nasdaq"] = (
        df.loc[stock_mask, "log_return"] - df.loc[stock_mask, "NASDAQ_log_return"]
    )

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns so that identifiers, raw data, and engineered features
    appear in a clean and predictable order.
    """
    base_cols = [
        "Date",
        "SeriesName",
        "Ticker",
        "AssetType",
        "Sector",
        "Industry",
        "Exchange",
        "Currency",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Price",
    ]

    feature_cols = [
        "simple_return",
        "log_return",
        "abs_log_return",
        "return_5d",
        "return_20d",
        "hl_range_pct",
        "oc_change_pct",
        "ma_5",
        "ma_20",
        "ma_60",
        "dist_ma_20",
        "dist_ma_60",
        "vol_5",
        "vol_20",
        "vol_60",
        "z_return_20",
        "log_volume",
        "volume_change",
        "volume_z_20",
        "SP500_level",
        "SP500_log_return",
        "SP500_vol_20",
        "SP500_z_20",
        "NASDAQ_level",
        "NASDAQ_log_return",
        "NASDAQ_vol_20",
        "NASDAQ_z_20",
        "VIX_level",
        "VIX_log_return",
        "VIX_vol_20",
        "VIX_z_20",
        "excess_return_sp500",
        "excess_return_nasdaq",
    ]

    ordered = [c for c in base_cols + feature_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def main() -> None:
    df = pd.read_csv(RAW_PATH, parse_dates=["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Compute features per ticker without using groupby.apply,
    # which avoids the pandas deprecation warning and future behavior changes.
    featured_parts = []

    for ticker, group in df.groupby("Ticker", sort=False):
        result = compute_single_series_features(group.copy())
        featured_parts.append(result)

    featured = pd.concat(featured_parts, ignore_index=True)

    # Add market context
    featured = build_market_context(featured)

    # Final ordering
    featured = reorder_columns(featured)

    # IMPORTANT - Keeping the period of interest 
    featured = featured[(featured["Date"] >= "2021-01-01")].copy()

    # Save
    featured.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved features to: {OUTPUT_PATH}")
    print(f"Shape: {featured.shape}")
    print(featured.head(10))


if __name__ == "__main__":
    main()