from pathlib import Path
import pandas as pd
import yfinance as yf


"""
This script already solves an important part of the project:
 - collects the S&P 500
 - collects the NASDAQ
 - collects our 12 stocks
 - stores them in one clean dataset
 - gives us a reproducible starting point for preprocessing and anomaly detection

What comes immediately after this

Once this works, the next implementation step should be to create preprocessing and feature engineering
for:
 - log returns
 - absolute returns
 - rolling mean
 - rolling volatility
 - rolling z-score of returns
 - rolling z-score of volume

That will be the real input to your anomaly detectors.
"""

START_DATE = "2020-09-01"
END_DATE = "2026-03-31"

INDEX_TICKERS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
}

STOCK_TICKERS = {
    "META": "META",
    "AAPL": "AAPL",
    "GOOGL": "GOOGL",
    "TSLA": "TSLA",
    "MSTR": "MSTR",
    "COIN": "COIN",
    "NVDA": "NVDA",
    "AMD": "AMD",
    "AMZN": "AMZN",
    "ORCL": "ORCL",
    "PLTR": "PLTR",
    "HOOD": "HOOD",
}

ALL_TICKERS = {**INDEX_TICKERS, **STOCK_TICKERS}

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def get_ticker_metadata(symbol: str) -> dict:
    """
    Fetch static metadata for one ticker from yfinance.
    """
    try:
        info = yf.Ticker(symbol).info

        return {
            "AssetType": info.get("quoteType"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Exchange": info.get("exchange"),
            "Currency": info.get("currency"),
        }
    except Exception as e:
        print(f"Warning: metadata fetch failed for {symbol}: {e}")
        return {
            "AssetType": None,
            "Sector": None,
            "Industry": None,
            "Exchange": None,
            "Currency": None,
        }


def download_market_data(tickers: dict[str, str], start: str, end: str) -> pd.DataFrame:
    yahoo_symbols = list(tickers.values())

    df = yf.download(
        tickers=yahoo_symbols,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if df.empty:
        raise ValueError("No data was downloaded. Check ticker symbols or date range.")

    frames = []

    for series_name, symbol in tickers.items():
        if symbol not in df.columns.get_level_values(0):
            print(f"Warning: {symbol} not found in downloaded data.")
            continue

        sub = df[symbol].copy()
        sub.reset_index(inplace=True)

        sub["SeriesName"] = series_name
        sub["Ticker"] = symbol

        metadata = get_ticker_metadata(symbol)
        for key, value in metadata.items():
            sub[key] = value

        frames.append(sub)

    if not frames:
        raise ValueError("No valid ticker data was processed.")

    out = pd.concat(frames, ignore_index=True)

    final_cols = [
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
    ]
    existing_cols = [c for c in final_cols if c in out.columns]
    out = out[existing_cols]

    out = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return out


data = download_market_data(ALL_TICKERS, START_DATE, END_DATE)
#data = data[data["Date"].between("2021-01-01", "2021-12-31")]
# data = data[(data["Date"] >= "2021-01-01") & (data["Date"] <= "2021-12-31")]
data = data[(data["Date"] >= "2021-01-01")]
print(f"Shape: {data.shape}")
print(data.head())
print(data.tail())

