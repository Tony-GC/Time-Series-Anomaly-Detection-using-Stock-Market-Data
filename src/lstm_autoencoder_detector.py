from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense,
    Dropout,
    LayerNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


"""
Improved LSTM autoencoder anomaly detector.

This version is designed to be more useful for your project by focusing on:
1. Better inputs
2. Better thresholding
3. Better interpretation
"""


# ---------------------------
# Paths
# ---------------------------
INPUT_PATH = Path("data/processed/market_data_features_2021_2026.csv")
OUTPUT_DATA_PATH = Path("data/processed/lstm_autoencoder_results.csv")

FIGURES_DIR = Path("results/figures/lstm_autoencoder")
TABLES_DIR = Path("results/tables")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = TABLES_DIR / "lstm_autoencoder_summary.csv"


# ---------------------------
# Configuration
# ---------------------------
TRAIN_END_DATE = pd.Timestamp("2024-12-31")
# SEQUENCE_LENGTH = 30
SEQUENCE_LENGTH = 15

RANDOM_STATE = 42
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

FEATURES = [
    "simple_return",
    "abs_simple_return",
    "return_5d",
    "return_10d",
    "gap_pct",
    "hl_range_pct",
    "body_to_range",
    "dist_ma_20",
    "simple_vol_20",
    "vol_ratio_5_20",
    "z_simple_return_20",
    "relative_volume_20",
    "volume_z_20",
    "SP500_simple_return",
    "NASDAQ_simple_return",
    "VIX_simple_return",
    "VIX_z_simple_20",
    "excess_simple_return_sp500",
    "excess_simple_return_nasdaq",
]

# Z_SIMPLE_MAX = 2.0
# VOLUME_Z_MAX = 2.5
# VIX_Z_MAX = 2.0
Z_SIMPLE_MAX = 1.5
VOLUME_Z_MAX = 2.0
VIX_Z_MAX = 1.5

# THRESHOLD_MULTIPLIER = 6.0
# TRAIN_THRESHOLD_QUANTILE = 0.9975
THRESHOLD_MULTIPLIER = 4.5
TRAIN_THRESHOLD_QUANTILE = 0.995

# EVENT_GAP_BARS = 5
EVENT_GAP_BARS = 3

PLOT_RAW_ANOMALIES = True


np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def robust_scale_train_full(train_df: pd.DataFrame, full_df: pd.DataFrame):
    medians = train_df.median()
    q1 = train_df.quantile(0.25)
    q3 = train_df.quantile(0.75)
    iqr = (q3 - q1).replace(0, 1.0)

    train_scaled = (train_df - medians) / iqr
    full_scaled = (full_df - medians) / iqr

    return train_scaled, full_scaled


def make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    sequences = []
    for i in range(seq_len - 1, len(X)):
        sequences.append(X[i - seq_len + 1 : i + 1])
    return np.array(sequences)


def last_step_mse(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return np.mean((x_true[:, -1, :] - x_pred[:, -1, :]) ** 2, axis=1)


def robust_threshold(train_errors: np.ndarray, multiplier: float = THRESHOLD_MULTIPLIER) -> float:
    train_errors = np.asarray(train_errors)
    train_errors = train_errors[~np.isnan(train_errors)]

    median = np.median(train_errors)
    mad = np.median(np.abs(train_errors - median))

    robust_sigma = 1.4826 * mad
    if robust_sigma == 0:
        robust_sigma = np.std(train_errors) if np.std(train_errors) > 0 else 1e-6

    return median + multiplier * robust_sigma


# def combined_threshold(train_errors: np.ndarray) -> float:
#     mad_thr = robust_threshold(train_errors, THRESHOLD_MULTIPLIER)
#     q_thr = np.quantile(train_errors, TRAIN_THRESHOLD_QUANTILE)
#     return max(mad_thr, q_thr)

# def combined_threshold(train_errors: np.ndarray) -> float:
#     return np.quantile(train_errors, 0.995)

def combined_threshold(train_errors: np.ndarray) -> float:
    mad_thr = robust_threshold(train_errors, THRESHOLD_MULTIPLIER)
    q_thr = np.quantile(train_errors, TRAIN_THRESHOLD_QUANTILE)
    return 0.5 * mad_thr + 0.5 * q_thr


def build_lstm_autoencoder(seq_len: int, n_features: int) -> Model:
    inputs = Input(shape=(seq_len, n_features))

    x = LSTM(48, return_sequences=True, activation="tanh")(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(24, activation="tanh")(x)
    x = LayerNormalization()(x)

    x = RepeatVector(seq_len)(x)

    x = LSTM(24, return_sequences=True, activation="tanh")(x)
    x = Dropout(0.2)(x)
    x = LSTM(48, return_sequences=True, activation="tanh")(x)

    outputs = TimeDistributed(Dense(n_features))(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.Huber()
    )
    return model


def prepare_group_data(group: pd.DataFrame):
    group = group.sort_values("Date").copy()

    available_features = [c for c in FEATURES if c in group.columns]
    if not available_features:
        raise ValueError("No requested features are available.")

    train_mask = group["Date"] <= TRAIN_END_DATE
    train_df = group.loc[train_mask].copy()

    if train_df.empty:
        raise ValueError("Training split is empty.")

    X_train = train_df[available_features].replace([np.inf, -np.inf], np.nan).copy()
    X_full = group[available_features].replace([np.inf, -np.inf], np.nan).copy()

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_full = X_full.fillna(train_medians)

    valid_cols = [c for c in available_features if X_train[c].notna().any() and X_full[c].notna().any()]
    X_train = X_train[valid_cols]
    X_full = X_full[valid_cols]

    if X_train.shape[1] == 0:
        raise ValueError("No valid features remain after NaN handling.")

    X_train_scaled, X_full_scaled = robust_scale_train_full(X_train, X_full)

    return train_df, group, valid_cols, X_train_scaled, X_full_scaled


def build_clean_training_mask(train_df: pd.DataFrame) -> np.ndarray:
    n = len(train_df)
    keep = []

    z_col = "z_simple_return_20" if "z_simple_return_20" in train_df.columns else None
    vol_col = "volume_z_20" if "volume_z_20" in train_df.columns else None
    vix_col = "VIX_z_simple_20" if "VIX_z_simple_20" in train_df.columns else None

    for end_idx in range(SEQUENCE_LENGTH - 1, n):
        window = train_df.iloc[end_idx - SEQUENCE_LENGTH + 1 : end_idx + 1]
        ok = True

        if z_col is not None:
            vals = window[z_col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                ok = False
            elif np.max(np.abs(vals)) > Z_SIMPLE_MAX:
                ok = False

        if vol_col is not None:
            vals = window[vol_col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                ok = False
            elif np.max(np.abs(vals)) > VOLUME_Z_MAX:
                ok = False

        if vix_col is not None:
            vals = window[vix_col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                ok = False
            elif np.max(np.abs(vals)) > VIX_Z_MAX:
                ok = False

        keep.append(ok)

    return np.array(keep, dtype=bool)


# def add_event_peaks(full_group: pd.DataFrame) -> pd.DataFrame:
#     full_group = full_group.copy()
#     full_group["lstm_event_group"] = np.nan
#     full_group["lstm_is_event_peak"] = False

#     anomaly_idx = full_group.index[full_group["lstm_is_anomaly"].fillna(False)].to_list()
#     if not anomaly_idx:
#         return full_group

#     group_ids = []
#     current_group = 0
#     prev_idx = anomaly_idx[0]

#     for i, idx in enumerate(anomaly_idx):
#         if i == 0:
#             group_ids.append(current_group)
#             continue

#         if (idx - prev_idx) <= EVENT_GAP_BARS:
#             group_ids.append(current_group)
#         else:
#             current_group += 1
#             group_ids.append(current_group)

#         prev_idx = idx

#     full_group.loc[anomaly_idx, "lstm_event_group"] = group_ids

#     event_peak_idx = (
#         full_group.loc[anomaly_idx]
#         .groupby("lstm_event_group")["lstm_score"]
#         .idxmax()
#     )
#     full_group.loc[event_peak_idx, "lstm_is_event_peak"] = True

#     return full_group
def add_event_peaks_from_flag(
    full_group: pd.DataFrame,
    flag_col: str,
    peak_col: str,
    group_col: str,
    gap_bars: int = EVENT_GAP_BARS,
) -> pd.DataFrame:
    """
    Merge nearby anomaly points into event peaks.

    Points within `gap_bars` rows are treated as one event.
    The event peak is the point with the highest lstm_score in that event.
    """
    full_group = full_group.copy()
    full_group[group_col] = np.nan
    full_group[peak_col] = False

    anomaly_idx = full_group.index[full_group[flag_col].fillna(False)].to_list()
    if not anomaly_idx:
        return full_group

    group_ids = []
    current_group = 0
    prev_idx = anomaly_idx[0]

    for i, idx in enumerate(anomaly_idx):
        if i == 0:
            group_ids.append(current_group)
            continue

        if (idx - prev_idx) <= gap_bars:
            group_ids.append(current_group)
        else:
            current_group += 1
            group_ids.append(current_group)

        prev_idx = idx

    full_group.loc[anomaly_idx, group_col] = group_ids

    event_peak_idx = (
        full_group.loc[anomaly_idx]
        .groupby(group_col)["lstm_score"]
        .idxmax()
    )
    full_group.loc[event_peak_idx, peak_col] = True

    return full_group


def score_group(group: pd.DataFrame) -> pd.DataFrame:
    train_df, full_group, used_features, X_train_scaled, X_full_scaled = prepare_group_data(group)

    X_train_seq = make_sequences(X_train_scaled.values, SEQUENCE_LENGTH)
    X_full_seq = make_sequences(X_full_scaled.values, SEQUENCE_LENGTH)

    if len(X_train_seq) == 0 or len(X_full_seq) == 0:
        raise ValueError("Not enough observations to build sequences.")

    clean_mask = build_clean_training_mask(train_df)
    if clean_mask.sum() >= max(20, int(0.2 * len(clean_mask))):
        X_train_seq_clean = X_train_seq[clean_mask]
    else:
        X_train_seq_clean = X_train_seq

    model = build_lstm_autoencoder(SEQUENCE_LENGTH, X_train_seq_clean.shape[2])

    validation_split = 0.1 if len(X_train_seq_clean) >= 30 else 0.0
    monitor_metric = "val_loss" if validation_split > 0 else "loss"

    callbacks = [
        EarlyStopping(monitor=monitor_metric, patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=3, verbose=0),
    ]

    model.fit(
        X_train_seq_clean,
        X_train_seq_clean,
        validation_split=validation_split,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks
    )

    recon_full = model.predict(X_full_seq, verbose=0)
    full_errors = last_step_mse(X_full_seq, recon_full)

    recon_train = model.predict(X_train_seq_clean, verbose=0)
    train_errors = last_step_mse(X_train_seq_clean, recon_train)

    threshold = combined_threshold(train_errors)

    full_group = full_group.copy()
    full_group["lstm_score"] = np.nan
    full_group.iloc[SEQUENCE_LENGTH - 1 :, full_group.columns.get_loc("lstm_score")] = full_errors

    # full_group["lstm_threshold"] = threshold
    # full_group["lstm_is_anomaly_all"] = full_group["lstm_score"] >= threshold

    # full_group["lstm_is_anomaly"] = False
    # post_train_mask = full_group["Date"] > TRAIN_END_DATE
    # full_group.loc[post_train_mask, "lstm_is_anomaly"] = (
    #     full_group.loc[post_train_mask, "lstm_score"] >= threshold
    # )

    # full_group = add_event_peaks(full_group)
    full_group["lstm_threshold"] = threshold

    # All-sample anomaly flags
    full_group["lstm_is_anomaly_all"] = full_group["lstm_score"] >= threshold

    # Post-train-only anomaly flags
    full_group["lstm_is_anomaly"] = False
    post_train_mask = full_group["Date"] > TRAIN_END_DATE
    full_group.loc[post_train_mask, "lstm_is_anomaly"] = (
        full_group.loc[post_train_mask, "lstm_score"] >= threshold
    )

    # Event peaks for the whole sample
    full_group = add_event_peaks_from_flag(
        full_group,
        flag_col="lstm_is_anomaly_all",
        peak_col="lstm_is_event_peak_all",
        group_col="lstm_event_group_all",
    )

    # Event peaks for post-train evaluation only
    full_group = add_event_peaks_from_flag(
        full_group,
        flag_col="lstm_is_anomaly",
        peak_col="lstm_is_event_peak",
        group_col="lstm_event_group",
    )

    full_group["lstm_used_n_features"] = len(used_features)
    full_group["lstm_features_used"] = ", ".join(used_features)
    full_group["lstm_train_end_date"] = TRAIN_END_DATE
    full_group["lstm_sequence_length"] = SEQUENCE_LENGTH
    full_group["lstm_clean_train_sequences"] = len(X_train_seq_clean)
    full_group["lstm_all_train_sequences"] = len(X_train_seq)

    return full_group


# def summarize_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    # summary = (
    # df.groupby(["SeriesName", "Ticker"], as_index=False)
    #     .agg(
    #         n_rows=("Date", "count"),
    #         n_anomalies_all=("lstm_is_anomaly_all", "sum"),
    #         n_event_peaks_all=("lstm_is_event_peak_all", "sum"),
    #         n_anomalies=("lstm_is_anomaly", "sum"),
    #         n_event_peaks=("lstm_is_event_peak", "sum"),
    #         anomaly_rate=("lstm_is_anomaly", "mean"),
    #         event_peak_rate=("lstm_is_event_peak", "mean"),
    #         avg_score=("lstm_score", "mean"),
    #         max_score=("lstm_score", "max"),
    #         threshold=("lstm_threshold", "max"),
    #         n_features=("lstm_used_n_features", "max"),
    #         clean_train_sequences=("lstm_clean_train_sequences", "max"),
    #         all_train_sequences=("lstm_all_train_sequences", "max"),
    #     )
    #     .sort_values(["SeriesName", "Ticker"])
    #     .reset_index(drop=True)
    # )
def summarize_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-ticker summary table with clearer names.
    """
    summary = (
        df.groupby(["SeriesName", "Ticker"], as_index=False)
        .agg(
            n_rows=("Date", "count"),
            n_anomalies_all=("lstm_is_anomaly_all", "sum"),
            n_event_peaks_all=("lstm_is_event_peak_all", "sum"),
            n_anomalies_detected=("lstm_is_anomaly", "sum"),
            n_event_peaks_detected=("lstm_is_event_peak", "sum"),
            anomaly_rate_detected=("lstm_is_anomaly", "mean"),
            event_peak_rate_detected=("lstm_is_event_peak", "mean"),
            avg_score=("lstm_score", "mean"),
            max_score=("lstm_score", "max"),
            threshold=("lstm_threshold", "max"),
            n_features=("lstm_used_n_features", "max"),
            clean_train_sequences=("lstm_clean_train_sequences", "max"),
            all_train_sequences=("lstm_all_train_sequences", "max"),
        )
        .sort_values(["SeriesName", "Ticker"])
        .reset_index(drop=True)
    )
    return summary    


# def plot_anomalies(group: pd.DataFrame, outdir: Path) -> None:
#     """
#     Save a price plot with:
#     - shaded training period
#     - green vertical line at train end
#     - training and post-train anomalies
#     - training and post-train event peaks
#     """
#     group = group.sort_values("Date").copy()
#     series_name = group["SeriesName"].iloc[0]
#     ticker = group["Ticker"].iloc[0]

#     train_anomalies = group[
#         (group["lstm_is_anomaly_all"]) & (group["Date"] <= TRAIN_END_DATE)
#     ].copy()
#     post_anomalies = group[group["lstm_is_anomaly"]].copy()

#     train_peaks = group[
#         (group["lstm_is_event_peak_all"]) & (group["Date"] <= TRAIN_END_DATE)
#     ].copy()
#     post_peaks = group[group["lstm_is_event_peak"]].copy()

#     plt.figure(figsize=(12, 5))
#     plt.plot(group["Date"], group["Price"], label="Price", linewidth=1.5, color="blue")

#     # Shade training period
#     plt.axvspan(
#         group["Date"].min(),
#         TRAIN_END_DATE,
#         color="lightgrey",
#         alpha=0.15,
#         label="Training period"
#     )

#     # Green vertical line at the end of training
#     plt.axvline(
#         x=TRAIN_END_DATE,
#         color="green",
#         linestyle="-",
#         linewidth=2.0,
#         alpha=0.9,
#         label="Train end"
#     )

#     # Vertical grey lines at ALL event peaks
#     for peak_date in pd.concat([train_peaks["Date"], post_peaks["Date"]]).sort_values():
#         plt.axvline(
#             x=peak_date,
#             color="grey",
#             linestyle="--",
#             linewidth=0.8,
#             alpha=0.45,
#             zorder=1
#         )

#     # Training anomalies
#     if not train_anomalies.empty:
#         plt.scatter(
#             train_anomalies["Date"],
#             train_anomalies["Price"],
#             color="red",
#             s=18,
#             alpha=0.35,
#             label="Train anomaly",
#             zorder=3
#         )

#     # Post-train anomalies
#     if not post_anomalies.empty:
#         plt.scatter(
#             post_anomalies["Date"],
#             post_anomalies["Price"],
#             color="red",
#             s=32,
#             alpha=0.95,
#             label="Post-train anomaly",
#             zorder=4
#         )

#     # Training event peaks
#     if not train_peaks.empty:
#         plt.scatter(
#             train_peaks["Date"],
#             train_peaks["Price"],
#             color="yellow",
#             edgecolor="black",
#             marker="^",
#             s=70,
#             alpha=0.55,
#             label="Train event peak",
#             zorder=5
#         )

#     # Post-train event peaks
#     if not post_peaks.empty:
#         plt.scatter(
#             post_peaks["Date"],
#             post_peaks["Price"],
#             color="yellow",
#             edgecolor="black",
#             marker="^",
#             s=110,
#             alpha=1.0,
#             label="Post-train event peak",
#             zorder=6
#         )

#     plt.title(f"{series_name} ({ticker}) - Improved LSTM Autoencoder Anomalies", pad=20)
#     plt.xlabel("Date")
#     plt.ylabel("Price")
#     plt.legend()
#     plt.tight_layout()

#     filename = f"{ticker}_lstm_anomalies.png"
#     plt.savefig(outdir / filename, dpi=150)
#     plt.close()
def plot_anomalies(group: pd.DataFrame, outdir: Path) -> None:
    """
    Save a price plot with:
    - training-period shading
    - green line at train end
    - vertical grey dashed lines at anomaly dates
    - train/post-train anomalies and event peaks
    """
    group = group.sort_values("Date").copy()
    series_name = group["SeriesName"].iloc[0]
    ticker = group["Ticker"].iloc[0]

    train_anomalies = group[
        (group["lstm_is_anomaly_all"]) & (group["Date"] <= TRAIN_END_DATE)
    ].copy()
    post_anomalies = group[group["lstm_is_anomaly"]].copy()

    train_peaks = group[
        (group["lstm_is_event_peak_all"]) & (group["Date"] <= TRAIN_END_DATE)
    ].copy()
    post_peaks = group[group["lstm_is_event_peak"]].copy()

    plt.figure(figsize=(12, 5))
    plt.plot(group["Date"], group["Price"], label="Price", linewidth=1.5, color="blue")

    # Shade training period
    plt.axvspan(
        group["Date"].min(),
        TRAIN_END_DATE,
        color="lightgrey",
        alpha=0.15,
        label="Training period"
    )

    # Green vertical line at train end
    plt.axvline(
        x=TRAIN_END_DATE,
        color="green",
        linestyle="-",
        linewidth=2.0,
        alpha=0.9,
        label="Train end"
    )

    # Vertical grey lines at anomaly dates, not peak dates
    all_anomaly_dates = pd.concat(
        [train_anomalies["Date"], post_anomalies["Date"]]
    ).sort_values()

    for anomaly_date in all_anomaly_dates:
        plt.axvline(
            x=anomaly_date,
            color="grey",
            linestyle="--",
            linewidth=0.8,
            alpha=0.45,
            zorder=1
        )

    # Training anomalies
    if not train_anomalies.empty:
        plt.scatter(
            train_anomalies["Date"],
            train_anomalies["Price"],
            color="red",
            s=18,
            alpha=0.35,
            label="Train anomaly",
            zorder=3
        )

    # Post-train anomalies
    if not post_anomalies.empty:
        plt.scatter(
            post_anomalies["Date"],
            post_anomalies["Price"],
            color="red",
            s=32,
            alpha=0.95,
            label="Post-train anomaly",
            zorder=4
        )

    # Training event peaks
    if not train_peaks.empty:
        plt.scatter(
            train_peaks["Date"],
            train_peaks["Price"],
            color="yellow",
            edgecolor="black",
            marker="^",
            s=70,
            alpha=0.55,
            label="Train event peak",
            zorder=5
        )

    # Post-train event peaks
    if not post_peaks.empty:
        plt.scatter(
            post_peaks["Date"],
            post_peaks["Price"],
            color="yellow",
            edgecolor="black",
            marker="^",
            s=110,
            alpha=1.0,
            label="Post-train event peak",
            zorder=6
        )

    plt.title(f"{series_name} ({ticker}) - Improved LSTM Autoencoder Anomalies", pad=20)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    filename = f"{ticker}_lstm_anomalies.png"
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
        try:
            result = score_group(group)
            parts.append(result)
            print(f"Finished: {ticker}")
        except Exception as e:
            print(f"Skipped {ticker}: {e}")

    if not parts:
        raise ValueError("No ticker was successfully processed.")

    result_df = pd.concat(parts, ignore_index=True)
    result_df.to_csv(OUTPUT_DATA_PATH, index=False)

    summary = summarize_by_ticker(result_df)
    summary.to_csv(SUMMARY_PATH, index=False)

    for ticker, group in result_df.groupby("Ticker", sort=False):
        plot_anomalies(group, FIGURES_DIR)

    print(f"Saved detailed results to: {OUTPUT_DATA_PATH}")
    print(f"Saved summary table to: {SUMMARY_PATH}")
    print(f"Saved plots to: {FIGURES_DIR}")
    print()
    print(summary)


if __name__ == "__main__":
    main()

