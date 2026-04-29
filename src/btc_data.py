"""
Bitcoin data pipeline for Griffin anomaly detection PoC.

Downloads 1-second kline data from Binance API and extracts features
for the Griffin model to learn "normal" BTC behavior and detect anomalies.

Features per tick (8 dims):
    0: price_return     - log return from previous close
    1: volume_norm      - log-normalized volume
    2: high_low_spread  - (high - low) / close — intra-candle volatility
    3: close_vs_open    - (close - open) / open — candle direction
    4: volatility_5     - rolling 5-tick std of returns
    5: volatility_20    - rolling 20-tick std of returns
    6: momentum_10      - 10-tick momentum (sum of returns)
    7: volume_ratio     - current volume / rolling 20-tick avg volume
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "btc_data"
DATA_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "price_return", "volume_norm", "high_low_spread", "close_vs_open",
    "volatility_5", "volatility_20", "momentum_10", "volume_ratio",
]
NUM_FEATURES = len(FEATURE_NAMES)


def download_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    days: int = 365,
    save: bool = True,
) -> pd.DataFrame:
    """
    Download kline (candlestick) data from Binance public API.

    Args:
        symbol: Trading pair
        interval: Candle interval - '1m' for 1-minute (default), '1s' for 1-second
        days: Number of days to download (default 365 = 1 year)
        save: Whether to save to CSV

    Returns:
        DataFrame with OHLCV data
    """
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []

    # Binance returns max 1000 candles per request
    end_time = int(time.time() * 1000)  # now in ms
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    current_start = start_time
    request_count = 0

    # Calculate expected requests
    if interval == "1s":
        candles_per_day = 86400
    elif interval == "1m":
        candles_per_day = 1440
    elif interval == "5m":
        candles_per_day = 288
    else:
        candles_per_day = 1440  # default to 1m

    expected_candles = days * candles_per_day
    expected_requests = expected_candles // 1000 + 1
    max_requests = expected_requests + 100

    print(f"Downloading {days} days of {interval} {symbol} data "
          f"(~{expected_candles:,} candles, ~{expected_requests} requests)...")

    while current_start < end_time and request_count < max_requests:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": 1000,
        }

        try:
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Request failed: {e}, retrying in 2s...")
            time.sleep(2)
            continue

        if not data:
            break

        all_data.extend(data)
        current_start = data[-1][0] + 1  # next ms after last candle
        request_count += 1

        if request_count % 100 == 0:
            pct = len(all_data) / expected_candles * 100
            print(f"  Downloaded {len(all_data):,} candles ({pct:.0f}%)...")

        # Respect rate limits
        time.sleep(0.05)

    print(f"  Total: {len(all_data):,} candles from {request_count} requests")

    # Parse into DataFrame
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ])

    # Convert types
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df["trades"] = df["trades"].astype(int)

    # Drop duplicates
    df = df.drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    if save:
        path = DATA_DIR / f"{symbol}_{interval}_{days}d.csv"
        df.to_csv(path, index=False)
        print(f"  Saved to {path}")

    return df


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract normalized features from OHLCV data.

    Returns:
        np.ndarray of shape (n_ticks, NUM_FEATURES)
    """
    close = df["close"].values
    volume = df["volume"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values

    n = len(close)

    # 1. Log returns
    returns = np.zeros(n)
    returns[1:] = np.log(close[1:] / close[:-1])

    # 2. Log-normalized volume
    vol_log = np.log1p(volume)
    vol_mean, vol_std = vol_log.mean(), vol_log.std() + 1e-8
    volume_norm = (vol_log - vol_mean) / vol_std

    # 3. High-low spread (intra-candle volatility)
    spread = (high - low) / (close + 1e-8)

    # 4. Close vs open direction
    close_vs_open = (close - open_) / (open_ + 1e-8)

    # 5. Rolling volatility (5-tick)
    vol5 = pd.Series(returns).rolling(5, min_periods=1).std().fillna(0).values

    # 6. Rolling volatility (20-tick)
    vol20 = pd.Series(returns).rolling(20, min_periods=1).std().fillna(0).values

    # 7. Momentum (10-tick sum of returns)
    mom10 = pd.Series(returns).rolling(10, min_periods=1).sum().fillna(0).values

    # 8. Volume ratio (current / rolling 20-tick average)
    vol_ma20 = pd.Series(volume).rolling(20, min_periods=1).mean().fillna(1).values
    vol_ratio = volume / (vol_ma20 + 1e-8)

    # Stack features
    features = np.stack([
        returns, volume_norm, spread, close_vs_open,
        vol5, vol20, mom10, vol_ratio,
    ], axis=1)

    # Clip extreme outliers
    features = np.clip(features, -10, 10)

    return features.astype(np.float32)


def find_anomaly_events(
    df: pd.DataFrame,
    threshold_pct: float = 0.5,
    window: int = 60,
) -> list[dict]:
    """
    Find significant price movement events in the data.

    A "significant event" is when the price moves more than threshold_pct%
    within a window of `window` ticks.

    Returns:
        List of dicts with event info
    """
    close = df["close"].values
    events = []

    for i in range(window, len(close)):
        pct_change = (close[i] - close[i - window]) / close[i - window] * 100

        if abs(pct_change) >= threshold_pct:
            events.append({
                "index": i,
                "time": str(df["open_time"].iloc[i]),
                "price": close[i],
                "pct_change": round(pct_change, 3),
                "direction": "PUMP" if pct_change > 0 else "CRASH",
            })

    # Deduplicate: keep only the peak of each event cluster
    if not events:
        return events

    deduped = [events[0]]
    for e in events[1:]:
        if e["index"] - deduped[-1]["index"] > window * 2:
            deduped.append(e)
        elif abs(e["pct_change"]) > abs(deduped[-1]["pct_change"]):
            deduped[-1] = e

    return deduped


def prepare_sequences(
    features: np.ndarray,
    seq_len: int = 512,
    stride: int = 256,
) -> np.ndarray:
    """
    Split feature array into overlapping sequences for training.

    Returns:
        np.ndarray of shape (n_sequences, seq_len, n_features)
    """
    sequences = []
    for i in range(0, len(features) - seq_len, stride):
        sequences.append(features[i : i + seq_len])

    return np.array(sequences, dtype=np.float32)


def load_or_download(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    days: int = 365,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load cached data or download fresh."""
    path = DATA_DIR / f"{symbol}_{interval}_{days}d.csv"

    if path.exists():
        print(f"Loading cached data from {path}")
        df = pd.read_csv(path)
        df["open_time"] = pd.to_datetime(df["open_time"])
        df["close_time"] = pd.to_datetime(df["close_time"])
    else:
        df = download_binance_klines(symbol, interval, days)

    features = extract_features(df)
    print(f"Data: {len(df):,} ticks ({interval} candles, {days} days)")
    print(f"Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    print(f"Date range: {df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}")

    # For 1m data, use wider windows for anomaly detection
    if interval == "1m":
        events = find_anomaly_events(df, threshold_pct=2.0, window=60)
        label = ">2% in 60 min"
    else:
        events = find_anomaly_events(df, threshold_pct=0.5, window=60)
        label = ">0.5% in 60s"

    if events:
        print(f"Found {len(events)} significant price movements ({label})")
        for e in events[:10]:
            print(f"  {e['direction']}: {e['pct_change']:+.2f}% at {e['time']}")
        if len(events) > 10:
            print(f"  ... and {len(events) - 10} more")

    return df, features


if __name__ == "__main__":
    df, features = load_or_download(interval="1m", days=365)
    print(f"\nFeature shape: {features.shape}")
    print(f"Feature stats:")
    for i, name in enumerate(FEATURE_NAMES):
        col = features[:, i]
        print(f"  {name:20s}: mean={col.mean():.4f}, std={col.std():.4f}, "
              f"min={col.min():.4f}, max={col.max():.4f}")
