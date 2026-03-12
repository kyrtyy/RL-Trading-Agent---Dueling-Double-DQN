"""
Computes technical indicators used as the agent's state.

Instead of feeding raw returns into the network, we give it
meaningful features that traders actually use: momentum (RSI),
trend (MACD, EMA ratio), volatility (ATR, Bollinger %B),
and context (position, unrealized PnL).

This alone is one of the biggest accuracy improvements over
the original raw-returns approach.
"""

import numpy as np


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index, normalized to [0, 1]."""
    deltas = np.diff(prices)
    rsi = np.full(len(prices), np.nan)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / (avg_loss + 1e-8)
        rsi[i + 1] = 1 / (1 + rs)  # scaled to [0,1] instead of 0-100

    return rsi


def compute_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD histogram (MACD line minus signal line), normalized by price."""
    def ema(arr, span):
        result = np.full(len(arr), np.nan)
        result[span - 1] = np.mean(arr[:span])
        k = 2 / (span + 1)
        for i in range(span, len(arr)):
            result[i] = arr[i] * k + result[i - 1] * (1 - k)
        return result

    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(np.where(np.isnan(macd_line), 0, macd_line), signal)
    histogram = (macd_line - signal_line) / (prices + 1e-8)
    return histogram


def compute_bollinger_pct_b(prices: np.ndarray, period: int = 20) -> np.ndarray:
    """%B position within Bollinger Bands. 0 = lower band, 1 = upper band."""
    pct_b = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1: i + 1]
        mid = np.mean(window)
        std = np.std(window)
        upper = mid + 2 * std
        lower = mid - 2 * std
        band_range = upper - lower
        pct_b[i] = (prices[i] - lower) / (band_range + 1e-8)
    return pct_b


def compute_atr_ratio(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range as a fraction of price.
    Measures recent volatility relative to current price level.
    """
    atr = np.full(len(prices), np.nan)
    tr_values = np.abs(np.diff(prices))  # simplified TR (no gaps/volume)
    for i in range(period, len(prices)):
        atr[i] = np.mean(tr_values[i - period: i]) / (prices[i] + 1e-8)
    return atr


def compute_ema_ratio(prices: np.ndarray, span: int = 20) -> np.ndarray:
    """Price / EMA(span) — shows how far price has deviated from its trend."""
    ema = np.full(len(prices), np.nan)
    ema[span - 1] = np.mean(prices[:span])
    k = 2 / (span + 1)
    for i in range(span, len(prices)):
        ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
    ratio = prices / (ema + 1e-8) - 1.0  # centered around 0
    return ratio


def compute_rolling_volatility(prices: np.ndarray, period: int = 20) -> np.ndarray:
    """Annualized rolling volatility of log returns."""
    log_returns = np.diff(np.log(prices + 1e-8))
    vol = np.full(len(prices), np.nan)
    for i in range(period, len(prices)):
        vol[i] = np.std(log_returns[i - period: i]) * np.sqrt(252)
    return vol


def build_feature_matrix(prices: np.ndarray) -> np.ndarray:
    """
    Assembles all indicators into a (T, 6) feature matrix.
    NaN rows at the start are filled with 0 — the agent just
    sees neutral values until enough history is available.
    """
    features = np.column_stack([
        compute_rsi(prices),
        compute_macd(prices),
        compute_bollinger_pct_b(prices),
        compute_atr_ratio(prices),
        compute_ema_ratio(prices),
        compute_rolling_volatility(prices),
    ])
    # Replace NaN with 0 and clip extreme values
    features = np.nan_to_num(features, nan=0.0)
    features = np.clip(features, -5.0, 5.0)
    return features


# Number of market features (used by env and agent to set state_dim)
N_MARKET_FEATURES = 6
