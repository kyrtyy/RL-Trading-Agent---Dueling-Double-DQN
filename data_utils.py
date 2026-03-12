"""
Data loading and preprocessing utilities.

Supports:
  - Loading price data from CSV (handles common column naming variations)
  - Synthetic price generation via Geometric Brownian Motion
  - Walk-forward train/test split (no look-ahead bias)
"""

import numpy as np
import pandas as pd
import os


def load_csv_prices(file_path: str) -> np.ndarray:
    """
    Loads closing prices from a CSV file.

    Tries common column names in order. If none match, falls back to
    the first numeric column. Strips NaNs and negative values.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV not found: {file_path}")

    df = pd.read_csv(file_path)

    # Try standard column names
    candidates = ["Close", "close", "Adj Close", "adj_close", "Price", "price", "Last"]
    for col in candidates:
        if col in df.columns:
            prices = df[col].dropna().values
            prices = prices[prices > 0]
            print(f"Loaded {len(prices)} price points from column '{col}'")
            return prices.astype(np.float64)

    # Fallback: first numeric column
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            prices = df[col].dropna().values
            prices = prices[prices > 0]
            print(f"Using fallback column '{col}' ({len(prices)} points)")
            return prices.astype(np.float64)

    raise ValueError(f"No usable price column found in {file_path}")


def generate_gbm_prices(
    length: int = 2000,
    mu: float = 0.0003,
    sigma: float = 0.012,
    start: float = 100.0,
    seed: int = None,
) -> np.ndarray:
    """
    Generates a synthetic price series using Geometric Brownian Motion.

    mu    — expected daily drift (0.0003 ≈ ~7.5% annualized)
    sigma — daily volatility (0.012 ≈ ~19% annualized)
    """
    if seed is not None:
        np.random.seed(seed)
    log_returns = np.random.normal(mu - 0.5 * sigma ** 2, sigma, length)
    prices = start * np.exp(np.cumsum(log_returns))
    prices = np.insert(prices, 0, start)
    return prices


def train_test_split(prices: np.ndarray, train_ratio: float = 0.7):
    """
    Walk-forward split — train on the first portion, test on the rest.
    No shuffling; temporal order must be preserved to avoid look-ahead bias.
    """
    split = int(len(prices) * train_ratio)
    # Ensure test set is long enough to be meaningful
    min_test = 200
    if len(prices) - split < min_test:
        split = len(prices) - min_test

    train_prices = prices[:split]
    test_prices  = prices[split:]
    print(f"Train: {len(train_prices)} steps | Test: {len(test_prices)} steps")
    return train_prices, test_prices
