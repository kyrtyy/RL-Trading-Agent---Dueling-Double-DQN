"""
Performance analytics for evaluating the trained agent.

All functions take either an equity curve (list/array of portfolio values
starting at 1.0) or a list of step returns. Where annualization is needed,
252 trading days is assumed.
"""

import numpy as np
from typing import List, Dict


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    r = np.array(returns)
    if r.std() < 1e-8:
        return 0.0
    return (r.mean() - risk_free) / (r.std() + 1e-8) * np.sqrt(252)


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Annualized Sortino ratio — like Sharpe but only penalizes downside volatility.
    Better metric when return distribution is skewed.
    """
    r = np.array(returns)
    downside = r[r < risk_free] - risk_free
    downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-8
    return (r.mean() - risk_free) / (downside_std + 1e-8) * np.sqrt(252)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough decline as a fraction."""
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-8)
    return float(dd.min())


def calmar_ratio(equity_curve: np.ndarray, n_trading_days: int = None) -> float:
    """
    Annualized return divided by max drawdown.
    Useful for comparing strategies with different drawdown profiles.
    """
    equity = np.array(equity_curve)
    if n_trading_days is None:
        n_trading_days = len(equity)
    ann_return = (equity[-1] / equity[0]) ** (252 / max(n_trading_days, 1)) - 1
    mdd = abs(max_drawdown(equity))
    if mdd < 1e-8:
        return 0.0
    return ann_return / mdd


def win_rate(trade_log: List[Dict]) -> float:
    """Fraction of closed trades that were profitable."""
    if not trade_log:
        return 0.0
    wins = sum(1 for t in trade_log if t["pnl"] > 0)
    return wins / len(trade_log)


def profit_factor(trade_log: List[Dict]) -> float:
    """Gross profit / gross loss. > 1 means the strategy makes money overall."""
    gross_profit = sum(t["pnl"] for t in trade_log if t["pnl"] > 0)
    gross_loss   = abs(sum(t["pnl"] for t in trade_log if t["pnl"] < 0))
    if gross_loss < 1e-8:
        return float("inf")
    return gross_profit / gross_loss


def avg_win_loss_ratio(trade_log: List[Dict]) -> float:
    """Average winning trade size / average losing trade size."""
    wins   = [t["pnl"] for t in trade_log if t["pnl"] > 0]
    losses = [abs(t["pnl"]) for t in trade_log if t["pnl"] < 0]
    avg_win  = np.mean(wins)  if wins   else 0.0
    avg_loss = np.mean(losses) if losses else 1e-8
    return avg_win / (avg_loss + 1e-8)


def annualized_return(equity_curve: np.ndarray, n_trading_days: int = None) -> float:
    equity = np.array(equity_curve)
    if n_trading_days is None:
        n_trading_days = len(equity)
    return (equity[-1] / equity[0]) ** (252 / max(n_trading_days, 1)) - 1


def full_report(equity_curve: np.ndarray, step_returns: np.ndarray, trade_log: List[Dict]) -> Dict:
    """
    Returns a dict of all metrics. Print-friendly via print_report().
    """
    return {
        "Total Return":        f"{(equity_curve[-1] - 1) * 100:.2f}%",
        "Annualized Return":   f"{annualized_return(equity_curve) * 100:.2f}%",
        "Sharpe Ratio":        f"{sharpe_ratio(step_returns):.4f}",
        "Sortino Ratio":       f"{sortino_ratio(step_returns):.4f}",
        "Calmar Ratio":        f"{calmar_ratio(equity_curve):.4f}",
        "Max Drawdown":        f"{max_drawdown(equity_curve) * 100:.2f}%",
        "Total Trades":        len(trade_log),
        "Win Rate":            f"{win_rate(trade_log) * 100:.1f}%",
        "Profit Factor":       f"{profit_factor(trade_log):.4f}",
        "Avg Win/Loss Ratio":  f"{avg_win_loss_ratio(trade_log):.4f}",
    }


def print_report(metrics: Dict):
    print("\n" + "=" * 40)
    print("  PERFORMANCE REPORT")
    print("=" * 40)
    for key, val in metrics.items():
        print(f"  {key:<22} {val}")
    print("=" * 40 + "\n")
