# RL Trading Agent — Dueling Double DQN

| | |
|---|---|
| **Author** | Yash Dewangan |
| **Language** | Python 3.9+ |
| **Framework** | PyTorch |
| **Algorithm** | Dueling Double DQN + Prioritized Experience Replay |

---

## What This Is

A reinforcement learning agent that learns to trade a single asset going long, short, or flat. Using a custom trading environment and a significantly improved DQN architecture.

I built this starting from a basic DQN implementation and progressively upgraded it. The original version used raw price returns as state and vanilla DQN. This version uses technical indicators as features, a dueling double DQN, prioritized experience replay, stop-loss logic, and a proper train/test split with full performance analytics. The architecture changes alone cut down overestimation bias noticeably and the technical features give the agent actual market context instead of noise.

---

## Architecture Overview

```
Price Series
     │
     ▼
features.py ──► RSI, MACD histogram, Bollinger %B,
                ATR ratio, EMA ratio, Rolling vol
                          │
                          ▼
                    env.py (TradingEnv)
                    ┌──────────────────────────┐
                    │  state = [6 indicators]  │
                    │        + [position]      │
                    │        + [unrealized PnL]│
                    │                          │
                    │  reward = risk-adjusted  │
                    │  stop-loss @ -2%         │
                    └──────────────────────────┘
                          │
                          ▼
                    agent.py (DDQNAgent)
                    ┌──────────────────────────┐
                    │  Dueling DQN             │
                    │   encoder → V(s)         │
                    │          → A(s,a)        │
                    │  Q = V + A - mean(A)     │
                    │                          │
                    │  Double DQN target       │
                    │  Prioritized replay      │
                    │  Huber loss + grad clip  │
                    └──────────────────────────┘
```

---

## What Changed from the Original

| Area | Before | After |
|---|---|---|
| State representation | Raw returns window (29 values) | 8 features: RSI, MACD, BB%B, ATR, EMA ratio, vol, position, PnL |
| Network | Vanilla MLP | Dueling architecture (V + A streams) |
| Q-learning | Standard DQN (overestimates Q) | Double DQN (policy selects, target evaluates) |
| Replay buffer | Uniform random sampling | Prioritized: high TD-error transitions sampled more |
| Reward | Raw PnL only | Risk-adjusted: PnL minus rolling volatility penalty |
| Risk controls | None | Stop-loss closes position at -2% from entry |
| Evaluation | Single run, no test split | Walk-forward 70/30 split, no look-ahead |
| Metrics | Sharpe + max drawdown | Sharpe, Sortino, Calmar, win rate, profit factor, avg win/loss |
| Model persistence | None | Save/load checkpoints |

---

## Project Structure

```
rl-trading-agent/
│
├── features.py       — technical indicators (RSI, MACD, Bollinger, ATR, EMA, vol)
├── env.py            — trading environment with stop-loss and risk-adjusted reward
├── agent.py          — Dueling Double DQN + Prioritized Experience Replay
├── metrics.py        — full performance analytics suite
├── data_utils.py     — CSV loading, GBM generator, walk-forward split
├── train.py          — training loop, evaluation, plots, CLI
└── README.md
```

---

## Setup

```bash
pip install torch numpy pandas matplotlib
```

No other dependencies. PyTorch will use a GPU automatically if one is available.

---

## Running

**Synthetic data (quickest start):**
```bash
python train.py --episodes 80
```

**Your own price CSV:**
```bash
python train.py --csv data/AAPL.csv --episodes 100
```

The CSV just needs a column named `Close`, `close`, `Adj Close`, or `Price`. Standard Yahoo Finance downloads work without any pre-processing.

**Resume from a saved checkpoint:**
```bash
python train.py --load model.pt --csv data/AAPL.csv
```

**Evaluate only (no training):**
```bash
python train.py --eval-only --load model.pt --csv data/AAPL.csv
```

---

## Key Arguments

| Argument | Default | Notes |
|---|---|---|
| `--csv` | None | Path to price CSV. Omit to use synthetic GBM data |
| `--episodes` | 80 | More episodes = more training, diminishing returns past ~150 |
| `--lr` | 5e-4 | Adam learning rate. Try 1e-4 for noisier datasets |
| `--stop-loss` | 0.02 | Auto-close if position loses more than 2% from entry |
| `--transaction-cost` | 0.001 | 0.1% per trade. Increase for illiquid assets |
| `--train-ratio` | 0.7 | Fraction of data used for training |
| `--hidden-dim` | 128 | Network hidden layer size |
| `--seed` | None | Set for reproducible runs |
| `--no-plot` | False | Skip saving the results plot |

---

## Output

After training, `train.py` prints a report for both train and test sets:

```
========================================
  PERFORMANCE REPORT
========================================
  Total Return           +12.34%
  Annualized Return      +18.76%
  Sharpe Ratio           1.4231
  Sortino Ratio          2.1044
  Calmar Ratio           0.9832
  Max Drawdown           -8.21%
  Total Trades           47
  Win Rate               57.4%
  Profit Factor          1.6300
  Avg Win/Loss Ratio     1.2100
========================================
```

And saves `results.png` with 6 panels: price series, training reward, loss curve, action distribution, train equity, and test equity.

---

## Technical Notes

**Why Dueling DQN?**
In many timesteps the agent is holding and the action choice doesn't matter, what matters is the state value. The dueling architecture lets the value stream learn this without interfering with the advantage estimates. It converges faster and produces more stable Q-values in flat market conditions.

**Why Double DQN?**
Vanilla DQN systematically overestimates Q values because it uses the same network to select and evaluate the best next action. Double DQN decouples this: the policy net picks the action, the lagged target net evaluates it. The result is less biased Q targets and more stable training.

**Why Prioritized Replay?**
Uniform sampling wastes gradient updates on transitions the agent already handles well. PER uses TD error as a proxy for how surprising a transition is and samples those more often. Importance-sampling weights correct for the resulting distribution shift.

**Why technical indicators over raw returns?**
Raw returns are very noisy on their own. RSI, MACD, and Bollinger %B compress multi-period price history into features with known market relevance. The agent gets context about whether the market is trending, mean-reverting, or volatile, the kind of context that determines whether a trade makes sense.

---

## Limitations

- Single asset only: no cross-asset features or correlation signals
- No volume data in the default feature set (can be added to `features.py`)
- Synthetic GBM doesn't capture regime changes, fat tails, or autocorrelation present in real markets
- Binary position sizing: the agent is always fully in or fully out, no fractional sizing

---

## About Me

**Yash Dewangan:** I'm a 4th year BS-MS physics student at Indian institute of Science, interested in the intersection of physics, machine learning and quantitative finance.

- **GitHub**: [https://github.com/kyrtyy]
- **LinkedIn**: [https://linkedin.com/in/yash-dewangan-a61619250]
