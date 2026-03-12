"""
Enhanced trading environment.

Key improvements over the original:
  - State includes technical indicators instead of raw returns
  - Position and unrealized PnL are part of the state
  - Stop-loss automatically closes losing positions
  - Reward has a risk-adjusted component (rolling Sharpe penalty)
  - Trade log records every entry/exit for post-analysis
"""

import numpy as np
from features import build_feature_matrix, N_MARKET_FEATURES


class TradingEnv:
    """
    Actions: 0 = Hold, 1 = Buy (go long), 2 = Sell (go short)

    State vector per step:
      [6 market features] + [position_encoded] + [unrealized_pnl]
      = 8 features total

    Reward is risk-adjusted: raw PnL minus a penalty proportional
    to rolling volatility of recent step returns. This discourages
    the agent from taking high-variance paths even if the mean
    return looks acceptable.
    """

    STATE_DIM = N_MARKET_FEATURES + 2  # 6 indicators + position + unrealized pnl

    def __init__(
        self,
        prices: np.ndarray,
        transaction_cost: float = 0.001,
        stop_loss: float = 0.02,
        reward_scaling: float = 100.0,
        vol_penalty: float = 0.1,
    ):
        self.prices = np.array(prices, dtype=np.float64)
        self.features = build_feature_matrix(self.prices)
        self.n_steps = len(self.prices)

        self.transaction_cost = transaction_cost
        self.stop_loss = stop_loss          # close position if loss exceeds this fraction
        self.reward_scaling = reward_scaling
        self.vol_penalty = vol_penalty

        self.action_space = 3
        self.observation_space_dim = self.STATE_DIM

        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0          # 0: flat, 1: long, -1: short
        self.entry_price = 0.0
        self.equity = 1.0
        self.equity_curve = [1.0]
        self.step_returns = []
        self.trade_log = []        # list of dicts, one per closed trade
        self.actions = []
        return self._get_state()

    def _get_state(self):
        market = self.features[self.current_step]

        # Unrealized PnL as a fraction
        if self.position != 0 and self.entry_price > 0:
            price = self.prices[self.current_step]
            unrealized = self.position * (price - self.entry_price) / self.entry_price
        else:
            unrealized = 0.0

        # Encode position as a single float: -1, 0, or 1
        return np.append(market, [float(self.position), unrealized]).astype(np.float32)

    def _rolling_vol_penalty(self):
        """Penalize reward by recent return volatility."""
        if len(self.step_returns) < 10:
            return 0.0
        recent = self.step_returns[-20:]
        return self.vol_penalty * np.std(recent)

    def step(self, action: int):
        assert 0 <= action < 3, f"Invalid action: {action}"

        price = self.prices[self.current_step]
        reward = 0.0
        done = False

        # -- Stop-loss check before processing action --
        if self.position != 0 and self.entry_price > 0:
            unrealized = self.position * (price - self.entry_price) / self.entry_price
            if unrealized < -self.stop_loss:
                # Force close the position
                pnl = unrealized - self.transaction_cost
                reward += pnl
                self.trade_log.append({
                    "entry": self.entry_price,
                    "exit": price,
                    "direction": self.position,
                    "pnl": pnl,
                    "closed_by": "stop_loss",
                    "step": self.current_step,
                })
                self.position = 0
                self.entry_price = 0.0
                action = 0  # override to hold after stop-loss

        # -- Process action --
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = price
                reward -= self.transaction_cost
            elif self.position == -1:
                # Close short, open long
                pnl = (self.entry_price - price) / self.entry_price - self.transaction_cost
                reward += pnl
                self.trade_log.append({
                    "entry": self.entry_price,
                    "exit": price,
                    "direction": -1,
                    "pnl": pnl,
                    "closed_by": "signal",
                    "step": self.current_step,
                })
                self.position = 1
                self.entry_price = price
                reward -= self.transaction_cost

        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = price
                reward -= self.transaction_cost
            elif self.position == 1:
                # Close long, open short
                pnl = (price - self.entry_price) / self.entry_price - self.transaction_cost
                reward += pnl
                self.trade_log.append({
                    "entry": self.entry_price,
                    "exit": price,
                    "direction": 1,
                    "pnl": pnl,
                    "closed_by": "signal",
                    "step": self.current_step,
                })
                self.position = -1
                self.entry_price = price
                reward -= self.transaction_cost

        else:  # Hold — mark-to-market on open position
            if self.position != 0 and self.current_step > 0:
                prev_price = self.prices[self.current_step - 1]
                reward += self.position * (price - prev_price) / prev_price

        # Risk-adjusted reward
        reward -= self._rolling_vol_penalty()
        reward *= self.reward_scaling

        self.current_step += 1
        self.actions.append(action)
        self.step_returns.append(reward)
        self.equity *= (1 + reward / self.reward_scaling)
        self.equity_curve.append(self.equity)

        # End of episode
        if self.current_step >= self.n_steps - 1:
            done = True
            if self.position != 0:
                final_price = self.prices[self.current_step]
                pnl = self.position * (final_price - self.entry_price) / self.entry_price - self.transaction_cost
                self.trade_log.append({
                    "entry": self.entry_price,
                    "exit": final_price,
                    "direction": self.position,
                    "pnl": pnl,
                    "closed_by": "eod",
                    "step": self.current_step,
                })
                self.position = 0

        return self._get_state(), reward, done, {}
