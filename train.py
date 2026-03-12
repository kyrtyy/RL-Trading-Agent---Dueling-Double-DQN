"""
Main entry point. Trains the agent on the train split,
evaluates on the held-out test split, prints a full performance
report, saves the model, and shows plots.

Usage:
  python train.py                          # synthetic data, default settings
  python train.py --csv prices.csv         # your own price data
  python train.py --episodes 100 --lr 3e-4 # tune hyperparameters
  python train.py --load model.pt          # resume from saved checkpoint
  python train.py --eval-only --load model.pt --csv prices.csv
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works everywhere
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_utils import load_csv_prices, generate_gbm_prices, train_test_split
from env import TradingEnv
from agent import DDQNAgent
from metrics import full_report, print_report


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def run_training(env: TradingEnv, agent: DDQNAgent, episodes: int, verbose: bool = True):
    reward_history = []
    equity_finals  = []
    loss_history   = []
    action_counts  = np.zeros(3)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        ep_losses    = []
        done         = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss > 0:
                ep_losses.append(loss)
            state = next_state
            total_reward += reward
            action_counts[action] += 1

        reward_history.append(total_reward)
        equity_finals.append(env.equity_curve[-1])
        if ep_losses:
            loss_history.append(np.mean(ep_losses))

        if verbose and (ep + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(
                f"  Ep {ep+1:>4}/{episodes} | "
                f"Reward: {total_reward:>8.3f} | "
                f"Avg(10): {avg_reward:>8.3f} | "
                f"Equity: {env.equity_curve[-1]:.4f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    return reward_history, equity_finals, loss_history, action_counts


# -----------------------------------------------------------------------
# Evaluation (no exploration, no training)
# -----------------------------------------------------------------------

def run_evaluation(env: TradingEnv, agent: DDQNAgent) -> dict:
    state = env.reset()
    done  = False
    while not done:
        action = agent.select_action(state, training=False)
        state, _, done, _ = env.step(action)

    step_returns = np.array(env.step_returns) / env.reward_scaling
    return full_report(
        np.array(env.equity_curve),
        step_returns,
        env.trade_log,
    )


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

def plot_results(
    train_rewards: list,
    train_losses:  list,
    action_counts: np.ndarray,
    train_env:     TradingEnv,
    test_env:      TradingEnv,
    prices:        np.ndarray,
    output_dir:    str = ".",
):
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # -- Price series --
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(prices, linewidth=0.8, color="#2c7bb6")
    ax0.set_title("Price Series (full dataset)", fontsize=11)
    ax0.set_xlabel("Step")
    ax0.set_ylabel("Price")
    ax0.grid(alpha=0.3)

    # -- Training reward curve --
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(train_rewards, linewidth=1.0, color="#1a9641", alpha=0.7)
    rolling = np.convolve(train_rewards, np.ones(10) / 10, mode="valid")
    ax1.plot(range(9, len(train_rewards)), rolling, color="#1a9641", linewidth=2.0, label="MA10")
    ax1.set_title("Training Reward", fontsize=11)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # -- Training loss --
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(train_losses, linewidth=0.8, color="#d7191c", alpha=0.7)
    ax2.set_title("Training Loss", fontsize=11)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Huber Loss")
    ax2.grid(alpha=0.3)

    # -- Action distribution --
    ax3 = fig.add_subplot(gs[1, 2])
    colors = ["#91bfdb", "#4dac26", "#d01c8b"]
    bars = ax3.bar(["Hold", "Buy", "Sell"], action_counts, color=colors, edgecolor="white")
    for bar, count in zip(bars, action_counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{int(count):,}", ha="center", va="bottom", fontsize=8)
    ax3.set_title("Action Distribution (Training)", fontsize=11)
    ax3.set_ylabel("Count")
    ax3.grid(axis="y", alpha=0.3)

    # -- Train equity curve --
    ax4 = fig.add_subplot(gs[2, 0:2])
    ax4.plot(train_env.equity_curve, linewidth=1.2, color="#4dac26", label="Agent")
    ax4.axhline(1.0, linestyle="--", linewidth=0.8, color="grey", label="Baseline (1.0)")
    ax4.set_title("Train Equity Curve (last episode)", fontsize=11)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Equity")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # -- Test equity curve --
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.plot(test_env.equity_curve, linewidth=1.2, color="#2c7bb6", label="Agent (test)")
    ax5.axhline(1.0, linestyle="--", linewidth=0.8, color="grey", label="Baseline")
    ax5.set_title("Test Equity Curve", fontsize=11)
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Equity")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    out_path = os.path.join(output_dir, "results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dueling Double DQN Trading Agent")
    p.add_argument("--csv",              type=str,   default=None,    help="Path to CSV with price data")
    p.add_argument("--episodes",         type=int,   default=80,      help="Training episodes")
    p.add_argument("--lr",               type=float, default=5e-4,    help="Learning rate")
    p.add_argument("--gamma",            type=float, default=0.99,    help="Discount factor")
    p.add_argument("--transaction-cost", type=float, default=0.001,   help="Per-trade transaction cost")
    p.add_argument("--stop-loss",        type=float, default=0.02,    help="Stop-loss threshold (fraction)")
    p.add_argument("--batch-size",       type=int,   default=128,     help="Replay batch size")
    p.add_argument("--buffer-size",      type=int,   default=20000,   help="Replay buffer capacity")
    p.add_argument("--hidden-dim",       type=int,   default=128,     help="Hidden layer size")
    p.add_argument("--train-ratio",      type=float, default=0.7,     help="Train/test split ratio")
    p.add_argument("--save",             type=str,   default="model.pt", help="Path to save model")
    p.add_argument("--load",             type=str,   default=None,    help="Path to load model from")
    p.add_argument("--eval-only",        action="store_true",          help="Skip training, just evaluate")
    p.add_argument("--no-plot",          action="store_true",          help="Skip plot generation")
    p.add_argument("--seed",             type=int,   default=None,    help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        import torch; import random
        import torch as torch_mod
        torch_mod.manual_seed(args.seed)
        random.seed(args.seed)

    # -- Load data --
    if args.csv:
        prices = load_csv_prices(args.csv)
    else:
        print("No CSV provided — using synthetic GBM prices.")
        prices = generate_gbm_prices(length=2000, seed=args.seed)

    train_prices, test_prices = train_test_split(prices, train_ratio=args.train_ratio)

    # -- Build environments --
    train_env = TradingEnv(
        train_prices,
        transaction_cost=args.transaction_cost,
        stop_loss=args.stop_loss,
    )
    test_env = TradingEnv(
        test_prices,
        transaction_cost=args.transaction_cost,
        stop_loss=args.stop_loss,
    )

    state_dim = TradingEnv.STATE_DIM

    # -- Build agent --
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=3,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        hidden_dim=args.hidden_dim,
    )

    if args.load:
        agent.load(args.load)

    # -- Train --
    train_rewards, train_equities, train_losses, action_counts = [], [], [], np.zeros(3)

    if not args.eval_only:
        print(f"\nTraining for {args.episodes} episodes on {len(train_prices)} steps...")
        print(f"Device: {agent.device} | State dim: {state_dim} | Hidden: {args.hidden_dim}\n")
        train_rewards, train_equities, train_losses, action_counts = run_training(
            train_env, agent, episodes=args.episodes
        )
        agent.save(args.save)

    # -- Evaluate on test set --
    print("\nEvaluating on held-out test set...")

    print("\n--- TRAIN (last episode) ---")
    train_step_returns = np.array(train_env.step_returns) / train_env.reward_scaling
    from metrics import full_report, print_report
    print_report(full_report(
        np.array(train_env.equity_curve),
        train_step_returns,
        train_env.trade_log,
    ))

    print("--- TEST ---")
    test_metrics = run_evaluation(test_env, agent)
    print_report(test_metrics)

    # -- Plots --
    if not args.no_plot and train_rewards:
        plot_results(
            train_rewards, train_losses, action_counts,
            train_env, test_env, prices,
        )


if __name__ == "__main__":
    main()
