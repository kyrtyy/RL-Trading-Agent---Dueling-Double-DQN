"""
Dueling Double DQN with Prioritized Experience Replay (PER).

Three upgrades over vanilla DQN, each addressing a known failure mode:

  1. Double DQN — vanilla DQN overestimates Q-values because it uses
     the same network to both select and evaluate actions. Double DQN
     selects actions with the policy net but evaluates them with the
     target net, which cuts this bias significantly.

  2. Dueling architecture — splits Q(s,a) into V(s) + A(s,a).
     The value stream learns how good a state is in general;
     the advantage stream learns the relative value of each action.
     This is especially useful when many actions lead to similar outcomes
     (e.g., holding vs. doing nothing in a flat market).

  3. Prioritized Experience Replay — instead of sampling transitions
     uniformly, PER samples high-TD-error transitions more often.
     The agent spends more time learning from its mistakes.
     Importance-sampling weights correct for the resulting bias.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# -----------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------

class DuelingDQN(nn.Module):
    """
    Dueling architecture:
      shared encoder → [value stream, advantage stream]
      Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        value = self.value_stream(enc)
        advantage = self.advantage_stream(enc)
        # Combine — subtract mean advantage for identifiability
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


# -----------------------------------------------------------------------
# Prioritized Replay Buffer
# -----------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Stores transitions with a priority score (TD error).
    Higher priority = sampled more often.

    Uses a simple heap-free approach: store priorities in a numpy
    array and sample proportionally. Fast enough for buffers up to ~50k.
    """

    def __init__(self, capacity: int = 20000, alpha: float = 0.6, beta_start: float = 0.4, beta_steps: int = 50000):
        self.capacity = capacity
        self.alpha = alpha          # how much prioritization to use (0 = uniform)
        self.beta = beta_start      # importance-sampling correction (annealed to 1)
        self.beta_increment = (1.0 - beta_start) / beta_steps

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities[:len(self)].max() if len(self) > 0 else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.pos == 0 if self.full else self.full or self.pos == 0

    def sample(self, batch_size: int):
        n = len(self)
        priorities = self.priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        state, action, reward, next_state, done = map(np.array, zip(*samples))
        return state, action, reward, next_state, done, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6  # small constant to avoid zero priority

    def __len__(self):
        return len(self.buffer)


# -----------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------

class DDQNAgent:
    """
    Dueling Double DQN agent.

    Hyperparameter notes:
      - gamma=0.99 works well for episodic tasks with dense rewards
      - epsilon decays over episodes, not steps, so it's more predictable
      - target net syncs every `target_update` gradient steps (not episodes)
      - gradient clipping at 1.0 prevents exploding updates early in training
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        buffer_size: int = 20000,
        batch_size: int = 128,
        target_update: int = 200,
        hidden_dim: int = 128,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)

        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_t)
        return q.argmax().item()

    def push(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        state, action, reward, next_state, done, indices, weights = \
            self.replay_buffer.sample(self.batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        action     = torch.LongTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done       = torch.FloatTensor(done).to(self.device)
        weights    = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        q_values = self.policy_net(state)
        q_value  = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Double DQN target: select action with policy net, evaluate with target net
        with torch.no_grad():
            next_actions  = self.policy_net(next_state).argmax(1)
            next_q_values = self.target_net(next_state)
            next_q        = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q      = reward + self.gamma * next_q * (1 - done)

        td_errors = (q_value - target_q).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Weighted Huber loss (less sensitive to outliers than MSE)
        loss = (weights * nn.HuberLoss(reduction="none")(q_value, target_q)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.update_steps += 1
        if self.update_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item()

    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_steps": self.update_steps,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.update_steps = checkpoint["update_steps"]
        print(f"Model loaded from {path}")
