"""The learned executor: a small actor-critic network and the ``Strategy`` that runs it (plan section 5).

``PolicyNet`` maps the feature vector of ``features.py`` to action logits and a
state value. ``PolicyStrategy`` wraps it as an ordinary execution ``Strategy``
so the trained policy is evaluated through the same ``ExecutorAgent``, limits
and forced completion as every baseline, and scored by the same experiments.

During training the strategy runs in "record" mode: every decision appends a
``Transition`` (features, action, log-prob, value) to a buffer that the PPO
trainer consumes, while the environment fills in the rewards. During
evaluation it runs greedily (or stochastically) with no buffer.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from order_book.execution.features import N_ACTIONS, N_FEATURES, Transition, decision, features
from order_book.execution.task import Decision, ExecState, Strategy


class PolicyNet(nn.Module):
    """Shared MLP trunk with an actor head (action logits) and a critic head (state value)."""

    def __init__(self, n_features: int = N_FEATURES, n_actions: int = N_ACTIONS, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        self.n_features, self.n_actions, self.hidden = n_features, n_actions, hidden

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits, _ = self.forward(obs)
        return torch.distributions.Categorical(logits=logits)

    # ----- persistence ----------------------------------------------------- #

    def save(self, path) -> None:
        torch.save({"state_dict": self.state_dict(), "n_features": self.n_features,
                    "n_actions": self.n_actions, "hidden": self.hidden}, path)

    @staticmethod
    def load(path, map_location="cpu") -> "PolicyNet":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        net = PolicyNet(blob["n_features"], blob["n_actions"], blob["hidden"])
        net.load_state_dict(blob["state_dict"])
        net.eval()
        return net


class PolicyStrategy(Strategy):
    """Runs a ``PolicyNet`` as an execution strategy. Set ``recorder`` for on-policy training."""

    name = "rl"

    def __init__(self, net: PolicyNet, deterministic: bool = True, device: str = "cpu"):
        self.net = net.to(device)
        self.deterministic = deterministic
        self.device = device
        self.recorder: list[Transition] | None = None  # when set, decisions are logged for PPO

    @classmethod
    def from_file(cls, path, deterministic: bool = True, device: str = "cpu") -> "PolicyStrategy":
        return cls(PolicyNet.load(path, map_location=device), deterministic, device)

    def reset(self, s: ExecState, rng) -> None:
        pass

    def decide(self, s: ExecState, rng) -> Decision:
        obs = features(s)
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(t)
        dist = torch.distributions.Categorical(logits=logits)
        if self.recorder is not None or not self.deterministic:
            # a seeded torch generator keeps sampling reproducible from the executor's own stream
            gen = torch.Generator(device=self.device).manual_seed(int(rng.integers(1 << 31)))
            action = torch.multinomial(dist.probs, 1, generator=gen).item()
        else:
            action = int(torch.argmax(logits, dim=-1).item())
        if self.recorder is not None:
            self.recorder.append(Transition(obs, int(action), float(dist.log_prob(torch.tensor([action]))),
                                            float(value)))
        return decision(action, s)


def new_policy(device: str = "cpu", seed: int | None = None) -> PolicyNet:
    """A freshly initialised policy; ``seed`` makes the initial weights reproducible."""
    if seed is not None:
        torch.manual_seed(seed)
    return PolicyNet().to(device)
