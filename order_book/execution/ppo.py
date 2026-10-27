"""PPO training for the learned executor (plan section 5.3).

On-policy PPO over whole execution episodes. Each iteration rolls out a batch
of episodes with the current policy (recording per-decision transitions and
telescoping rewards from ``ExecutionEnv``), computes GAE advantages, and takes
several epochs of clipped-objective minibatch updates. One verbose line per
iteration reports the mean implementation shortfall, completion, return,
entropy and the loss terms so training can be followed across epochs.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from order_book.execution.env import ExecutionEnv
from order_book.execution.policy import PolicyNet, PolicyStrategy, new_policy


@dataclass
class PPOConfig:
    iterations: int = 200
    episodes_per_update: int = 16        # whole episodes rolled out per iteration
    epochs: int = 4                      # optimisation passes over the collected batch
    minibatch: int = 512
    gamma: float = 0.999                 # near 1: execution cost is barely discounted over a horizon
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    seed: int = 0
    train_seeds: tuple[int, ...] = tuple(range(0, 1000))   # episode seeds are drawn from the training pool


def _gae(rewards, values, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns for one terminal episode (bootstrap value after the end is 0)."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


class PPOTrainer:
    def __init__(self, env: ExecutionEnv | None = None, config: PPOConfig | None = None,
                 net: PolicyNet | None = None, device: str = "cpu"):
        self.env = env or ExecutionEnv()
        self.cfg = config or PPOConfig()
        self.device = device
        self.net = (net or new_policy(device, self.cfg.seed)).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.history: list[dict] = []

    # ----- data collection ------------------------------------------------- #

    def collect(self) -> tuple[dict, dict]:
        """Roll out a batch of episodes; returns stacked tensors and episode statistics."""
        strategy = PolicyStrategy(self.net, deterministic=False, device=self.device)
        obs, acts, logp, adv, ret, vals = [], [], [], [], [], []
        shortfalls, completions, rets = [], [], []
        for _ in range(self.cfg.episodes_per_update):
            seed = int(self.rng.choice(self.cfg.train_seeds))
            transitions, info = self.env.rollout(strategy, seed)
            if not transitions:
                continue
            rewards = [t.reward for t in transitions]
            values = [t.value for t in transitions]
            a, r = _gae(rewards, values, self.cfg.gamma, self.cfg.gae_lambda)
            obs.extend(t.obs for t in transitions)
            acts.extend(t.action for t in transitions)
            logp.extend(t.log_prob for t in transitions)
            vals.extend(values)
            adv.extend(a.tolist())
            ret.extend(r.tolist())
            if info.shortfall_bps is not None:
                shortfalls.append(info.shortfall_bps)
            completions.append(info.completion)
            rets.append(info.ret)
        batch = {
            "obs": torch.as_tensor(np.array(obs), dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(acts, dtype=torch.long, device=self.device),
            "log_probs": torch.as_tensor(logp, dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(adv, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(ret, dtype=torch.float32, device=self.device),
            "values": torch.as_tensor(vals, dtype=torch.float32, device=self.device),
        }
        stats = {
            "shortfall": float(np.mean(shortfalls)) if shortfalls else float("nan"),
            "completion": float(np.mean(completions)) if completions else float("nan"),
            "return": float(np.mean(rets)) if rets else float("nan"),
            "transitions": len(obs),
        }
        return batch, stats

    # ----- optimisation ---------------------------------------------------- #

    def update(self, batch: dict) -> dict:
        n = batch["obs"].shape[0]
        adv = batch["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.arange(n)
        p_loss = v_loss = ent = kl = 0.0
        passes = 0
        for _ in range(self.cfg.epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch):
                mb = idx[start:start + self.cfg.minibatch]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                logits, values = self.net(batch["obs"][mb_t])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(batch["actions"][mb_t])
                ratio = torch.exp(new_logp - batch["log_probs"][mb_t])
                a = adv[mb_t]
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * a
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = ((values - batch["returns"][mb_t]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    p_loss += float(policy_loss); v_loss += float(value_loss); ent += float(entropy)
                    kl += float((batch["log_probs"][mb_t] - new_logp).mean())
                passes += 1
        passes = max(passes, 1)
        return {"policy_loss": p_loss / passes, "value_loss": v_loss / passes,
                "entropy": ent / passes, "kl": kl / passes}

    # ----- loop ------------------------------------------------------------ #

    def train(self, verbose: bool = True) -> PolicyNet:
        t0 = time.perf_counter()
        for it in range(1, self.cfg.iterations + 1):
            batch, stats = self.collect()
            losses = self.update(batch) if stats["transitions"] else {}
            row = {"iteration": it, **stats, **losses}
            self.history.append(row)
            if verbose:
                elapsed = time.perf_counter() - t0
                print(f"[ppo] it {it:4d}/{self.cfg.iterations}  IS {stats['shortfall']:+6.2f}bps  "
                      f"compl {stats['completion']:.3f}  ret {stats['return']:+6.2f}  "
                      f"ent {losses.get('entropy', float('nan')):.3f}  "
                      f"ploss {losses.get('policy_loss', float('nan')):+.4f}  "
                      f"vloss {losses.get('value_loss', float('nan')):.3f}  "
                      f"kl {losses.get('kl', float('nan')):+.4f}  ({elapsed:.0f}s)", flush=True)
        return self.net
