"""Training the executor inside the paired worlds: a reward measured against the reference, parallel rollouts, and
a resumable PPO loop that selects on seeds it never tests on.

Why the paired worlds. A policy difference worth having is a few tenths of a basis point, and the shortfall of one
episode moves by several basis points with the market's own drift. The reference world steps in lockstep and never
reacts to the executor, so its mid path is a control variate: the reward is the shortfall minus what a uniform schedule
would have paid in drift along Arm 0's mid, which the policy cannot influence. This leaves the objective exactly the
shortfall and removes most of the noise (the episode's standard deviation falls from about 7.3 to about 2.7 bps). The
first attempt charged each fill against Arm 0's mid at that step instead, which removes the drift entirely, but also
stops charging the drift an adaptive schedule pays, and the policies it produced saved on the spread and lost it on
timing; that reward stays as the ``"net"`` ablation. The maker-facing flow detector is live in the treated world too,
so being detected can be charged per step and a policy trained to stay under it.

Two methods. PPO with GAE over the per-decision rewards, and GRPO: the same clipped objective with no critic, where a
group of episodes shares one seed, size and market and differs only in the policy's sampling, and each episode's
advantage is its return minus the group's mean. The group shares Arm 0's path, so drift cancels inside it exactly.

Seeds. Episodes draw their seed from the training pool (below 800) and their size and market from the lists given,
so one policy is trained across many environments. The checkpoint is chosen greedily on the validation seeds (800-879),
paired with TWAP on the same episodes, and seeds from 1000 up are never touched here: they are where the decomposition
scores what was learned.

One episode is a deterministic function of the weights, the seed, the size, the market and a sample index, because the
policy samples from the executor's own keyed stream. Rollouts therefore parallelise over processes without changing
anything, and a run resumes from its checkpoint to the same weights it would have reached in one go.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from order_book.config import SimConfig
from order_book.counterfactual import FULL, Channels, PairedRun, PairedSpec, burned_in
from order_book.execution import ExecutionTask, PolicyNet, PolicyStrategy, Transition, make_strategy
from order_book.execution.ppo import PPOConfig, PPOTrainer, _gae
from order_book.experiments import CAPITAL_REGIMES
from order_book.scoring import paired_difference

TRAIN_SEEDS = tuple(range(0, 800))
VALIDATION_SEEDS = tuple(range(800, 880))
FIRST_TEST_SEED = 1000


@dataclass(frozen=True)
class RewardSpec:
    """What the executor is rewarded for. Every term is in basis points of the arrival mid, per unit of the order."""

    benchmark: str = "shortfall"      # "shortfall": minus a uniform schedule's drift along Arm 0's mid; "net": each fill against Arm 0's mid
    detection_penalty: float = 0.0    # bps per decision interval spent at full detector signal in the executor's direction
    impact_penalty: float = 0.0       # extra weight on the displacement of the treated mid from Arm 0's at each fill
    incomplete_penalty: float = 5.0   # bps per unfilled unit-fraction, charged on the last decision

    def __post_init__(self):
        if self.benchmark not in ("shortfall", "net"):
            raise ValueError(f"benchmark must be 'shortfall' or 'net', not {self.benchmark!r}")


@dataclass(frozen=True)
class PairedExecutionEnv:
    """Rolls out whole episodes in ``PairedRun``: the executor trades in the treated world while Arm 0 steps alongside.

    ``configs`` are (label, base market) pairs; an episode names one by index. The market is built exactly as the
    decomposition builds it (``PairedSpec.config``), so a policy trains in the world it is later decomposed in.
    """

    configs: tuple = tuple((r.name, r.base()) for r in CAPITAL_REGIMES if r.name == "low_cover_capital10")
    sizes: tuple = (0.1, 0.2, 0.4)
    reward: RewardSpec = field(default_factory=RewardSpec)
    channels: Channels = FULL
    horizon: int = 2_000
    decision_interval: int = 5
    burn_in: int = 2_000
    threshold: str = "medium"

    def spec(self, seed: int, size: float, config_index: int = 0, side: str | None = None, strategy: str = "rl") -> PairedSpec:
        """The decomposition's description of this episode; with a baseline's name, ``run_paired`` reproduces it."""
        label, base = self.configs[config_index]
        return PairedSpec(seed, strategy, size, self.channels, side=side, horizon=self.horizon, decision_interval=self.decision_interval,
                          burn_in=self.burn_in, after=ExecutionTask().grace, threshold=self.threshold, base_config=base, regime=label)

    def rollout(self, strategy, seed: int, size: float, config_index: int = 0, side: str | None = None) -> tuple[list[Transition], "EpisodeRecord"]:
        """One episode. Transitions are empty unless ``strategy`` is a ``PolicyStrategy``, which records its decisions."""
        spec, rw = self.spec(seed, size, config_index, side), self.reward
        if isinstance(strategy, PolicyStrategy):
            strategy.recorder = []
        blob, volume = burned_in(spec.seed, spec.config(), spec.burn_in)
        run = PairedRun(blob, spec.channels, spec.ghost)
        task = ExecutionTask(side=spec.resolved_side, size_fraction=size, horizon=spec.horizon, decision_interval=spec.decision_interval)
        executor = run.add_executor(task, strategy, volume)
        public, reference = run.market.public, run.reference.market.public
        sign = 1 if task.side == "bid" else -1
        transitions = strategy.recorder if isinstance(strategy, PolicyStrategy) else []
        window = spec.window  # horizon + grace, as run_paired steps it
        arm0_mid, treated_mid = np.empty(window + 1), np.empty(window + 1)
        start, quantity, arrival = run.market.step, executor.quantity, None
        cost_is = cost_net = control = impact_charge = detection_charge = exposure = 0.0
        seen, active = 0, 0
        for k in range(window):
            arm0_mid[k], treated_mid[k] = reference.mid, public.mid  # the mids a fill in this step is measured against
            run.step()
            if executor.result is None:
                continue
            arrival = executor.result.arrival_mid
            fills = executor.result.fills
            while seen < len(fills):
                step, price, qty, _ = fills[seen]
                i = step - start
                scale = qty / quantity / arrival * 1e4
                is_cost = sign * (price - arrival) * scale
                net_cost = sign * (price - arm0_mid[i]) * scale
                displacement = sign * (treated_mid[i] - arm0_mid[i]) * scale
                cost_is, cost_net = cost_is + is_cost, cost_net + net_cost
                charge = (is_cost if rw.benchmark == "shortfall" else net_cost) + rw.impact_penalty * displacement
                impact_charge += rw.impact_penalty * displacement
                if transitions:
                    transitions[-1].reward -= charge
                seen += 1
            charge = 0.0
            if rw.benchmark == "shortfall" and k < spec.horizon:
                # credit back what a uniform schedule pays in drift at this step: policy-independent, so it is a control variate
                credit = sign * (arm0_mid[k] - arrival) / arrival * 1e4 / spec.horizon
                control += credit
                charge -= credit
            if not executor.done:
                seen_signal = max(0.0, sign * run.flow.signal)
                exposure += seen_signal
                active += 1
                detection = rw.detection_penalty * seen_signal / spec.decision_interval
                detection_charge += detection
                charge += detection
            if transitions:
                transitions[-1].reward -= charge
            if executor.done and (k >= spec.horizon or rw.benchmark != "shortfall"):
                break
        if not executor.done:
            executor._finish()
        r = executor.result
        incomplete_charge = rw.incomplete_penalty * (r.incomplete / quantity if r.incomplete else 0.0)
        if transitions:
            transitions[-1].reward -= incomplete_charge
        record = EpisodeRecord(seed, size, self.configs[config_index][0], task.side, quantity, r.executed, r.forced, r.incomplete, r.completion,
                               r.aggressive_executed / r.executed if r.executed else 0.0, r.shortfall_bps, cost_is, cost_net, cost_is - cost_net, control,
                               exposure / max(active, 1), detection_charge, impact_charge, incomplete_charge, sum(t.reward for t in transitions), len(transitions))
        return transitions, record


@dataclass
class EpisodeRecord:
    """What one episode came to, for the trainer, the validation and the tests."""

    seed: int
    size: float
    regime: str
    side: str
    quantity: int
    executed: int
    forced: int
    incomplete: int
    completion: float
    aggressive_share: float       # of what was executed
    shortfall_bps: float | None   # the executor's own figure, per executed lot
    cost_is: float                # shortfall per unit of the order
    cost_net: float               # the same against Arm 0's mid at each fill: cost net of natural drift
    drift: float                  # cost_is - cost_net
    control: float                # a uniform schedule's drift along Arm 0's mid, credited back under the "shortfall" reward
    exposure: float               # mean detector signal in the executor's direction over its active steps
    detection_charge: float
    impact_charge: float
    incomplete_charge: float
    ret: float                    # sum of rewards
    decisions: int


# ----- rollouts in parallel ------------------------------------------------------------------- #

_WORKER: dict = {}


def _init_worker(env: PairedExecutionEnv) -> None:
    import torch
    torch.set_num_threads(1)  # workers are the unit of parallelism
    _WORKER["env"] = env
    _WORKER["nets"] = {}


def _rollout_job(job: tuple) -> tuple[list[Transition], EpisodeRecord]:
    """One episode in a worker. ``weights`` is a policy's state dict, or a baseline's name (greedy, no transitions);
    ``sample`` picks the policy's sampling stream, so that one seed can give a group of different episodes."""
    weights, greedy, seed, size, config_index, sample = job
    env = _WORKER["env"]
    if isinstance(weights, str):
        return env.rollout(make_strategy(weights), seed, size, config_index)
    shape = (weights["trunk.0.weight"].shape[1], weights["actor.weight"].shape[0], weights["trunk.0.weight"].shape[0])  # features, actions, hidden
    if shape not in _WORKER["nets"]:  # a policy saved before a feature or an action was appended keeps its own shape
        _WORKER["nets"][shape] = PolicyNet(*shape)
    net = _WORKER["nets"][shape]
    net.load_state_dict(weights)
    return env.rollout(PolicyStrategy(net, deterministic=greedy, sample=sample), seed, size, config_index)


class Rollouts:
    """Runs episodes of one environment in a pool of processes, or in this one when ``workers`` is 1."""

    def __init__(self, env: PairedExecutionEnv, workers: int = 1):
        self.env, self.workers = env, workers
        self.pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(env,)) if workers > 1 else None
        if self.pool is None:
            _init_worker(env)

    def run(self, jobs: list[tuple]) -> list[tuple[list[Transition], EpisodeRecord]]:
        if self.pool is None:
            return [_rollout_job(job) for job in jobs]
        return list(self.pool.map(_rollout_job, jobs, chunksize=1))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()


def cpu_state_dict(net: PolicyNet) -> dict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


# ----- the trainer ---------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrainingConfig:
    """Everything a training run is determined by, besides the environment."""

    ppo: PPOConfig = field(default_factory=lambda: PPOConfig(iterations=1_000, episodes_per_update=64, train_seeds=TRAIN_SEEDS))
    method: str = "ppo"               # "ppo": GAE advantages and a critic; "grpo": group-relative returns, no critic
    group_size: int = 4               # grpo: episodes sharing one seed, size and market; episodes_per_update must be a multiple
    validation_seeds: tuple = VALIDATION_SEEDS
    eval_every: int = 25
    anneal: bool = True               # learning rate and entropy bonus fall linearly to zero over the run
    checkpoint_every: int = 5

    def __post_init__(self):
        seeds = set(self.ppo.train_seeds) | set(self.validation_seeds)
        if max(seeds) >= FIRST_TEST_SEED:
            raise ValueError(f"training and validation seeds must stay below {FIRST_TEST_SEED}: those are the test seeds")
        if set(self.ppo.train_seeds) & set(self.validation_seeds):
            raise ValueError("a seed cannot be both trained on and selected on")
        if self.method not in ("ppo", "grpo"):
            raise ValueError(f"method must be 'ppo' or 'grpo', not {self.method!r}")
        if self.method == "grpo" and (self.group_size < 2 or self.ppo.episodes_per_update % self.group_size):
            raise ValueError("grpo needs groups of at least 2 that divide the episodes per update")


class PairedTrainer(PPOTrainer):
    """PPO or GRPO over whole episodes of ``PairedExecutionEnv``, rolled out in parallel, validated against TWAP on the same
    episodes, and checkpointed so that a run stopped at a walltime continues to the same result."""

    def __init__(self, env: PairedExecutionEnv, training: TrainingConfig | None = None, workers: int = 1, out: Path | None = None):
        import torch
        torch.set_num_threads(1)  # the update is small; one thread keeps it identical on every machine
        self.training = training or TrainingConfig()
        super().__init__(env, replace(self.training.ppo))  # a copy: the loop anneals the entropy bonus in place
        if self.training.method == "grpo":
            self.cfg.value_coef = 0.0       # no critic: the group's mean return is the baseline
            self.centre_advantages = False  # already centred per group; a batch mean over transitions would weight episodes by length
        self.out = Path(out) if out is not None else None
        self.rollouts = Rollouts(env, workers)
        self.iteration = 0
        self._resumed_at = 0
        self.evals: list[dict] = []
        self.best: dict | None = None
        self.twap_validation: list[EpisodeRecord] | None = None
        self._base_lr, self._base_entropy = self.cfg.lr, self.cfg.entropy_coef

    # ----- episodes ---------------------------------------------------------------------- #

    def episodes(self) -> list[tuple]:
        """The (seed, size, market, sample) of this iteration's episodes, drawn from the training pool; under grpo, in groups
        that share everything but the sample."""
        cfg, env, tr = self.cfg, self.env, self.training
        group = tr.group_size if tr.method == "grpo" else 1
        draws = [(int(self.rng.choice(cfg.train_seeds)), float(env.sizes[int(self.rng.integers(len(env.sizes)))]),
                  int(self.rng.integers(len(env.configs)))) for _ in range(cfg.episodes_per_update // group)]
        return [(*draw, sample) for draw in draws for sample in range(group)]

    def collect(self) -> tuple[dict, dict]:
        import torch
        weights = cpu_state_dict(self.net)
        results = self.rollouts.run([(weights, False, *episode) for episode in self.episodes()])
        if self.training.method == "grpo":
            group = self.training.group_size
            returns = np.array([record.ret for _, record in results])
            baseline = returns.reshape(-1, group).mean(axis=1).repeat(group)  # the group's mean: drift cancels inside it exactly
        obs, acts, logp, adv, ret = [], [], [], [], []
        records = []
        for n, (transitions, record) in enumerate(results):
            if not transitions:
                continue
            if self.training.method == "grpo":
                a = np.full(len(transitions), record.ret - baseline[n], dtype=np.float32)
                r = a
            else:
                a, r = _gae([t.reward for t in transitions], [t.value for t in transitions], self.cfg.gamma, self.cfg.gae_lambda)
            obs.extend(t.obs for t in transitions)
            acts.extend(t.action for t in transitions)
            logp.extend(t.log_prob for t in transitions)
            adv.extend(a.tolist())
            ret.extend(r.tolist())
            records.append(record)
        batch = {"obs": torch.as_tensor(np.array(obs), dtype=torch.float32), "actions": torch.as_tensor(acts, dtype=torch.long),
                 "log_probs": torch.as_tensor(logp, dtype=torch.float32), "advantages": torch.as_tensor(adv, dtype=torch.float32),
                 "returns": torch.as_tensor(ret, dtype=torch.float32)}
        mean = lambda key: float(np.mean([getattr(x, key) for x in records])) if records else float("nan")
        stats = {"cost_is": mean("cost_is"), "cost_net": mean("cost_net"), "exposure": mean("exposure"), "completion": mean("completion"),
                 "return": mean("ret"), "transitions": len(obs)}
        return batch, stats

    # ----- validation -------------------------------------------------------------------- #

    def validation_jobs(self, weights) -> list[tuple]:
        return [(weights, True, seed, size, ci, 0) for ci in range(len(self.env.configs)) for size in self.env.sizes
                for seed in self.training.validation_seeds]

    def validate(self) -> dict:
        """The greedy policy against TWAP on the validation episodes, paired. ``gap_is`` is the shortfall gap, the criterion."""
        if self.twap_validation is None:
            self.twap_validation = [r for _, r in self.rollouts.run(self.validation_jobs("twap_market"))]
        policy = [r for _, r in self.rollouts.run(self.validation_jobs(cpu_state_dict(self.net)))]
        return {"iteration": self.iteration, **compare(policy, self.twap_validation)}

    # ----- checkpoints ------------------------------------------------------------------- #

    def save_checkpoint(self) -> None:
        import torch
        if self.out is None:
            return
        self.out.mkdir(parents=True, exist_ok=True)
        torch.save({"net": cpu_state_dict(self.net), "opt": self.opt.state_dict(), "iteration": self.iteration, "np_rng": self.rng.bit_generator.state,
                    "history": self.history, "evals": self.evals, "best": self.best,
                    "twap_validation": [asdict(r) for r in self.twap_validation] if self.twap_validation is not None else None},
                   self.out / "checkpoint.pt.tmp")
        os.replace(self.out / "checkpoint.pt.tmp", self.out / "checkpoint.pt")  # never leave a half-written checkpoint for the next link

    def load_checkpoint(self) -> bool:
        import torch
        path = self.out / "checkpoint.pt" if self.out is not None else None
        if path is None or not path.exists():
            return False
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(blob["net"])
        self.opt.load_state_dict(blob["opt"])
        self.iteration, self.history, self.evals, self.best = blob["iteration"], blob["history"], blob["evals"], blob["best"]
        self.rng.bit_generator.state = blob["np_rng"]  # torch's global generator is never used: sampling is keyed per decision
        if blob["twap_validation"] is not None:
            self.twap_validation = [EpisodeRecord(**r) for r in blob["twap_validation"]]
        return True

    # ----- the loop ---------------------------------------------------------------------- #

    def anneal(self) -> None:
        if not self.training.anneal:
            return
        frac = max(0.0, 1.0 - self.iteration / self.cfg.iterations)
        for group in self.opt.param_groups:
            group["lr"] = self._base_lr * frac
        self.cfg.entropy_coef = self._base_entropy * frac

    def train(self, hours: float | None = None, verbose: bool = True) -> str:
        """Runs to ``iterations`` and returns "done", or stops before ``hours`` of wall time with a checkpoint and returns "paused"."""
        t0, cfg, tr = time.perf_counter(), self.cfg, self.training
        longest = 0.0  # of this link's iterations; one with a validation in it takes several times an ordinary one
        while self.iteration < cfg.iterations:
            elapsed = time.perf_counter() - t0
            if hours is not None and self.iteration > 0 and elapsed + 2 * longest > hours * 3600:
                self.save_checkpoint()
                return "paused"
            started = time.perf_counter()
            self.anneal()
            batch, stats = self.collect()
            losses = self.update(batch) if stats["transitions"] else {}
            self.iteration += 1
            row = {"iteration": self.iteration, **stats, **losses, "lr": self.opt.param_groups[0]["lr"]}
            if self.iteration % tr.eval_every == 0 or self.iteration == cfg.iterations:
                ev = self.validate()
                self.evals.append(ev)
                row.update({f"val_{k}": v for k, v in ev.items() if k != "iteration"})
                if self.best is None or ev["gap_is"] < self.best["gap_is"]:
                    self.best = ev
                    if self.out is not None:
                        self.out.mkdir(parents=True, exist_ok=True)
                        self.net.save(self.out / "best_policy.pt")
            self.history.append(row)
            longest = max(longest, time.perf_counter() - started)
            if verbose:
                print(f"[{tr.method} {self.iteration:4d}/{cfg.iterations}] IS {stats['cost_is']:+6.2f}  net {stats['cost_net']:+6.2f}  "
                      f"expo {stats['exposure']:.3f}  compl {stats['completion']:.3f}  ent {losses.get('entropy', float('nan')):.3f}  "
                      f"kl {losses.get('kl', float('nan')):+.4f}" + (f"  | val gap IS {self.evals[-1]['gap_is']:+.2f} "
                      f"[{self.evals[-1]['gap_is_low']:+.2f},{self.evals[-1]['gap_is_high']:+.2f}] net {self.evals[-1]['gap_net']:+.2f}  best {self.best['gap_is']:+.2f}@{self.best['iteration']}"
                      if row.get("val_gap_net") is not None else "") + f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
            if self.iteration % tr.checkpoint_every == 0:
                self.save_checkpoint()
        self.save_checkpoint()
        if self.out is not None:
            self.net.save(self.out / "policy.pt")
        return "done"

    def resume(self) -> bool:
        return self.load_checkpoint()

    def close(self) -> None:
        self.rollouts.close()


def compare(policy: list[EpisodeRecord], twap: list[EpisodeRecord]) -> dict:
    """A policy against TWAP on the same episodes, paired: the shortfall and net-of-drift gaps with intervals, and what each did."""
    is_ = paired_difference([r.cost_is for r in policy], [r.cost_is for r in twap])
    net = paired_difference([r.cost_net for r in policy], [r.cost_net for r in twap])
    mean = lambda records, key: float(np.mean([getattr(r, key) for r in records]))
    return {"gap_is": is_["mean"], "gap_is_low": is_["low"], "gap_is_high": is_["high"],
            "gap_net": net["mean"], "gap_net_low": net["low"], "gap_net_high": net["high"],
            "cost_is": mean(policy, "cost_is"), "twap_cost_is": mean(twap, "cost_is"), "cost_net": mean(policy, "cost_net"), "twap_cost_net": mean(twap, "cost_net"),
            "exposure": mean(policy, "exposure"), "twap_exposure": mean(twap, "exposure"), "completion": mean(policy, "completion"),
            "forced_share": float(np.mean([r.forced / r.quantity for r in policy])), "passive_share": 1 - mean(policy, "aggressive_share")}


def with_reward(env: PairedExecutionEnv, **changes) -> PairedExecutionEnv:
    return replace(env, reward=replace(env.reward, **changes))


def config_summary(env: PairedExecutionEnv, training: TrainingConfig) -> dict:
    """JSON-serialisable record of a run's settings, for manifests and histories."""
    return {"regimes": [label for label, _ in env.configs], "sizes": list(env.sizes), "reward": asdict(env.reward), "channels": env.channels.key,
            "horizon": env.horizon, "decision_interval": env.decision_interval, "burn_in": env.burn_in, "threshold": env.threshold,
            "method": training.method, "group_size": training.group_size if training.method == "grpo" else None,
            "ppo": asdict(training.ppo) | {"train_seeds": [min(training.ppo.train_seeds), max(training.ppo.train_seeds)]},
            "validation_seeds": [min(training.validation_seeds), max(training.validation_seeds)], "eval_every": training.eval_every,
            "anneal": training.anneal}
