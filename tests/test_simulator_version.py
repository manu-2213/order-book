"""The simulator's behaviour, pinned per version.

A change that alters what a run produces fails here. That is the point: if the change is intended, bump
``SIMULATOR_VERSION`` in ``order_book/provenance.py`` and add the new fingerprint below, keeping the old ones as
the record. Papers cite the version their results ran on.

The hashes were recorded on Apple Silicon with NumPy 2.4.6. A different platform or NumPy version may differ in
the last bit of a float and so flip a decision, which is why the cluster's smoke test runs this file on its own:
a failure there, alone, means results are not byte-comparable across machines, not that the code is broken.
"""

import hashlib
from dataclasses import replace

import pytest

from order_book.config import SimConfig
from order_book.counterfactual import ARMS, FULL, GHOST, PairedSpec, run_paired
from order_book.experiments import EpisodeSpec, run_episode
from order_book.presets import adversary, private_valuations
from order_book.provenance import SIMULATOR_VERSION
from order_book.simulator import Simulator

FINGERPRINTS = {
    "1": {
        "stateful default": "9fb1cce3d37bfce9",
        "stateful A4": "7672eb495b086f83",
        "stateful private": "06e37c8d2d9b56fb",
        "counter default": "f1929200f78e00b1",
        "episode shortfall": 1.7150859944124728,
        "episode path": "bd3be24998712978",
        "paired ghost": 0.5000250012500624,
        "paired IP": 0.12222833363894459,
        "paired full": 0.05555833347142069,
    },
}


def tape_hash(sim: Simulator) -> str:
    blob = repr((sim.market.tape, sim.book.order_id, sim.book.trade_id, sim.market.public.mid))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@pytest.fixture
def expected() -> dict:
    assert SIMULATOR_VERSION in FINGERPRINTS, f"record the fingerprint of simulator version {SIMULATOR_VERSION} here"
    return FINGERPRINTS[SIMULATOR_VERSION]


@pytest.mark.parametrize("name, cfg", [
    ("stateful default", None), ("stateful A4", adversary("A4")), ("stateful private", private_valuations()),
    ("counter default", replace(SimConfig(), rng="counter")),
])
def test_a_seeded_run_is_the_one_this_version_recorded(name, cfg, expected):
    sim = Simulator(seed=7, config=cfg)
    sim.run(1500)
    assert tape_hash(sim) == expected[name]


def test_an_execution_episode_is_the_one_this_version_recorded(expected):
    episode = run_episode(EpisodeSpec(1000, "twap_market", 0.1, "A1-medium", "visible", horizon=300, burn_in=500,
                                      after=50, record_path=True))
    assert episode.shortfall_bps == expected["episode shortfall"]
    assert hashlib.sha256(repr(episode.mid_path).encode()).hexdigest()[:16] == expected["episode path"]


@pytest.mark.parametrize("name, channels", [("paired ghost", GHOST), ("paired IP", ARMS[3]), ("paired full", FULL)])
def test_a_paired_run_is_the_one_this_version_recorded(name, channels, expected):
    """Pins the ghost's matching rule and the lockstep, which the seeded runs above do not reach."""
    result = run_paired(PairedSpec(1000, "twap_passive", 0.3, channels, horizon=300, burn_in=600, after=60))
    assert result.shortfall_bps == expected[name]
