"""Counter-based randomness: every draw is a pure function of (seed, agent, step, purpose, index).

Shared-seed pairing with stateful generators is causally incoherent once an
intervention changes control flow: a market maker that requotes one extra time
consumes one extra draw, and every later draw of that agent shifts, so paired
runs stop sharing noise for reasons that have nothing to do with the
intervention (Buffalo, Pearson & Klein 2026). Per-agent streams only contain
the damage to one agent per event.

``KeyedStream`` removes it. It has no sequential state between calls: ``at``
rekeys a Philox generator from the address of the draw and returns it, so what
an agent receives at (step, purpose) cannot depend on how many draws it made
before or on what the other agents did. Several values drawn from one ``at``
call form a short sequence that is stable in its prefix, so a Poisson count
that differs between two runs only adds or removes orders at the tail.

Purposes are small integers, one per decision an agent draws for; a purpose
drawn more than once in a step (one per speculator) adds an ``index``. Select
this mode with ``SimConfig(rng="counter")``; the default ``"stateful"`` keeps
the plain per-agent ``Generator`` streams.
"""

import zlib

import numpy as np

# Purposes. Their values are part of a run's identity: changing one changes the draws of that purpose.
SCHEDULE = 0
VALUE = 1
MM_SIZE_BID, MM_SIZE_ASK = 2, 3
NOISE_CANCEL_COUNT, NOISE_CANCEL_PICK, NOISE_ARRIVALS, NOISE_SIDE, NOISE_SIZE, NOISE_KIND, NOISE_OFFSET = 4, 5, 6, 7, 8, 9, 10
INST_START, INST_JITTER = 11, 12
SPEC_BIAS, SPEC_MODE, SPEC_LOOKBACK, SPEC_PRIVATE, SPEC_ACTIVE, SPEC_OBS, SPEC_SWITCH = 13, 14, 15, 16, 17, 18, 19
EXEC_RESET, EXEC_DECIDE = 20, 21

CONSTRUCTION = -1  # the step value of draws made while an agent is being built


def label_key(label: str) -> int:
    """Stable integer for a stream label, so an agent's draws do not depend on who else exists."""
    return zlib.crc32(label.encode())


class KeyedStream:
    """All of one agent's randomness, addressed by (step, purpose, index) instead of by position.

    ``at`` returns a ``numpy.random.Generator`` positioned at the start of that
    address. The generator object is reused, so take what a purpose needs before
    asking for the next address. There is deliberately no ``random()`` here:
    an unaddressed draw would bring sequential state back.
    """

    def __init__(self, root: int, label: str):
        self.label = label
        self._bit = np.random.Philox(key=np.array([int(root), label_key(label)], dtype=np.uint64))
        self._gen = np.random.Generator(self._bit)
        state = self._bit.state
        state["buffer_pos"] = len(state["buffer"])  # discard any buffered output
        state["has_uint32"] = 0
        state["uinteger"] = 0
        self._state = state
        self._counter = state["state"]["counter"]

    def at(self, step: int, purpose: int, index: int = 0) -> np.random.Generator:
        counter = self._counter
        # Philox increments counter[0] per block, so the three address words never collide with a sequence.
        counter[0], counter[1], counter[2], counter[3] = 0, step + 1, purpose, index
        self._bit.state = self._state
        return self._gen

    def reseed(self, root: int) -> None:
        self._state["state"]["key"][:] = (int(root), label_key(self.label))


def keyed(rng, step: int, purpose: int, index: int = 0):
    """The generator for this address: ``rng`` itself for a stateful stream, so one code path serves both modes."""
    return rng.at(step, purpose, index) if rng.__class__ is KeyedStream else rng
