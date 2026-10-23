"""Common interface for metaorder detectors.

A detector reads only ``PublicState`` (book and anonymous flow statistics),
never any agent's account, orders or identity. The simulator calls
``update`` once per step after the public state is refreshed; agents that
react to detection (market makers, order anticipators) read ``signal`` or
the per-side probabilities.
"""

from order_book.public_state import PublicState


class Detector:
    """``signal`` in [-1, 1] (positive: buy metaorder suspected); ``p_buy``/``p_sell`` in [0, 1]."""

    signal: float = 0.0
    p_buy: float = 0.0
    p_sell: float = 0.0

    def update(self, public: PublicState) -> float:
        raise NotImplementedError
