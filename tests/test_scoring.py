"""Detector scoring and the statistical protocol (plan sections 6 and 10)."""

import numpy as np
import pytest

from order_book.scoring import (
    activity_labels, bootstrap_ci, cvar, detection_delay, false_alarm_rate, holm, paired_difference, roc_auc,
)


def test_roc_auc_perfect_random_ties_and_degenerate():
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5   # all tied -> chance
    assert roc_auc([0.1, 0.5, 0.5, 0.9], [0, 0, 1, 1]) == 0.875
    rng = np.random.default_rng(0)
    assert abs(roc_auc(rng.random(5000), rng.integers(0, 2, 5000)) - 0.5) < 0.03
    assert np.isnan(roc_auc([0.1, 0.2], [1, 1]))


def test_detection_delay_and_false_alarms():
    signal = np.array([0.0, 0.1, 0.2, 0.6, 0.7, 0.1, -0.8, 0.0])
    assert detection_delay(signal, 1, 6, "bid", 0.5) == 2
    assert detection_delay(signal, 1, 6, "ask", 0.5) is None
    assert detection_delay(signal, 5, 8, "ask", 0.5) == 1
    assert false_alarm_rate(signal, np.array([1, 1, 1, 1, 0, 0, 1, 1], bool), 0.5) == pytest.approx(2 / 6)
    assert np.isnan(false_alarm_rate(signal, np.zeros(8, bool), 0.5))


def test_activity_labels_net_side_and_offset():
    y = activity_labels(10, [(2, 6, "bid"), (4, 8, "ask"), (8, 12, "ask")], offset=0)
    assert y.tolist() == [0, 0, 1, 1, 0, 0, -1, -1, -1, -1]
    assert activity_labels(4, [(12, 14, "bid")], offset=10).tolist() == [0, 0, 1, 1]


def test_bootstrap_ci_and_paired_difference():
    est, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=1)
    assert est == 3.0 and lo <= 3.0 <= hi and lo >= 1.0 and hi <= 5.0
    d = paired_difference([2.0, 2.5, 3.0, 2.2], [1.0, 1.5, 2.0, 1.2])
    assert d["mean"] == pytest.approx(1.0) and d["p_greater"] == 0.0 and d["n"] == 4 and d["low"] > 0
    d = paired_difference([1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0])
    assert 0.2 < d["p_greater"] < 0.8
    d = paired_difference([float("nan"), 1.0], [0.0, 0.0])
    assert d["n"] == 1
    assert np.isnan(bootstrap_ci([])[0])


def test_cvar_and_holm():
    assert cvar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 10.0
    assert cvar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.8) == 9.5
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])  # step-down, monotone
    assert holm([0.5]) == [0.5]
