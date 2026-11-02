"""Detector scoring and the statistical protocol of the plan (sections 6 and 10).

Detector scoring, on ground-truth labels:
    roc_auc            rank-based AUC of a score at telling positive from negative steps
    detection_delay    steps from a metaorder's start to the first confident signal in its direction
    false_alarm_rate   share of negative steps with |signal| above the operating threshold

Statistics, paired by seed:
    bootstrap_ci       percentile CI of a statistic by resampling seeds
    paired_difference  mean difference a - b with bootstrap CI and one-sided p-value for > 0
    cvar               expected value of the worst (1 - alpha) share
    holm               Holm step-down correction of p-values
    bootstrap_ratio    ratio of two means with a CI from resampling seeds jointly
    crn_diagnostics    variance reduction from pairing and Sharma's positive-correlation condition
"""

import math

import numpy as np


# ----- detector scoring --------------------------------------------------------------------- #

def roc_auc(scores, labels) -> float:
    """Mann-Whitney AUC with average ranks for ties; nan when one class is missing."""
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def detection_delay(signal: np.ndarray, start: int, end: int, side: str, threshold: float) -> int | None:
    """First step in [start, end) where the signal is confidently in the metaorder's direction, minus start."""
    sign = 1 if side == "bid" else -1
    window = sign * np.asarray(signal[start:end], float)
    hits = np.flatnonzero(window >= threshold)
    return int(hits[0]) if len(hits) else None


def false_alarm_rate(signal: np.ndarray, negative_mask: np.ndarray, threshold: float) -> float:
    neg = np.abs(np.asarray(signal, float)[np.asarray(negative_mask, bool)])
    return float((neg >= threshold).mean()) if len(neg) else float("nan")


def activity_labels(length: int, windows, offset: int = 0) -> np.ndarray:
    """Signed per-step labels from (start, end, side) windows: +1 buy active, -1 sell active, 0 none."""
    y = np.zeros(length, int)
    for start, end, side in windows:
        lo, hi = max(0, start - offset), min(length, end - offset)
        if hi > lo:
            y[lo:hi] += 1 if side == "bid" else -1
    return np.sign(y)


# ----- statistics --------------------------------------------------------------------------- #

def bootstrap_ci(values, stat=np.mean, n_boot: int = 2_000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """(estimate, low, high) by percentile bootstrap over the given per-seed values."""
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    boots = np.array([stat(values[i]) for i in idx])
    return float(stat(values)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_difference(a, b, n_boot: int = 2_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean of a - b (paired by position), bootstrap CI, and bootstrap p-values: one-sided for mean > 0,
    and two-sided for a difference that may have either sign."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    if len(d) == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "p_greater": float("nan"),
                "p_two_sided": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    p_greater = float(np.mean(boots <= 0))
    return {"mean": float(d.mean()), "low": float(np.quantile(boots, alpha / 2)),
            "high": float(np.quantile(boots, 1 - alpha / 2)),
            "p_greater": p_greater, "p_two_sided": min(1.0, 2 * min(p_greater, float(np.mean(boots >= 0)))),
            "n": int(len(d))}


def cvar(values, alpha: float = 0.95) -> float:
    """Mean of the worst (1 - alpha) share of values (positive values are costs)."""
    v = np.sort(np.asarray(values, float))
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    k = max(1, int(math.ceil((1 - alpha) * len(v))))
    return float(v[-k:].mean())


def holm(pvalues) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_ratio(numerator, denominator, n_boot: int = 2_000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """(mean(numerator) / mean(denominator), low, high), resampling seeds jointly; for shares of a total."""
    a, b = np.asarray(numerator, float), np.asarray(denominator, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if len(a) == 0 or b.mean() == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    means = b[idx].mean(axis=1)
    boots = np.divide(a[idx].mean(axis=1), means, out=np.full(n_boot, np.nan), where=means != 0)
    return (float(a.mean() / b.mean()), float(np.nanquantile(boots, alpha / 2)), float(np.nanquantile(boots, 1 - alpha / 2)))


def crn_diagnostics(a, b, n_boot: int = 2_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """What pairing arms ``a`` and ``b`` on one seed buys, compared with independent seeds.

    ``variance_reduction`` is Var(a) + Var(b) over Var(a - b), the factor by which pairing shrinks the variance
    of the estimated difference (Klein et al.); above 1 it is worth having. It exceeds 1 exactly when
    Cov(a, b) > 0, which is Sharma's positive-correlation condition: ``sharma_holds`` asks that the correlation's
    whole interval be positive. Percentile CIs come from resampling seeds.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]

    def stats(x, y):
        var_diff = np.var(x - y, ddof=1)
        vr = (np.var(x, ddof=1) + np.var(y, ddof=1)) / var_diff if var_diff > 0 else float("inf")
        denom = np.std(x, ddof=1) * np.std(y, ddof=1)
        return vr, (np.cov(x, y)[0, 1] / denom if denom > 0 else float("nan"))

    if len(a) < 3:
        return {"n": int(len(a)), "variance_reduction": float("nan"), "vr_low": float("nan"), "vr_high": float("nan"),
                "correlation": float("nan"), "corr_low": float("nan"), "corr_high": float("nan"),
                "sharma_holds": False, "var_paired": float("nan"), "var_independent": float("nan")}
    vr, corr = stats(a, b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    boots = np.array([stats(a[i], b[i]) for i in idx])
    finite = np.where(np.isfinite(boots[:, 0]), boots[:, 0], np.nan)
    corr_low, corr_high = np.nanquantile(boots[:, 1], alpha / 2), np.nanquantile(boots[:, 1], 1 - alpha / 2)
    return {"n": int(len(a)), "variance_reduction": float(vr),
            "vr_low": float(np.nanquantile(finite, alpha / 2)), "vr_high": float(np.nanquantile(finite, 1 - alpha / 2)),
            "correlation": float(corr), "corr_low": float(corr_low), "corr_high": float(corr_high),
            "sharma_holds": bool(corr_low > 0), "var_paired": float(np.var(a - b, ddof=1)),
            "var_independent": float(np.var(a, ddof=1) + np.var(b, ddof=1))}
