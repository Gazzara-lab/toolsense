"""Ranking and classification metrics for the ToolSense evaluation harness.

All ranking metrics operate on a predicted ranking expressed as an ordered list
of item identifiers (most relevant first) and a set of gold (relevant) item
identifiers. Identifiers are compared by equality; the caller decides what an
"item" is (here: tool names).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of the gold set retrieved within the top-k.

    recall@k = |gold ∩ top_k| / |gold|. Returns 0.0 for an empty gold set.
    """
    if not gold:
        return 0.0
    top_k = set(ranked[:k])
    return len(gold & top_k) / len(gold)


def hit_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """1.0 if at least one gold item appears in the top-k, else 0.0."""
    if not gold:
        return 0.0
    return 1.0 if gold & set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: set[str]) -> float:
    """Reciprocal rank of the first relevant item (1/rank); 0.0 if none ranked."""
    if not gold:
        return 0.0
    for i, item in enumerate(ranked, start=1):
        if item in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """Normalized DCG@k with binary relevance.

    DCG = Σ_{i≤k} rel_i / log2(i + 1); IDCG is the DCG of the ideal ranking
    (all gold items first). Returns 0.0 for an empty gold set.
    """
    if not gold:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def accuracy(predictions: Sequence[str | None], golds: Sequence[str]) -> float:
    """Exact-match accuracy. A None prediction (unparseable) counts as wrong."""
    if len(predictions) != len(golds):
        raise ValueError(f"predictions ({len(predictions)}) and golds ({len(golds)}) must be equal length")
    if not golds:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, golds) if p is not None and p == g)
    return correct / len(golds)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    scores: Sequence[float], n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of per-record scores.

    ``scores`` are per-record values (e.g. 0/1 correct flags, or per-query
    recall@k). Returns the ``(1 - alpha)`` interval ``(lo, hi)`` for the mean.
    Pure standard library and deterministic given ``seed`` — the paper reports
    95% bootstrap CIs on every number, and this reports them in the same form.
    """
    n = len(scores)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    pool = list(scores)
    means = sorted(sum(rng.choices(pool, k=n)) / n for _ in range(n_resamples))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)


def bootstrap_diff_ci(
    a: Sequence[float], b: Sequence[float],
    n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float]:
    """Paired percentile bootstrap CI for ``mean(a) - mean(b)``.

    ``a`` and ``b`` are per-record values aligned by index (the same records
    scored under two conditions — e.g. ``--tool-ref name`` vs ``none``, which the
    harness runs over identical record ids in identical order). Used to put a CI
    on the tool-knowledge *lift* over the hidden-identity baseline.
    """
    n = len(a)
    if n == 0 or len(b) != n:
        return (0.0, 0.0)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    diffs.sort()
    lo = diffs[int((alpha / 2) * n_resamples)]
    hi = diffs[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)
