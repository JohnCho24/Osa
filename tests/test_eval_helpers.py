"""Eval-harness statistical helpers (bootstrap CIs, paired p-value).

No model required — pure stats. These guard against the kind of off-by-one bug
that quietly inverts a p-value the first time it actually matters.
"""
import random

from scripts.eval import bootstrap_ci, paired_bootstrap_p


def test_bootstrap_ci_brackets_mean():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    s = bootstrap_ci(xs, n_resamples=2000, alpha=0.05, rng=random.Random(0))
    assert s.n == 10
    assert s.mean == 5.5
    assert s.ci_low <= s.mean <= s.ci_high
    # 95% CI on n=10 uniform 1..10 should be roughly [3.5, 7.5]
    assert s.ci_low > 2.0
    assert s.ci_high < 9.0


def test_bootstrap_ci_empty():
    s = bootstrap_ci([], n_resamples=100, rng=random.Random(0))
    assert s.n == 0


def test_paired_p_identical_lists_returns_one():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = paired_bootstrap_p(xs, xs, n_resamples=500, rng=random.Random(0))
    assert p == 1.0


def test_paired_p_dramatic_difference_rejects_null():
    a = [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.05, 0.95, 1.0, 1.0]
    b = [5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.05, 4.95, 5.0, 5.0]
    p = paired_bootstrap_p(a, b, n_resamples=2000, rng=random.Random(0))
    assert p < 0.05, f"expected significant p-value, got {p}"


def test_paired_p_mismatched_lengths_nan():
    import math
    p = paired_bootstrap_p([1.0, 2.0], [1.0, 2.0, 3.0], n_resamples=100, rng=random.Random(0))
    assert math.isnan(p)
