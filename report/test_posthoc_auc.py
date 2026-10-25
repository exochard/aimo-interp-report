import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import posthoc_auc as P


def test_auc_perfect_inverted_and_tied():
    y = [True, True, False, False]
    assert P.auc(y, [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert P.auc(y, [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert P.auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert P.auc([True, True], [0.1, 0.2]) is None


def test_prior_matched_cut_predicts_the_stated_share():
    y = [True, True, True, False, False, True, False, True, True, True]
    s = [9, 8, 7, 0.1, 0.2, 6, 0.3, 5, 4, 3]
    correct, predicted = P.prior_matched(y, s, 0.7)
    assert predicted == 7 and correct == 10
    assert P.prior_matched(y, s, 0.0) == (3, 0)
    # a tie at the cut keeps every tied row
    assert P.prior_matched([True, False, False], [1, 1, 0], 1 / 3)[1] == 2


def test_training_share_leaves_the_rows_own_group_out():
    y = [True, True, False, False]
    g = ["a", "a", "b", "c"]
    # row of a: others are 0 of 2 robust; row of b: 2 of 3; row of c: 2 of 3
    expected = np.mean([0.0, 0.0, 2 / 3, 2 / 3])
    assert abs(P.training_share(y, g) - expected) < 1e-12


def test_bootstrap_separable_signal_stays_at_one_and_is_reproducible():
    y = [i % 2 == 0 for i in range(12)]
    s = [1.0 if v else 0.0 for v in y]
    g = [f"p{i}" for i in range(12)]
    assert P.bootstrap_auc(y, s, g, draws=200) == (1.0, 1.0)
    rng = np.random.default_rng(1)
    s2 = rng.normal(size=12)
    assert P.bootstrap_auc(y, s2, g, draws=200, seed=5) == P.bootstrap_auc(y, s2, g, draws=200, seed=5)
    lo, hi = P.bootstrap_auc(y, s2, g, draws=300)
    assert 0.0 <= lo <= hi <= 1.0


def test_resampling_keeps_a_problems_rows_together():
    # one problem holds both classes; every draw that contains a class pair has both its rows
    y = [True, False, True, False]
    s = [0.9, 0.1, 0.8, 0.2]
    g = ["a", "a", "b", "b"]
    lo, hi = P.bootstrap_auc(y, s, g, draws=100)
    assert (lo, hi) == (1.0, 1.0)


def test_summary_fields_on_a_fixture():
    y = [True] * 7 + [False] * 3
    s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.2, 0.1]
    g = [f"p{i}" for i in range(10)]
    r = P.summarise(y, s, g, cut=0.33, draws=200)
    assert r["rows"] == 10 and r["own_cut_correct"] == 10 and r["auc"] == 1.0
    assert r["prior_matched_predicted_robust"] == 7 and r["prior_matched_correct"] == 10


def test_activation_row_is_skipped_when_features_are_not_shipped(tmp_path, capsys):
    assert P.activation_row(None, tmp_path, 25) is None
    assert "hidden-activation row" in capsys.readouterr().err
