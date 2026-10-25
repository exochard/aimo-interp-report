import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import lesson9_shuffle as S


def row(p, m, r):
    return {"problem": p, "model_id": m, "reasoning_effort": "default", "is_robust": r}


# 4 problems, two models each: three agree (two True, one False), one disagrees.
ROWS = [row("A", "m1", True), row("A", "m2", True), row("B", "m1", True), row("B", "m2", True),
        row("C", "m1", False), row("C", "m2", False), row("D", "m1", True), row("D", "m2", False),
        row("E", "m1", None), row("E", "m2", True)]


def test_observed_and_null_bounds():
    r = S.shuffle_null(ROWS, perms=200, seed=0)
    assert (r["observed_multi"], r["observed_agree"]) == (4, 3)
    assert 0 <= r["null_range_95"][0] <= r["null_mean_agree"] <= r["null_range_95"][1] <= 4
    assert r["shuffles_reaching_observed"] <= 200


def test_deterministic_for_seed():
    assert S.shuffle_null(ROWS, 100, 1) == S.shuffle_null(ROWS, 100, 1)


def test_all_equal_labels_always_agree():
    rows = [row("A", "m1", True), row("A", "m2", True), row("B", "m1", True), row("B", "m2", True)]
    r = S.shuffle_null(rows, perms=50)
    assert r["null_range_95"] == [2, 2] and r["shuffles_reaching_observed"] == 50


def test_disagreeing():
    assert S.disagreeing(ROWS) == [[("m1", True), ("m2", False)]]
