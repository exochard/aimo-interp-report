import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import per_model_auc as P


def row(m, robust, mean, mx):
    return {"model_id": m, "robust": robust, "flip_mean": mean, "flip_max": mx}


ROWS = [
    row("a/Q", True, 0.0, 0.0), row("a/Q", True, 0.5, 1.0), row("a/Q", False, 0.5, 1.0),
    row("a/Q", False, 1.0, 1.0),
    row("b/D", True, 1.0, 1.0), row("b/D", False, 0.0, 0.0),
    row("c/S", True, 0.0, 0.0), row("c/S", False, 1.0, 1.0),
]


def test_auc_ties_count_half():
    q = [r for r in ROWS if r["model_id"] == "a/Q"]
    # robust {0, .5} vs non-robust {.5, 1}: pairs 1, 1, .5, 1 -> 3.5/4
    assert P.auc(q, "flip_mean") == 0.875
    # flip_max: robust {0, 1} vs non-robust {1, 1}: pairs 1, 1, 0.5, 0.5 -> 0.75
    assert P.auc(q, "flip_max") == 0.75


def test_direction_and_single_class():
    d = [r for r in ROWS if r["model_id"] == "b/D"]
    assert P.auc(d, "flip_mean") == 0.0
    assert P.auc([row("x", True, 0.1, 0.1)], "flip_mean") is None


def test_by_model_and_pool():
    out = P.by_model(ROWS, ["Q", "S"])
    assert out["a/Q"]["rows"] == 4 and out["b/D"]["rows"] == 2
    assert out["pooled: Q + S"]["rows"] == 6
    assert out["b/D"]["flip_mean"] == 0.0
