import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import lesson6_counts as L


def row(base, n, robust):
    return {"base_accuracy": base, "n_base_predictions": n, "model_is_robust": robust}


ROWS = [
    row(1.0, 10, True), row(1.0, 10, False), row(1.0, 10, False),
    row(1.0, 1, False), row(1.0, 60, True),
    row(0.9, 10, False),          # not solved every sample: excluded
]


def test_any_sample_count():
    r = L.nonrobust_share(ROWS)
    assert r == {"solved_rows": 5, "non_robust": 3, "share": 0.6}


def test_ten_samples_only():
    r = L.nonrobust_share(ROWS, n_base=10)
    assert r == {"solved_rows": 3, "non_robust": 2, "share": 2 / 3}


def test_empty_selection():
    assert L.nonrobust_share([row(0.5, 10, True)]) == {"solved_rows": 0, "non_robust": 0, "share": None}


def test_overall_share_and_unique_problems():
    rows = [dict(row(1.0, 10, False), problem_id="a"), dict(row(1.0, 10, True), problem_id="a"),
            dict(row(0.9, 10, False), problem_id="b"), dict(row(1.0, 10, True), problem_id="c")]
    assert L.overall(rows) == {"rows": 4, "non_robust": 2, "share": 0.5,
                               "solved_rows": 3, "unique_problems": 3}
