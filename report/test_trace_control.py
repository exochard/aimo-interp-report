import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import trace_control as T


def row(ratio, missing=0.0):
    return {"features": {"neg_ratio": ratio, "neg_missing": missing}}


def test_counts_ignore_missing_controls():
    rows = [row(0.2), row(0.5), row(1.5), row(float("nan"), 1.0), row(float("nan"), 1.0)]
    assert T.summarise(rows) == {"rows": 5, "with_control": 3, "control_moved_more": 2, "median_ratio": 0.5}


def test_no_control_at_all():
    assert T.summarise([row(float("nan"), 1.0)])["median_ratio"] is None
