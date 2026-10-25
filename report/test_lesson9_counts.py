import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import lesson9_counts as L


def row(p, m, e, r):
    return {"problem": p, "model_id": m, "reasoning_effort": e, "is_robust": r}


ROWS = [
    row("A", "m1", "default", True), row("A", "m2", "default", True),
    row("B", "m1", "default", True), row("B", "m2", "default", False),
    row("C", "m1", "default", False),
    row("D", "g", "low", True), row("D", "g", "high", True),
    row("E", "m1", "default", None), row("E", "m2", "default", True),
    row("F", "g", "low", True), row("F", "g", "high", False),
]


def test_by_model():
    r = L.agreement(ROWS, by="model")
    assert r["labelled_problems"] == 6
    assert r["multi"] == 2      # A and B; D and F have one model, E has one labelled row
    assert r["agree"] == 1      # A


def test_by_config():
    r = L.agreement(ROWS, by="config")
    assert r["multi"] == 4      # A, B, D, F
    assert r["agree"] == 2      # A, D


def test_unlabelled_rows_ignored():
    r = L.agreement([row("X", "m1", "d", None), row("X", "m2", "d", True)], by="model")
    assert r["labelled_problems"] == 1 and r["multi"] == 0


def test_overlap_normalises_whitespace():
    train = [row("Find  x.\n", "m", "d", True), row("Other", "m", "d", None)]
    r = L.overlap(["Find x.", "Other", "Missing"], train)
    assert r == {"public_problems": 3, "in_train": 2, "in_train_labelled": 1}
