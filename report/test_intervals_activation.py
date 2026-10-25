import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import intervals as I


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


PREDS = [
    {"problem_sha256": sha("P1"), "model_id": "m1", "truth": True, "probe": True, "majority": True},
    {"problem_sha256": sha("P1"), "model_id": "m2", "truth": False, "probe": True, "majority": True},
    {"problem_sha256": sha("P2"), "model_id": "m1", "truth": False, "probe": False, "majority": True},
]


def test_keys_are_model_and_problem_text():
    got = I.activation_keyed(PREDS, ["P1", "P2", "P3"], "probe")
    assert got == {("m1", "P1"): (True, True), ("m2", "P1"): (False, True), ("m1", "P2"): (False, False)}


def test_majority_column():
    got = I.activation_keyed(PREDS, ["P1", "P2"], "majority")
    assert got[("m1", "P2")] == (False, True)


def test_unknown_problem_is_an_error():
    try:
        I.activation_keyed(PREDS, ["P1"], "probe")
    except KeyError:
        return
    raise AssertionError("expected KeyError")
