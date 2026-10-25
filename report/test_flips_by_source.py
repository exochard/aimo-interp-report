import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import flips_by_source as F


def row(pid, robust, mean, mx=None):
    return {"problem_id": pid, "robust": robust, "flip_mean": mean, "flip_max": mean if mx is None else mx,
            "orig_consistency": 0.5, "no_answer": 0.0}


def test_heldout_finds_a_clean_rule():
    rows = [row(f"p{i}", i % 2 == 0, 0.0 if i % 2 == 0 else 1.0) for i in range(8)]
    h = F.heldout(rows)
    assert h == {"n": 8, "heldout_correct": 8, "majority_correct": 0}


def test_heldout_noise_does_not_beat_majority_by_construction():
    # the feature is constant, so no rule can do better than the constant class
    rows = [row(f"p{i}", i < 6, 0.5) for i in range(8)]
    h = F.heldout(rows)
    assert h["heldout_correct"] <= h["n"] and h["majority_correct"] >= 5


def test_problem_is_held_out_as_a_unit():
    # two rows of one problem with opposite labels: both are held out together
    rows = [row("a", True, 0.0), row("a", False, 1.0), row("b", True, 0.0), row("c", False, 1.0)]
    assert F.heldout(rows)["n"] == 4


def test_bootstrap_interval_on_a_separable_fixture():
    # robust rows flip less on every problem: AUC is 1 and every resample keeps it at 1
    rows = [row(f"p{i}", i % 2 == 0, 0.1 if i % 2 == 0 else 0.9) for i in range(12)]
    assert F.auc_mean_flip(rows) == 1.0
    assert F.bootstrap_auc(rows) == (1.0, 1.0)
    assert F.bootstrap_auc(rows, draws=200, seed=3) == F.bootstrap_auc(rows, draws=200, seed=3)


def test_bootstrap_interval_brackets_a_mixed_fixture_and_diff_is_zero_for_identical_sets():
    rows = [row(f"p{i}", i % 2 == 0, 0.3 + 0.1 * (i % 3)) for i in range(20)]
    lo, hi = F.bootstrap_auc(rows, draws=500)
    assert 0.0 <= lo <= F.auc_mean_flip(rows) <= hi <= 1.0
    dlo, dhi = F.bootstrap_diff(rows, rows, draws=500)
    assert dlo <= 0.0 <= dhi


def test_restrict_keeps_only_surface_permutation_problems():
    rows = [dict(row(f"p{i}", True, 0.1), problem=f"text{i}") for i in range(5)]
    perms = [("text0", "distract"), ("text1", '["distract", "rename"]'), ("text2", "domain"),
             ("text3", '["distract", "rephrase"]'), ("text4", "typos"), ("text4", "domain")]
    kept = F.restrict_to_types(rows, perms)
    assert [r["problem"] for r in kept] == ["text0", "text1"]
    assert F.permutation_types('["a", "b"]') == {"a", "b"} and F.permutation_types(" rename ") == {"rename"}
