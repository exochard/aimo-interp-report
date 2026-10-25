import json
import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import artifact_gate as G

DS = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def test_select_rows_casts_bool_and_keeps_index(tmp_path):
    p = tmp_path / "t.parquet"
    pd.DataFrame({"model_id": [DS, "other", DS, DS], "reasoning_effort": ["low"] * 4,
                  "problem": list("abcd"), "is_robust": pd.Series([True, True, None, False], dtype=object),
                  "max_drop": [0.1, 0.2, 0.3, 0.4]}).to_parquet(p)
    sub = G.select_rows(pd.read_parquet(p), DS)
    assert list(sub.index) == [0, 3]
    assert sub["is_robust"].dtype == bool and list(sub["is_robust"]) == [True, False]


def test_resume_skips_done_rows(tmp_path):
    path = tmp_path / "features.jsonl"
    assert G.done_rows(str(path)) == set()
    G.append_lines(str(path), [{"row": 1}, {"row": 3}])
    done = G.done_rows(str(path))
    assert done == {1, 3}
    assert G.pending_chunks([0, 1, 2, 3, 4], done, 2) == [[0, 2], [4]]


def test_shim_sets_default_keeps_explicit_and_restores():
    calls = []

    class Real:
        @staticmethod
        def from_pretrained(*a, **k):
            calls.append((a, k))
            return "m"

    mod = types.SimpleNamespace(AutoModelForCausalLM=Real)
    mm = {0: "11GiB", "cpu": "22GiB"}
    original = G.install_shim(mod, mm)
    try:
        assert mod.AutoModelForCausalLM.from_pretrained("p", dtype=1) == "m"
        mod.AutoModelForCausalLM.from_pretrained("p", max_memory={0: "1GiB"})
    finally:
        mod.AutoModelForCausalLM = original
    assert calls[0] == (("p",), {"dtype": 1, "max_memory": mm})
    assert calls[1][1]["max_memory"] == {0: "1GiB"}
    assert mod.AutoModelForCausalLM is Real


def test_device_counts():
    m = types.SimpleNamespace(hf_device_map={"a": 0, "b": 0, "c": "cpu", "d": "disk"})
    assert G.device_counts(m) == {"cuda": 2, "cpu": 1, "other": 1}
    assert G.device_counts(object()) is None


def test_stats_hand_computed():
    labels = [True, True, False, False, False]
    scores = [0.0, 1.0, 1.0, 2.0, 3.0]
    # robust vs non-robust pairs: (0<1,0<2,0<3,1==1 half,1<2,1<3) = 5.5 of 6
    assert G.auc_robust(scores, labels) == 5.5 / 6
    assert G.auc_robust([1, 2], [True, True]) is None
    assert G.majority_label(labels) is False
    pred = [True, False, False, False, True]  # recalls: robust 1/2, non 2/3
    assert abs(G.balanced_accuracy(pred, labels) - (0.5 + 2 / 3) / 2) < 1e-12


def rows_for(scores, labels, drops):
    return [{"row": i, "problem_sha": f"p{i}", "is_robust": y, "score": s, "max_drop": d}
            for i, (s, y, d) in enumerate(zip(scores, labels, drops))]


def test_summarize_and_bootstrap_deterministic():
    labels = [True, True, False, False, False]
    scores = [0.0, 1.0, 1.0, 2.0, 3.0]
    out = G.summarize(rows_for(scores, labels, [1, 2, 3, 4, 5]), threshold=1.0, n_boot=2000)
    # pred robust: rows 0 only; correct: row0, row2? row2 score 1.0 not < 1 -> non-robust ok, rows 3,4 ok
    assert out["majority_label"] == "non-robust" and out["majority_correct"] == 3
    assert out["artifact_correct"] == 4
    assert out["right_where_majority_wrong"] == 1 and out["wrong_where_majority_right"] == 0
    assert out["diff_correct_minus_majority"] == 1
    lo, hi = out["diff_ci95"]
    assert lo <= 1 <= hi
    again = G.summarize(rows_for(scores, labels, [1, 2, 3, 4, 5]), threshold=1.0, n_boot=2000)
    assert again["diff_ci95"] == out["diff_ci95"]
    assert out["spearman_score_max_drop"] > 0.9


def test_feature_line_has_no_problem_text(tmp_path):
    text = "SECRET PROBLEM STATEMENT"
    line = G.feature_line(7, text, 1, 0.5, ("f1", "f2"), {"f1": 1, "f2": 2.5, "extra": 9})
    path = tmp_path / "features.jsonl"
    G.append_lines(str(path), [line])
    raw = path.read_text()
    assert text not in raw and "SECRET" not in raw
    got = json.loads(raw)
    assert got == {"row": 7, "problem_sha": G.problem_sha(text), "is_robust": True, "max_drop": 0.5,
                   "features": {"f1": 1.0, "f2": 2.5}}
    assert len(got["problem_sha"]) == 16
