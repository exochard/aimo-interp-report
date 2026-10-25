import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import sample_transfer as S


def srow(model, problem, decay):
    return {"model_id": model, "original_problem": problem, "absolute_accuracy_decay": decay}


def test_labels_use_the_largest_decay_and_leave_the_band_unlabelled():
    rows = [srow("m", "p1", 0.0), srow("m", "p1", 0.10), srow("m", "p2", 0.0), srow("m", "p2", 0.25),
            srow("m", "p3", 0.18), srow("m", "p4  x", 0.0)]
    lab = S.sample_labels(rows)
    assert lab[("m", "p1")] is True and lab[("m", "p2")] is False
    assert lab[("m", "p3")] is None and lab[("m", "p4 x")] is True


def test_group_prior_scores_only_mapped_configurations():
    train = [{"model_id": "openai/gpt-oss-120b", "reasoning_effort": "low", "problem": "p1", "is_robust": v}
             for v in (True, True, False)]
    train.append({"model_id": "other", "reasoning_effort": "default", "problem": "p9", "is_robust": False})
    sample = [srow("gpt-oss-120b:low", "p1", 0.0), srow("gpt-oss-120b:low", "p2", 0.5),
              srow("gpt-oss-120b:low", "p3", 0.15), srow("unmapped:low", "p1", 0.0)]
    r = S.score(train, sample)
    assert r["scored"] == 2 and r["group_prior_correct"] == 1 and r["majority_correct"] == 1
    assert r["labelled"] == 3 and r["problems"] == 3 and r["problems_in_train"] == 1
