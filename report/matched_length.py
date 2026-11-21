"""Problem length under the organisers' own uncertainty-profiling selection (Section 4.2).

python report/matched_length.py   (run from the repository root; a few minutes on CPU)

Imports kit/solutions/uncertainty-profiling/scripts/train_uncertainty_regressor.py and runs its
run_model_selection (5 StratifiedGroupKFold folds by original_problem, seed 42, its 48-candidate
grid, its out-of-fold threshold rule) on augmented-sample-math-agg with the four length counts of
length_baseline.py in place of its uncertainty features. torch, transformers and accelerate are
imported there only for provenance, so empty stand-ins replace them. Also prints the length
confound: log-length AUC on all rows, without the change-of-setting ("domain") rows, and on the
small-track competition labels.
"""
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in ("torch", "transformers", "accelerate"):
    stub = types.ModuleType(name)
    stub.__version__ = "not-loaded"
    stub.Tensor = type("Tensor", (), {})
    sys.modules[name] = stub
sys.path[:0] = [str(ROOT / "kit/solutions/uncertainty-profiling/scripts"), str(ROOT / "report")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import train_uncertainty_regressor as T  # noqa: E402
from length_baseline import features  # noqa: E402

AGG = ROOT / "data/augmented-sample-math-agg/validation-00000-of-00001.parquet"
TRAIN = {"82-row version": ROOT / "data/train-main-v2/train-00000-of-00001.parquet",
         "2026-10-15": ROOT / "data/train-main-v2/train-2026-10-15.parquet"}
NAMES = ["len_log_chars", "len_words", "len_digits", "len_dollars"]


def main():
    d = pd.read_parquet(AGG).reset_index(drop=True)
    f = features(d.original_problem)
    f.columns = NAMES
    frame = pd.concat([d, f], axis=1)
    frame["model_is_robust"] = frame["model_is_robust"].map(bool)
    frame["absolute_accuracy_decay"] = frame["absolute_accuracy_decay"].astype(float)
    out = {}
    for tag, cols in (("four length counts", NAMES), ("log characters", NAMES[:1])):
        T.FEATURE_NAMES = tuple(cols)
        spec, best, results, dummy = T.run_model_selection(frame, specs=T.build_candidate_specs(), n_splits=5, seed=42)
        out[tag] = {"winner": spec.identifier, "balanced_accuracy": round(best["balanced_accuracy"], 4),
                    "median_candidate": round(float(np.median([r["balanced_accuracy"] for r in results])), 4),
                    "mean_predictor": round(dummy["balanced_accuracy"], 4)}
    y = d.model_is_robust.astype(bool)
    length = np.log1p(d.original_problem.str.len())
    domain = d.permutation_type.astype(str) == "domain"
    out["confound"] = {"auc_all": round(roc_auc_score(y, length), 3), "domain_rows": int(domain.sum()),
                       "domain_non_robust": int((~y[domain]).sum()),
                       "median_chars_domain": float(d.original_problem[domain].str.len().median()),
                       "median_chars_robust": float(d.original_problem[y].str.len().median()),
                       "auc_without_domain": round(roc_auc_score(y[~domain], length[~domain]), 3)}
    for tag, path in TRAIN.items():
        t = pd.read_parquet(path)
        t = t[~t.model_id.str.contains("gpt-oss") & t.is_robust.notna()]
        out["confound"][f"auc_competition_{tag}"] = round(roc_auc_score(t.is_robust.astype(bool), np.log1p(t.problem.str.len())), 3)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
