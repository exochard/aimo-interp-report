"""Nested calibration with the overall rate p0 computed without the row's own problem (Section 4.3).

python report/calib_leak.py   (run from the repository root; about ten minutes on two threads)

sub/calibration.py's effect() takes p0 from all fitting rows, so a training row's own label enters
its p0. This rerun excludes the rows that share the row's problem (or model, for the difficulty
term) and prints the nested problem-held-out score on both label versions.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sub"), str(ROOT)]
import calibration as C  # noqa: E402


def effect_without_own(train, rows, key, other, k):
    out = np.empty(len(rows))
    for i, (kv, ov) in enumerate(zip(rows[key], rows[other])):
        p0 = min(max(train[train[other] != ov].y.mean(), 0.05), 0.95)
        t = train[(train[key] == kv) & (train[other] != ov)]
        p = (t.y.sum() + k * p0) / (len(t) + k)
        out[i] = np.log(p / (1 - p))
    return out


def main():
    C.effect = effect_without_own
    for version in (None, "1015"):
        df = C.load(ROOT / "runs/feat/features.jsonl", version)
        sel, _, _ = C.outer(df, "problem", False)
        print(version or "82-row version", C.score(df.y.to_numpy(), sel))


if __name__ == "__main__":
    main()
