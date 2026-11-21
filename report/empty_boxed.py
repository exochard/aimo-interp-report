"""Forced-answer flips with empty \\boxed{} answers treated as missing (Section 4.3).

python report/empty_boxed.py   (run from the repository root)

Reads runs/force-r4/*.jsonl, recomputes each row's statistics with answer_flips.flip_features after
mapping "" to None, and repeats answer_flips.evaluate's problem-held-out threshold rule. Prints the
pooled result as first parsed and corrected, a fold-local per-model majority, and the rule against
each label source's own majority.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import answer_flips as af  # noqa: E402

FILES = ["Qwen3.5-4B", "Skywork-OR1-Math-7B", "DeepSeek-R1-0528-Qwen3-8B"]
FEATS = ("flip_max", "flip_mean", "orig_consistency", "no_answer")


def heldout(rows):
    pred, maj_ok = {}, 0
    for p in sorted({r["problem_id"] for r in rows}):
        train = [r for r in rows if r["problem_id"] != p]
        best = None
        for f in FEATS:
            for t in sorted({r[f] for r in train}):
                for s in (1, -1):
                    acc = sum(((r[f] <= t) if s == 1 else (r[f] > t)) == r["robust"] for r in train)
                    if best is None or acc > best[0]:
                        best = (acc, f, t, s)
        _, f, t, s = best
        maj = Counter(r["robust"] for r in train).most_common(1)[0][0]
        for r in rows:
            if r["problem_id"] == p:
                pred[r["key"]] = (r[f] <= t) if s == 1 else (r[f] > t)
                maj_ok += maj == r["robust"]
    return pred, maj_ok


def per_model_majority(rows):
    ok = 0
    for p in sorted({r["problem_id"] for r in rows}):
        train = [r for r in rows if r["problem_id"] != p]
        for r in rows:
            if r["problem_id"] == p:
                same = [x["robust"] for x in train if x["model_id"] == r["model_id"]] or [x["robust"] for x in train]
                ok += Counter(same).most_common(1)[0][0] == r["robust"]
    return ok


def source(r):
    return "filtered subset" if r["key"].endswith("augmented-agg-filtered") else "competition rows"


def report(rows):
    pred, maj = heldout(rows)
    out = {"rule": sum(pred[r["key"]] == r["robust"] for r in rows), "pooled_majority": maj,
           "per_model_majority": per_model_majority(rows), "by_source": {}}
    for s in ("competition rows", "filtered subset"):
        sub = [r for r in rows if source(r) == s]
        own = Counter(r["robust"] for r in sub).most_common(1)[0][0]
        out["by_source"][s] = {"n": len(sub), "rule": sum(pred[r["key"]] == r["robust"] for r in sub),
                               "own_majority": sum(r["robust"] == own for r in sub)}
    return out


def main():
    rows = [json.loads(line) for f in FILES for line in open(ROOT / "runs/force-r4" / f"{f}.jsonl")]
    empty = sum(a == "" for r in rows for v in r["answers"].values() for a in v)
    total = sum(len(v) for r in rows for v in r["answers"].values())
    fixed = []
    for r in rows:
        r = json.loads(json.dumps(r))
        r["answers"] = {k: [None if a == "" else a for a in v] for k, v in r["answers"].items()}
        r.update(af.flip_features(r["answers"]))
        fixed.append(r)
    print(json.dumps({"rows": len(rows), "answers": total, "empty_answers": empty,
                      "rows_with_empty": sum(any(a == "" for v in r["answers"].values() for a in v) for r in rows),
                      "as_first_parsed": report(rows), "empty_as_missing": report(fixed)}, indent=1))


if __name__ == "__main__":
    main()
