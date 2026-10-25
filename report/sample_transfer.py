"""Group-prior transfer test on the organisers' public sample (Appendix C).

  python report/sample_transfer.py TRAIN.parquet SAMPLE.parquet

Labels on the sample follow the rule of Section 2: per (model, problem), the largest
`absolute_accuracy_decay` over the rows; robust at or below 0.10, non-robust from 0.25, the band between
unlabelled. The per-(model, effort) prior is the train file's majority label for the configuration, as in
entry/v1/small-groups-v1. Only sample configurations that also exist in the train file can be scored;
the pooled majority is the train file's most common label over all its labelled rows.
"""
import sys
from collections import Counter, defaultdict

ROBUST_MAX, NONROBUST_MIN = 0.10, 0.25
# sample model id -> (train model_id, train reasoning_effort)
SAMPLE_TO_TRAIN = {
    "gpt-oss-120b:low": ("openai/gpt-oss-120b", "low"),
    "qwen3-8b:low": ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default"),
}


def _norm(text):
    return " ".join(str(text).split())


def sample_labels(rows):
    """rows: dicts with model_id, original_problem, absolute_accuracy_decay.
    Returns {(model_id, problem): True / False / None (band)}."""
    worst = {}
    for r in rows:
        k = (r["model_id"], _norm(r["original_problem"]))
        worst[k] = max(worst.get(k, float("-inf")), float(r["absolute_accuracy_decay"]))
    return {k: True if v <= ROBUST_MAX else False if v >= NONROBUST_MIN else None for k, v in worst.items()}


def group_priors(train_rows):
    """{(model_id, effort): majority label} over labelled rows (ties count as robust)."""
    votes = defaultdict(list)
    for r in train_rows:
        if r["is_robust"] is not None and r["is_robust"] == r["is_robust"]:
            votes[(r["model_id"], r["reasoning_effort"])].append(bool(r["is_robust"]))
    return {k: sum(v) * 2 >= len(v) for k, v in votes.items()}


def pooled_majority(train_rows):
    c = Counter(bool(r["is_robust"]) for r in train_rows if r["is_robust"] is not None and r["is_robust"] == r["is_robust"])
    return c[True] >= c[False]


def score(train_rows, sample_rows):
    labels = sample_labels(sample_rows)
    priors, maj = group_priors(train_rows), pooled_majority(train_rows)
    out = {"sample_pairs": len(labels), "labelled": sum(v is not None for v in labels.values()),
           "robust": sum(v is True for v in labels.values()), "scored": 0, "group_prior_correct": 0,
           "majority_correct": 0, "scored_configs": sorted(m for m in {k[0] for k in labels} if m in SAMPLE_TO_TRAIN
                                                              and SAMPLE_TO_TRAIN[m] in priors)}
    for (model, _), v in labels.items():
        if v is None or model not in SAMPLE_TO_TRAIN or SAMPLE_TO_TRAIN[model] not in priors:
            continue
        out["scored"] += 1
        out["group_prior_correct"] += priors[SAMPLE_TO_TRAIN[model]] == v
        out["majority_correct"] += maj == v
    train_problems = {_norm(r["problem"]) for r in train_rows}
    sample_problems = {k[1] for k in labels}
    out["problems"] = len(sample_problems)
    out["problems_in_train"] = len(sample_problems & train_problems)
    return out


def main(train_path, sample_path):
    import pandas as pd
    t = pd.read_parquet(train_path)
    t = t.astype(object).where(t.notna(), None).to_dict("records")
    s = pd.read_parquet(sample_path).to_dict("records")
    r = score(t, s)
    print(f"sample: {r['sample_pairs']} (model, problem) pairs over {r['problems']} problems "
          f"({r['problems_in_train']} also in the train file); {r['labelled']} labelled, {r['robust']} robust")
    print(f"scorable configurations: {', '.join(r['scored_configs'])}")
    print(f"group prior {r['group_prior_correct']}/{r['scored']}  pooled majority {r['majority_correct']}/{r['scored']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
