"""Score the organisers' shipped DeepSeek uncertainty artifact on our labelled DeepSeek rows.

  python report/artifact_gate.py features --train data/train-main-v2/train-2026-10-15.parquet \
      --model-path PATH --out runs/artifact-gate [--batch-size 2] [--gpu-gib 11] [--cpu-gib 22] \
      [--limit N] [--max-new-tokens N]
  python report/artifact_gate.py score --out runs/artifact-gate

`features` generates the artifact's features for each DeepSeek-R1-0528-Qwen3-8B row and appends one
JSON line per row to OUT/features.jsonl (resumable; no problem text, only a hash). `score` runs the
artifact's estimator on those features and compares it with the majority class. The kit is imported
lazily from LANE/kit, so this module and its tests need neither the kit, torch nor transformers.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(LANE, "kit", "solutions", "uncertainty-profiling")


def use_kit():
    if KIT not in sys.path:
        sys.path.insert(0, KIT)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def problem_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def select_rows(df, model_id):
    """Labelled rows of one model; is_robust is object dtype in the parquet, so cast to bool."""
    sub = df[(df["model_id"] == model_id) & df["is_robust"].notna()].copy()
    sub["is_robust"] = sub["is_robust"].astype(bool)
    return sub


def done_rows(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["row"] for line in f if line.strip()}


def pending_chunks(indices, done, batch_size):
    todo = [i for i in indices if i not in done]
    return [todo[k:k + batch_size] for k in range(0, len(todo), batch_size)]


def feature_line(idx, problem, is_robust, max_drop, feature_names, feature_row):
    return {"row": int(idx), "problem_sha": problem_sha(problem), "is_robust": bool(is_robust),
            "max_drop": float(max_drop),
            "features": {n: float(feature_row[n]) for n in feature_names}}


def append_lines(path, lines):
    with open(path, "a") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
        f.flush()
        os.fsync(f.fileno())


class MaxMemoryShim:
    """Stands in for AutoModelForCausalLM: adds max_memory unless the caller set one."""

    def __init__(self, real, max_memory):
        self.real = real
        self.max_memory = max_memory

    def from_pretrained(self, *a, **k):
        k.setdefault("max_memory", self.max_memory)
        return self.real.from_pretrained(*a, **k)


def install_shim(module, max_memory):
    """Replace module.AutoModelForCausalLM; return the original for restoring."""
    original = module.AutoModelForCausalLM
    module.AutoModelForCausalLM = MaxMemoryShim(original, max_memory)
    return original


def device_counts(model):
    """Count model modules placed on cuda vs cpu from hf_device_map (None if absent)."""
    dmap = getattr(model, "hf_device_map", None)
    if not dmap:
        return None
    out = {"cuda": 0, "cpu": 0, "other": 0}
    for dev in dmap.values():
        d = str(dev)
        if d == "cpu":
            out["cpu"] += 1
        elif d.isdigit() or d.startswith("cuda"):
            out["cuda"] += 1
        else:
            out["other"] += 1
    return out


def majority_label(labels):
    robust = sum(bool(x) for x in labels)
    return robust * 2 >= len(labels)  # ties go to robust


def auc_robust(scores, labels):
    """P(score of a robust row < score of a non-robust row), ties half; None if a class is empty."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum((p < n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def balanced_accuracy(pred, labels):
    recalls = []
    for cls in (True, False):
        idx = [i for i, y in enumerate(labels) if y == cls]
        if idx:
            recalls.append(sum(pred[i] == cls for i in idx) / len(idx))
    return sum(recalls) / len(recalls)


def bootstrap_diff(art_ok, maj_ok, groups, n_boot=10000, seed=0):
    """95 % interval of (artifact correct - majority correct), resampling problems (groups)."""
    import numpy as np
    keys = sorted(set(groups))
    d = np.zeros(len(keys))
    for g, a, m in zip(groups, art_ok, maj_ok):
        d[keys.index(g)] += int(a) - int(m)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    sums = d[draws].sum(axis=1)
    return [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))]


def summarize(rows, threshold, n_boot=10000):
    """rows: dicts with problem_sha, is_robust, score, max_drop. Adds pred_robust to a copy."""
    from scipy.stats import spearmanr
    labels = [bool(r["is_robust"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    pred = [s < threshold for s in scores]
    maj = majority_label(labels)
    maj_ok = [y == maj for y in labels]
    art_ok = [p == y for p, y in zip(pred, labels)]
    rho = spearmanr(scores, [r["max_drop"] for r in rows])[0] if len(rows) > 2 else None
    return {
        "n": len(rows), "robust": sum(labels), "majority_label": "robust" if maj else "non-robust",
        "majority_correct": sum(maj_ok), "artifact_correct": sum(art_ok),
        "balanced_accuracy": balanced_accuracy(pred, labels),
        "right_where_majority_wrong": sum(a and not m for a, m in zip(art_ok, maj_ok)),
        "wrong_where_majority_right": sum(m and not a for a, m in zip(art_ok, maj_ok)),
        "auc_robust": auc_robust(scores, labels),
        "spearman_score_max_drop": None if rho is None or rho != rho else float(rho),
        "diff_correct_minus_majority": sum(art_ok) - sum(maj_ok),
        "diff_ci95": bootstrap_diff(art_ok, maj_ok, [r["problem_sha"] for r in rows], n_boot),
        "decision_threshold": threshold,
        "rows": [{"row": r["row"], "problem_sha": r["problem_sha"], "is_robust": bool(r["is_robust"]),
                  "score": float(r["score"]), "pred_robust": bool(p)} for r, p in zip(rows, pred)],
    }


def cv_summary(artifact):
    cv = getattr(artifact, "cv_results", None)
    if not cv:
        return None
    return {k: v for k, v in cv.items() if isinstance(v, (int, float, str, bool))} or None


def run_features(a):
    import pandas as pd
    use_kit()
    import torch
    import transformers
    from uncertainty_profile import extraction
    from uncertainty_profile.artifact import default_artifact_path, load_artifact
    from uncertainty_profile.config import DEEPSEEK_MODEL_ID

    os.makedirs(a.out, exist_ok=True)
    df = select_rows(pd.read_parquet(a.train), DEEPSEEK_MODEL_ID)
    if a.limit:
        df = df.iloc[:a.limit]
    artifact = load_artifact(DEEPSEEK_MODEL_ID)
    cfg = dataclasses.replace(artifact.generation_config, batch_size=a.batch_size,
                              **({"max_new_tokens": a.max_new_tokens} if a.max_new_tokens else {}))
    max_memory = {0: f"{a.gpu_gib}GiB", "cpu": f"{a.cpu_gib}GiB"}
    path = os.path.join(a.out, "features.jsonl")
    chunks = pending_chunks(list(df.index), done_rows(path), a.batch_size)
    print(f"{len(df)} rows, {len(chunks)} chunks to run")
    if not chunks:
        return

    t0 = time.time()
    original = install_shim(extraction, max_memory)
    try:
        model, tokenizer = extraction.load_model_and_tokenizer(a.model_path, local_files_only=True)
    finally:
        extraction.AutoModelForCausalLM = original
    chunk_seconds = []
    for chunk in chunks:
        t = time.time()
        feats = extraction.extract_generation_features(
            model=model, tokenizer=tokenizer, problem_texts=[df.loc[i, "problem"] for i in chunk],
            config=cfg, include_generated_text=False)
        append_lines(path, [feature_line(i, df.loc[i, "problem"], df.loc[i, "is_robust"],
                                         df.loc[i, "max_drop"], artifact.feature_names, f)
                            for i, f in zip(chunk, feats)])
        chunk_seconds.append(round(time.time() - t, 1))
        print(f"chunk {chunk}: {chunk_seconds[-1]}s")

    import accelerate
    import sklearn
    manifest = {
        "train_sha256": sha256_file(a.train),
        "artifact_sha256": sha256_file(str(default_artifact_path(DEEPSEEK_MODEL_ID))),
        "model_path": a.model_path, "generation_config": cfg.to_dict(), "max_memory": {str(k): v for k, v in max_memory.items()},
        "device_counts": device_counts(model),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "accelerate": accelerate.__version__, "sklearn": sklearn.__version__},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "chunk_seconds": chunk_seconds, "total_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


def run_score(a):
    import numpy as np
    import pandas as pd
    use_kit()
    from uncertainty_profile.artifact import load_artifact
    from uncertainty_profile.config import DEEPSEEK_MODEL_ID

    artifact = load_artifact(DEEPSEEK_MODEL_ID)
    with open(os.path.join(a.out, "features.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    feats = pd.DataFrame.from_records([r["features"] for r in rows], columns=list(artifact.feature_names))
    scores = np.asarray(artifact.estimator.predict(feats), dtype=float)
    for r, s in zip(rows, scores):
        r["score"] = float(s)
    out = summarize(rows, artifact.decision_threshold)
    out["cv_summary"] = cv_summary(artifact)
    out["selected_params"] = getattr(artifact, "selected_params", None)
    with open(os.path.join(a.out, "score.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"n={out['n']} robust={out['robust']} majority={out['majority_label']} "
          f"threshold={out['decision_threshold']:.4g}")
    print(f"{'majority correct':32s}{out['majority_correct']}")
    print(f"{'artifact correct':32s}{out['artifact_correct']}")
    print(f"{'balanced accuracy':32s}{out['balanced_accuracy']:.3f}")
    print(f"{'right where majority wrong':32s}{out['right_where_majority_wrong']}")
    print(f"{'wrong where majority right':32s}{out['wrong_where_majority_right']}")
    print(f"{'AUC robust':32s}{out['auc_robust']}")
    print(f"{'Spearman score vs max_drop':32s}{out['spearman_score_max_drop']}")
    lo, hi = out["diff_ci95"]
    print(f"{'artifact - majority, 95% CI':32s}{out['diff_correct_minus_majority']} [{lo:.1f}, {hi:.1f}]")
    if out["cv_summary"]:
        print("artifact cv:", out["cv_summary"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--train", required=True)
    f.add_argument("--model-path", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--batch-size", type=int, default=2)
    f.add_argument("--gpu-gib", type=int, default=11)
    f.add_argument("--cpu-gib", type=int, default=22)
    f.add_argument("--limit", type=int)
    f.add_argument("--max-new-tokens", type=int)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    run_features(a) if a.cmd == "features" else run_score(a)


if __name__ == "__main__":
    main()
