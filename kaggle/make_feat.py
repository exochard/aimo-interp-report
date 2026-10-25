"""Build kaggle/feat/run.py: sub/perturb.py and sub/features.py inlined with their hashes,
then a driver that writes one feature row per (model, problem) to /kaggle/working.

The driver takes optional flags (none on Kaggle): --train pinned|1015|PATH reads that local train file
through trainset.py (lane root from $AIMO_LANE, default the working folder) instead of the Hub download;
--append keeps OUT/features.jsonl and adds the (model_id, problem) rows it lacks, loading no model that
has nothing missing; --models ID,ID restricts the models (default: $MODELS, else the four small ones).
The train file's name and sha256 go to meta.json.
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ("perturb.py", "features.py")

DRIVER = r'''
import json, os, subprocess, sys, time
if not os.environ.get("SKIP_INSTALL"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.13.0", "accelerate"], check=True)
for name, src in SOURCES.items():
    open(name, "w").write(src)
sys.path.insert(0, os.getcwd())
import argparse, hashlib
import pandas as pd, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from features import FEATURES, problem_features
from perturb import variants
import trainset

_ap = argparse.ArgumentParser()
_ap.add_argument("--train", default=None)
_ap.add_argument("--append", action="store_true")
_ap.add_argument("--models", default=None)
ARGS, _ = _ap.parse_known_args()

OUT = os.environ.get("OUT", "/kaggle/working")
DEVICE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
MODELS = os.environ.get("MODELS", "Qwen/Qwen3.5-4B,Skywork/Skywork-OR1-Math-7B,allenai/Olmo-3-7B-Think,deepseek-ai/DeepSeek-R1-0528-Qwen3-8B").split(",")
if ARGS.models:
    MODELS = ARGS.models.split(",")
ONLY_LABELLED = os.environ.get("ONLY_LABELLED", "1" if not torch.cuda.is_available() else "") == "1"
if ARGS.train:
    train, TRAIN_NAME, TRAIN_SHA = trainset.read(ARGS.train, os.environ.get("AIMO_LANE", "."))
else:
    p = hf_hub_download("aimo-interp/train-main-v2", "data/train-00000-of-00001.parquet", repo_type="dataset")
    train = pd.read_parquet(p)
    TRAIN_SHA = hashlib.sha256(open(p, "rb").read()).hexdigest()
    TRAIN_NAME = next((n for n, (_, h) in trainset.KNOWN.items() if h == TRAIN_SHA), "hub")
# A local train file with labelled-only extraction needs no validation download.
if ARGS.train and ONLY_LABELLED:
    val_problems = []
else:
    q = hf_hub_download("aimo-interp/val-sample", "data/validation-00000-of-00001.parquet", repo_type="dataset")
    val_problems = list(pd.read_parquet(q).original_problem)
LABELLED = {(r.model_id, r.problem) for r in train.itertuples() if r.is_robust is not None}
# Largest model first: a session that runs out of time loses the cheap model, not the costly one.
MODELS = sorted(MODELS, key=lambda m: "4B" in m)
probs = list(dict.fromkeys(list(train.problem) + val_problems))
DONE = set()
if ARGS.append and os.path.exists(f"{OUT}/features.jsonl"):
    with open(f"{OUT}/features.jsonl") as fh:
        DONE = {(r["model_id"], r["problem"]) for r in map(json.loads, filter(str.strip, fh))}
print(f"{len(probs)} problems, transformers {transformers.__version__}, {DEVICE}", flush=True)
meta = {"module_sha256": HASHES, "transformers": transformers.__version__, "device": DEVICE,
        "rejected": {i: variants(t)[1] for i, t in enumerate(probs) if variants(t)[1]}, "models": {},
        "train_file": TRAIN_NAME, "train_sha256": TRAIN_SHA}
if ARGS.append and os.path.exists(f"{OUT}/meta.json"):
    old = json.load(open(f"{OUT}/meta.json"))
    meta["models"] = old.get("models", {})
    meta["earlier_train"] = old.get("earlier_train", []) + (
        [{"train_file": old.get("train_file"), "train_sha256": old["train_sha256"]}]
        if old.get("train_sha256") not in (None, TRAIN_SHA) else [])
with open(f"{OUT}/features.jsonl", "a" if ARGS.append else "w") as fh:
    for model_id in MODELS:
        todo = probs if not ONLY_LABELLED else [t for t in probs if (model_id, t) in LABELLED]
        if ARGS.append:
            todo = [t for t in todo if (model_id, t) not in DONE]
            if not todo:
                continue
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
        load_s = time.time() - t0
        times = []
        for k, text in enumerate(todo):
            t = time.time()
            try:
                row = problem_features(model, tok, text)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = None
            times.append(time.time() - t)
            fh.write(json.dumps({"model_id": model_id, "problem": text, "features": row}) + "\n")
            fh.flush()
        meta["models"][model_id] = {"load_s": round(load_s, 1), "per_problem_s": round(sum(times) / max(len(times), 1), 3),
                                    "max_problem_s": round(max(times), 3) if times else None, "n": len(times)}
        print(model_id, meta["models"][model_id], flush=True)
        del model
        torch.cuda.empty_cache()
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
'''


def build(out, cpu=False):
    sources = {m: (HERE.parent / "sub" / m).read_text() for m in MODULES}
    hashes = {m: hashlib.sha256(s.encode()).hexdigest() for m, s in sources.items()}
    sources["trainset.py"] = (HERE.parent / "trainset.py").read_text()
    out.mkdir(exist_ok=True)
    (out / "run.py").write_text('"""AIMO prompt-only feature extraction (built by kaggle/make_feat.py)."""\n'
                                f"SOURCES = {json.dumps(sources)}\nHASHES = {json.dumps(hashes)}\n" + DRIVER)
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"giuseppecastelluccio/aimo-feat{'-cpu' if cpu else ''}", "title": f"aimo-feat{'-cpu' if cpu else ''}", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": not cpu,
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "kernel_sources": [],
        "competition_sources": [], "model_sources": [], **({} if cpu else {"machine_shape": "NvidiaL4"})}, indent=2))
    return hashes


def main():
    import sys
    cpu = "--cpu" in sys.argv
    print(build(HERE / ("feat-cpu" if cpu else "feat"), cpu))


if __name__ == "__main__":
    main()
