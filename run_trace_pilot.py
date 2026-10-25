"""Local pilot of the counterfactual trace features on the small-track labels (plan 2026-10-13 r1, section 4).

  python run_trace_pilot.py OUT.jsonl [MINUTES] [--train pinned|1015|PATH] [--append] [--models ID ...]

--append keeps the rows already in OUT and runs only the (model, effort, problem) keys it lacks; a model with
nothing missing is never loaded. --models restricts the run to those model ids. The chosen train file's name
and sha256 go to OUT.train.json (earlier files stay listed under "earlier").

One model at a time from the local pinned copies: BF16 when it fits the 16 GB card, otherwise 4-bit
NF4 with BF16 compute, which makes this a timing and feature screen rather than official-model
evidence. Every labelled (model, problem, effort) row gets a line, `features: null` when the
deadline or memory stopped it. Feeds fit_trace_probe.py.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sub"))
import trainset  # noqa: E402
from trace_features import extract  # noqa: E402

MODELS = Path.home() / "AI/var/artifacts/models/src"
SMALL = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
         "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
BF16_MAX_B = 5.0  # billions of parameters that still fit in BF16 beside activations


def load(model_id):
    path = MODELS / model_id
    tok = AutoTokenizer.from_pretrained(path)
    cfg = json.loads((path / "config.json").read_text())
    big = not model_id.endswith("4B")
    kw = dict(dtype=torch.bfloat16, device_map="cuda:0")
    if big:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    return model, tok, ("nf4" if big else "bf16"), cfg.get("architectures")


def present_keys(out):
    """(model_id, reasoning_effort, problem) of every row already in `out`."""
    keys = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["model_id"], r["reasoning_effort"], r["problem"]))
    return keys


def write_train_record(out, name, sha, append):
    path = out.with_name(out.name + ".train.json")
    record = {"train_file": name, "train_sha256": sha}
    if append and path.exists():
        old = json.loads(path.read_text())
        earlier = old.pop("earlier", [])
        if old.get("train_sha256") != sha:
            earlier.append(old)
        if earlier:
            record["earlier"] = earlier
    path.write_text(json.dumps(record, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("minutes", nargs="?", type=float, default=30)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    trainset.add_argument(ap)
    a = ap.parse_args(argv)
    out = a.out
    deadline = time.monotonic() + 60 * a.minutes
    labels, tname, tsha = trainset.read(a.train, HERE)
    wanted = [m for m in SMALL if a.models is None or m in a.models]
    labels = labels[labels.model_id.isin(wanted)]
    have = present_keys(out) if a.append else set()
    write_train_record(out, tname, tsha, a.append)
    with out.open("a" if a.append else "w") as fh:
        for model_id in wanted:
            rows = labels[labels.model_id == model_id]
            rows = rows[[(model_id, e, p) not in have for e, p in zip(rows.reasoning_effort, rows.problem)]]
            if rows.empty:
                continue
            t0 = time.monotonic()
            model, tok, quant, arch = load(model_id)
            load_s = time.monotonic() - t0
            for effort, grp in rows.groupby("reasoning_effort"):
                t = time.monotonic()
                feats = extract(model, tok, list(grp.problem), effort, deadline)
                per = (time.monotonic() - t) / max(1, len(grp))
                for problem, f in zip(grp.problem, feats):
                    fh.write(json.dumps({"model_id": model_id, "problem": problem, "reasoning_effort": effort,
                                         "features": f, "quant": quant, "seconds_per_problem": round(per, 1),
                                         "load_seconds": round(load_s, 1)}) + "\n")
                fh.flush()
                print(json.dumps({"model": model_id, "effort": effort, "rows": len(grp),
                                  "done": sum(f is not None for f in feats), "s_per_problem": round(per, 1),
                                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
