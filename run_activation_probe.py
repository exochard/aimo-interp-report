"""Extract all small-track hidden vectors from pinned local models, one model at a time."""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sub"))
import trainset
from activation_features import capture

SMALL = ("Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
         "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")


def row_key(model, problem, effort):
    return hashlib.sha256(json.dumps([model, problem, effort]).encode()).hexdigest()


def load_model(model_id, models_root):
    """(model, tokenizer, provenance) of one pinned local model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    path = models_root / model_id
    revision = (path / ".cache/huggingface/download/config.json.metadata").read_text().splitlines()[0]
    provenance = {"revision": revision,
        "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
        "precision": "bf16" if model_id.endswith("4B") else "nf4-bf16"}
    kw = dict(dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True)
    if not model_id.endswith("4B"):
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    return model, tokenizer, provenance


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--models", type=Path, default=Path.home() / "AI/var/artifacts/models/src",
                    help="folder holding the pinned model copies")
    ap.add_argument("--only", nargs="+", default=None, help="restrict the run to these model ids")
    ap.add_argument("--append", action="store_true",
                    help="reuse --out: skip rows whose .npz exists and extend rows.json and manifest.json")
    trainset.add_argument(ap)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=a.append)
    labels, train_name, train_sha = trainset.read(a.train, HERE)
    labels = labels[labels.model_id.isin(SMALL) & labels.is_robust.notna()]
    started = time.monotonic()
    deadline = started + a.minutes * 60
    records, provenance, earlier_files = [], {}, []
    if a.append and (a.out / "rows.json").exists():
        records = json.loads((a.out / "rows.json").read_text())
    if a.append and (a.out / "manifest.json").exists():
        old = json.loads((a.out / "manifest.json").read_text())
        provenance = old["models"]
        earlier_files = old.get("label_files") or [{"train_file": old.get("train_file"), "train_sha256": old["label_sha256"]}]
    for model_id in SMALL:
        if time.monotonic() >= deadline:
            break
        if a.only is not None and model_id not in a.only:
            continue
        todo = [lab for lab in labels[labels.model_id == model_id].itertuples()
                if not (a.out / (row_key(model_id, lab.problem, lab.reasoning_effort) + ".npz")).exists()]
        if not todo:
            continue
        model, tokenizer, provenance[model_id] = load_model(model_id, a.models)
        for lab in todo:
            if time.monotonic() >= deadline:
                break
            key = row_key(model_id, lab.problem, lab.reasoning_effort)
            begin = time.monotonic()
            record = dict(key=key, model_id=model_id, problem_sha256=hashlib.sha256(lab.problem.encode()).hexdigest(),
                          reasoning_effort=lab.reasoning_effort, error=None)
            try:
                values, info = capture(model, tokenizer, lab.problem, lab.reasoning_effort)
                np.savez_compressed(a.out / (key + ".npz"), activation=values)
                record.update(info)
            except (torch.cuda.OutOfMemoryError, ValueError) as exc:
                record["error"] = repr(exc)
                torch.cuda.empty_cache()
            record["seconds"] = time.monotonic() - begin
            records = [r for r in records if r["key"] != key] + [record]
            (a.out / "rows.json").write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    manifest = dict(expected_rows=len(labels), completed_rows=sum(r["error"] is None for r in records),
        label_sha256=train_sha, train_file=train_name, models=provenance,
        seconds=time.monotonic() - started, complete=len(records) == len(labels) and all(r["error"] is None for r in records),
        source_sha256={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), HERE / "sub/activation_features.py", HERE / "sub/trace_features.py"]})
    if earlier_files:
        # Rows extracted under an earlier label file stay valid when the chosen file contains them.
        manifest["label_files"] = [f for f in earlier_files if f["train_sha256"] != train_sha] + \
            [{"train_file": train_name, "train_sha256": train_sha}]
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
