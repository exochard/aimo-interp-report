"""Build kaggle/perturb-gen/run.py: sub/invariance.py inlined with its hash, then a driver that
samples each small model on its labelled train-main-v2 problems as given and under the
verified perturbations, and writes every answer with its length and mean log-probability.

Differences from kaggle/gen/run.py, whose 8,192-token cap left 495 of 516 samples without a
boxed answer: a 16,384-token cap, only the 34 labelled (model, problem) pairs, perturbations
that change the mathematics (invariance.py) in place of prose typos, and a wall-clock guard
per model. The guard stops submitting new chunks when the model's share of BUDGET_S is spent
and records what was reached, so a partial run still yields complete (problem, variant) cells.
meta.json carries tokens per second per model: the input for the official 3,600 s path.

Round 2: a problem's variants are one per distinct family (rename_a and rename_b are one
family), semantic families before notation-only ones, ordered by a hash of problem and family so
the families are spread over problems. BUDGET_S is capped at 5,400 s (1.5 GPU-h) in the code:
a larger value stops the kernel before it loads a model. PREFLIGHT=1 (with TRAIN_PARQUET set to
a local copy) runs everything except vLLM and the GPU: source hash, split identity, the cell
table and its coverage, then exits; the kernel itself prints the same lines before loading.
Per-model tokens per second is printed after every chunk and kept in meta.json.

Not pushed from here; the lead runs `kaggle kernels push -p kaggle/perturb-gen` after the
2026-10-17 GPU reset.
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ("invariance.py",)

DRIVER = r'''
import hashlib, json, os, subprocess, sys, time
START = time.monotonic()
PREFLIGHT = os.environ.get("PREFLIGHT") == "1"
HARD_CAP_S = 5400  # 1.5 GPU-hours
for name, src in SOURCES.items():
    assert hashlib.sha256(src.encode()).hexdigest() == HASHES[name], f"source hash mismatch: {name}"
    open(name, "w").write(src)
sys.path.insert(0, os.getcwd())
if not PREFLIGHT:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "sympy"], check=True)
    # vLLM upgrades torch; the image's torchaudio stays on the old CUDA build and fails at import.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)

import pandas as pd
from invariance import coverage, family, one_per_family, perturbations, SURFACE
if not PREFLIGHT:
    import torch
    from huggingface_hub import hf_hub_download
    from vllm import LLM, SamplingParams

MODELS = os.environ.get("MODELS", "Qwen/Qwen3.5-4B,allenai/Olmo-3-7B-Think,deepseek-ai/DeepSeek-R1-0528-Qwen3-8B,Skywork/Skywork-OR1-Math-7B").split(",")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "4"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16384"))
BUDGET_S = float(os.environ.get("BUDGET_S", "5000"))  # all models, load time included
assert BUDGET_S <= HARD_CAP_S, f"BUDGET_S {BUDGET_S} exceeds the {HARD_CAP_S} s (1.5 GPU-h) cap"
EXPECTED_PAIRS = 34  # labelled (model, problem) rows of the four small models in train-main-v2
MAX_VARIANTS = int(os.environ.get("MAX_VARIANTS", "3"))
CHUNK = int(os.environ.get("CHUNK", "16"))  # prompts per generate call
OUT = os.environ.get("OUT", "/kaggle/working")
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + len("\\boxed{")
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            if depth == 0:
                return s[j:k].strip()
            depth -= 1
    return None


p = os.environ["TRAIN_PARQUET"] if PREFLIGHT else hf_hub_download("aimo-interp/train-main-v2", "data/train-00000-of-00001.parquet", repo_type="dataset")
train = pd.read_parquet(p)
pairs = {}
for r in train.itertuples():
    if r.model_id in MODELS and r.is_robust is not None:
        pairs.setdefault(r.model_id, []).append(r.problem)
n_pairs = sum(len(v) for v in pairs.values())
problems = sorted({t for ts in pairs.values() for t in ts})
split_id = hashlib.sha256("\n".join(problems).encode()).hexdigest()
if MODELS == ["Qwen/Qwen3.5-4B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Skywork/Skywork-OR1-Math-7B"]:
    assert n_pairs == EXPECTED_PAIRS, f"{n_pairs} labelled pairs, expected {EXPECTED_PAIRS}"
cells, manifest = {}, {}
for text in problems:
    ps = one_per_family(perturbations(text))
    # semantic families first; within a tier, spread families over problems by a hash.
    ps.sort(key=lambda v: (family(v.name) in SURFACE, hashlib.sha256((text + family(v.name)).encode()).hexdigest()))
    ps = ps[:MAX_VARIANTS]
    cells[text] = {"original": text, **{v.name: v.text for v in ps}}
    manifest[hashlib.sha256(text.encode()).hexdigest()[:12]] = [{"variant": v.name, "family": family(v.name), "check": v.check} for v in ps]
cov = coverage(problems, MAX_VARIANTS)
ident = {"parquet": os.path.basename(p), "parquet_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "split_sha256": split_id,
         "pairs": {m: len(ts) for m, ts in pairs.items()}, "n_pairs": n_pairs, "problems": len(problems),
         "problems_with_full_variants": sum(len(c) - 1 >= MAX_VARIANTS for c in cells.values()),
         "coverage_min_families": cov["with_min_families"], "coverage_min_semantic_families": cov["with_min_semantic_families"],
         "module_sha256": HASHES}
print("IDENTITY", json.dumps(ident), flush=True)
json.dump({"identity": ident, "variants": manifest}, open(f"{OUT}/variants.json", "w"), indent=1)
if PREFLIGHT:
    print("PREFLIGHT ok", flush=True)
    sys.exit(0)
meta = {"module_sha256": HASHES, "identity": ident, "device": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count(),
        "torch": torch.__version__, "n_samples": N_SAMPLES, "max_tokens": MAX_TOKENS, "budget_s": BUDGET_S, "models": {}}
print({m: len(ts) for m, ts in pairs.items()}, flush=True)
order = sorted(MODELS, key=lambda m: len(pairs.get(m, [])))
dtype = "bfloat16" if torch.cuda.get_device_capability(0)[0] >= 8 else "float16"
with open(f"{OUT}/perturb-gen.jsonl", "w") as fh:
    for idx, model_id in enumerate(order):
        if model_id not in pairs:
            continue
        share = (BUDGET_S - (time.monotonic() - START)) / (len(order) - idx)
        t0 = time.monotonic()
        llm = LLM(model_id, dtype=dtype, max_model_len=MAX_TOKENS + 2048, gpu_memory_utilization=0.92,
                  tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, seed=0)
        tok = llm.get_tokenizer()
        # Every problem's original first, then variants round by round, so a cut keeps coverage.
        items = []
        for rnd in range(1 + MAX_VARIANTS):
            for text in dict.fromkeys(pairs[model_id]):
                names = list(cells[text])
                if rnd < len(names):
                    msgs = [{"role": "user", "content": cells[text][names[rnd]] + SUFFIX}]
                    items.append({"problem": text, "variant": names[rnd], "prompt": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)})
        sp = SamplingParams(n=N_SAMPLES, temperature=0.6, top_p=0.95, max_tokens=MAX_TOKENS, logprobs=0, seed=0)
        load_s, toks, done = time.monotonic() - t0, 0, 0
        gen_t0 = time.monotonic()
        for i in range(0, len(items), CHUNK):
            if time.monotonic() - t0 >= share:
                break
            chunk = items[i:i + CHUNK]
            for it, o in zip(chunk, llm.generate([c["prompt"] for c in chunk], sp)):
                for c in o.outputs:
                    lp = [next(iter(d.values())).logprob for d in (c.logprobs or []) if d]
                    toks += len(c.token_ids)
                    fh.write(json.dumps({"model_id": model_id, "problem": it["problem"], "variant": it["variant"],
                                         "answer": boxed(c.text), "n_tokens": len(c.token_ids), "finish": c.finish_reason,
                                         "mean_logprob": sum(lp) / len(lp) if lp else None, "tail": c.text[-300:]}) + "\n")
            fh.flush()
            done += len(chunk)
            el = time.monotonic() - gen_t0
            print(f"{model_id} chunk {i // CHUNK + 1}: {done}/{len(items)} prompts, {toks} tokens, {toks / max(el, 1e-9):.1f} tok/s, elapsed {time.monotonic() - START:.0f}s", flush=True)
        gen_s = time.monotonic() - gen_t0
        meta["models"][model_id] = {"load_s": round(load_s, 1), "gen_s": round(gen_s, 1), "prompts_done": done,
                                    "prompts_total": len(items), "tokens": toks, "tokens_per_s": round(toks / max(gen_s, 1e-9), 1),
                                    "share_s": round(share, 1)}
        print(model_id, meta["models"][model_id], flush=True)
        del llm
        torch.cuda.empty_cache()
meta["wall_s"] = round(time.monotonic() - START, 1)
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
'''


def main():
    sources = {m: (HERE.parent / "sub" / m).read_text() for m in MODULES}
    hashes = {m: hashlib.sha256(s.encode()).hexdigest() for m, s in sources.items()}
    out = HERE / "perturb-gen"
    out.mkdir(exist_ok=True)
    (out / "run.py").write_text('"""AIMO answers under verified perturbations (built by kaggle/make_perturb_gen.py)."""\n'
                                f"SOURCES = {json.dumps(sources)}\nHASHES = {json.dumps(hashes)}\n" + DRIVER)
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": "giuseppecastelluccio/aimo-perturb-gen", "title": "aimo-perturb-gen", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "kernel_sources": [],
        "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaL4"}, indent=2))
    print(hashes)


if __name__ == "__main__":
    main()
