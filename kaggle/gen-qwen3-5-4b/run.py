"""AIMO robustness experiment: sample each small model on the public training problems,
as given and under cheap string perturbations, and save every answer with its length and
mean token log-probability. The labels are not used here; analysis happens offline.

MODELS (comma list) picks the models; the default is two per kernel to fit the session.
"""
import json
import os
import random
import re
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
# vLLM upgrades torch; the image's torchaudio stays on the old CUDA build and fails at import.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)

from huggingface_hub import hf_hub_download  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODELS = os.environ.get("MODELS", "Qwen/Qwen3.5-4B").split(",")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "4"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
OUT = "/kaggle/working"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
DISTRACTORS = [
    "Note that 2027 is a prime number.",
    "A friend claims the answer is 17, but this has not been checked.",
    "The number 360 has exactly 24 positive divisors.",
]


def problems():
    p = hf_hub_download("aimo-interp/train-main-v2", "data/train-00000-of-00001.parquet", repo_type="dataset")
    q = hf_hub_download("aimo-interp/val-sample", "data/validation-00000-of-00001.parquet", repo_type="dataset")
    texts = list(pd.read_parquet(p).problem) + list(pd.read_parquet(q).original_problem)
    return list(dict.fromkeys(texts))


def typos(text: str, seed: int) -> str:
    """Swap two inner letters in about one word in eight, outside $...$ math."""
    rng = random.Random(seed)
    parts = re.split(r"(\$[^$]*\$)", text)
    out = []
    for part in parts:
        if part.startswith("$"):
            out.append(part)
            continue
        words = part.split(" ")
        for i, w in enumerate(words):
            if len(w) > 4 and w.isalpha() and rng.random() < 0.125:
                j = rng.randrange(1, len(w) - 2)
                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        out.append(" ".join(words))
    return "".join(out)


def variants(text: str, k: int) -> dict:
    return {"original": text, "typos": typos(text, k), "distract": text + " " + DISTRACTORS[k % len(DISTRACTORS)]}


def boxed(s: str):
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


def main():
    probs = problems()
    print(f"{len(probs)} problems, models {MODELS}, gpus {torch.cuda.device_count()} "
          f"{torch.cuda.get_device_name(0)}", flush=True)
    dtype = "bfloat16" if torch.cuda.get_device_capability(0)[0] >= 8 else "float16"
    for model_id in MODELS:
        t0 = time.time()
        llm = LLM(model_id, dtype=dtype, max_model_len=MAX_TOKENS + 2048, gpu_memory_utilization=0.92,
                  tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, seed=0)
        tok = llm.get_tokenizer()
        items = []
        for k, p in enumerate(probs):
            for name, text in variants(p, k).items():
                msgs = [{"role": "user", "content": text + SUFFIX}]
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                items.append({"problem_index": k, "variant": name, "prompt": prompt})
        sp = SamplingParams(n=N_SAMPLES, temperature=0.6, top_p=0.95, max_tokens=MAX_TOKENS, logprobs=0, seed=0)
        outs = llm.generate([it["prompt"] for it in items], sp)
        path = f"{OUT}/gen-{model_id.replace('/', '__')}.jsonl"
        with open(path, "w") as fh:
            for it, o in zip(items, outs):
                for c in o.outputs:
                    lp = [next(iter(d.values())).logprob for d in (c.logprobs or []) if d]
                    fh.write(json.dumps({
                        "model_id": model_id, "problem": probs[it["problem_index"]], "variant": it["variant"],
                        "answer": boxed(c.text), "n_tokens": len(c.token_ids), "finish": c.finish_reason,
                        "mean_logprob": sum(lp) / len(lp) if lp else None, "tail": c.text[-300:]}) + "\n")
        print(f"{model_id}: {len(items)} prompts x {N_SAMPLES} in {time.time() - t0:.0f}s -> {path}", flush=True)
        del llm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
