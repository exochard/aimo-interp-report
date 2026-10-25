"""AIMO prompt-only feature extraction (built by kaggle/make_feat.py)."""
SOURCES = {"perturb.py": "\"\"\"Deterministic, math-preserving prompt variants.\n\nOnly prose changes: text inside $...$, \\\\(...\\\\), \\\\[...\\\\] and every digit stay byte for byte,\nand a variant whose digit sequence or math spans differ from the original is rejected (the\noriginal is used in its place and the rejection is reported). The organisers' variants\n(typos, distract, rename, rephrase, domain, expert) are unknown; these are proxies.\n\"\"\"\n\nimport hashlib\nimport random\nimport re\n\nMATH = re.compile(r\"(\\$\\$.*?\\$\\$|\\$[^$]*\\$|\\\\\\(.*?\\\\\\)|\\\\\\[.*?\\\\\\])\", re.S)\nNEUTRAL = (\n    \"The problem was first posed in a regional mathematics circle.\",\n    \"Take your time and read every condition carefully.\",\n    \"This question appeared in a practice set for a competition.\",\n)\n\n\ndef _seed(text: str) -> int:\n    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], \"little\")\n\n\ndef _math_and_digits(text: str) -> tuple:\n    return tuple(MATH.findall(text)), re.sub(r\"\\D\", \"\", text)\n\n\ndef typos(text: str) -> str:\n    \"\"\"Swap two inner letters in about one prose word in eight.\"\"\"\n    rng = random.Random(_seed(text))\n    out = []\n    for part in MATH.split(text):\n        if MATH.fullmatch(part):\n            out.append(part)\n            continue\n        words = part.split(\" \")\n        for i, w in enumerate(words):\n            if len(w) > 4 and w.isalpha() and rng.random() < 0.125:\n                j = rng.randrange(1, len(w) - 2)\n                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]\n        out.append(\" \".join(words))\n    return \"\".join(out)\n\n\ndef distract(text: str) -> str:\n    \"\"\"Append one irrelevant sentence that carries no number.\"\"\"\n    return f\"{text.rstrip()} {NEUTRAL[_seed(text) % len(NEUTRAL)]}\"\n\n\nVARIANTS = {\"typos\": typos, \"distract\": distract}\n\n\ndef variants(text: str) -> tuple:\n    \"\"\"({name: text}, [rejected names]).\"\"\"\n    want = _math_and_digits(text)\n    out, rejected = {}, []\n    for name, fn in VARIANTS.items():\n        v = fn(text)\n        if _math_and_digits(v) != want:\n            rejected.append(name)\n            v = text\n        out[name] = v\n    return out, rejected\n", "features.py": "\"\"\"Prompt-only robustness features: one forward pass over a problem and its variants.\n\nNo generation. Per problem, the original and each variant from perturb.py run as one padded\nbatch; from it come eight dimensionless scalars:\n\n- nll: mean next-token NLL of the original prompt (how familiar the text is to the model);\n- log_tokens: log of the original prompt length;\n- dnll_<v>: the variant's mean NLL minus the original's;\n- dcos50_<v>, dcos75_<v>: 1 - cosine between mask-mean-pooled hidden states of the original\n  and the variant at 50% and 75% of the depth (same fractions for every architecture).\n\"\"\"\n\nimport math\nimport time\n\nimport torch\n\nfrom perturb import VARIANTS, variants\n\nSUFFIX = \"\\n\\nPlease reason step by step, and put your final answer within \\\\boxed{}.\"\nFRACTIONS = (0.5, 0.75)\nFEATURES = [\"nll\", \"log_tokens\"] + [f\"{k}_{v}\" for v in VARIANTS for k in (\"dnll\", \"dcos50\", \"dcos75\")]\nMAX_PROMPT_TOKENS = 2048\n\n\ndef prompt(tok, text: str) -> str:\n    msgs = [{\"role\": \"user\", \"content\": text + SUFFIX}]\n    try:\n        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)\n    except Exception:  # a tokenizer without a chat template\n        return text + SUFFIX\n\n\n@torch.no_grad()\ndef problem_features(model, tok, text: str) -> dict:\n    texts = {\"original\": text, **variants(text)[0]}\n    names = list(texts)\n    if tok.pad_token is None:\n        tok.pad_token = tok.eos_token\n    tok.padding_side = \"right\"\n    batch = tok([prompt(tok, texts[n]) for n in names], return_tensors=\"pt\", padding=True,\n                truncation=True, max_length=MAX_PROMPT_TOKENS, add_special_tokens=False)\n    batch = {k: v.to(model.device) for k, v in batch.items()}\n    out = model(**batch, output_hidden_states=True, use_cache=False)\n    mask = batch[\"attention_mask\"].float()\n    ids = batch[\"input_ids\"]\n    logp = torch.log_softmax(out.logits[:, :-1].float(), -1)\n    nll_tok = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)\n    nll = (nll_tok * mask[:, 1:]).sum(1) / mask[:, 1:].sum(1).clamp(min=1)\n    depth = len(out.hidden_states) - 1\n    pooled = {}\n    for f in FRACTIONS:\n        h = out.hidden_states[max(1, round(f * depth))].float()\n        pooled[f] = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)\n    row = {\"nll\": float(nll[0]), \"log_tokens\": math.log(float(mask[0].sum()))}\n    for i, n in enumerate(names[1:], 1):\n        row[f\"dnll_{n}\"] = float(nll[i] - nll[0])\n        for f in FRACTIONS:\n            cos = torch.nn.functional.cosine_similarity(pooled[f][0], pooled[f][i], dim=0)\n            row[f\"dcos{int(f * 100)}_{n}\"] = float(1 - cos)\n    return row\n\n\ndef extract(model, tok, problems: list, deadline: float = math.inf) -> list:\n    \"\"\"One feature dict per problem, None for problems reached after the deadline.\"\"\"\n    rows = []\n    for text in problems:\n        if time.monotonic() >= deadline:\n            rows.append(None)\n            continue\n        try:\n            rows.append(problem_features(model, tok, text))\n        except torch.cuda.OutOfMemoryError:\n            torch.cuda.empty_cache()\n            rows.append(None)\n    return rows\n"}
HASHES = {"perturb.py": "fc0ed3885b6cc355f4ff492f95162475163b1582f6b15457df2cc786cfffb62f", "features.py": "989fd16d1b11605cb21d01fd8da60bd4dd3150d5955e42b9912cc856cf0966b5"}

import json, os, subprocess, sys, time
if not os.environ.get("SKIP_INSTALL"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.13.0", "accelerate"], check=True)
for name, src in SOURCES.items():
    open(name, "w").write(src)
sys.path.insert(0, os.getcwd())
import pandas as pd, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from features import FEATURES, problem_features
from perturb import variants

OUT = os.environ.get("OUT", "/kaggle/working")
DEVICE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
MODELS = os.environ.get("MODELS", "Qwen/Qwen3.5-4B,Skywork/Skywork-OR1-Math-7B,allenai/Olmo-3-7B-Think,deepseek-ai/DeepSeek-R1-0528-Qwen3-8B").split(",")
p = hf_hub_download("aimo-interp/train-main-v2", "data/train-00000-of-00001.parquet", repo_type="dataset")
q = hf_hub_download("aimo-interp/val-sample", "data/validation-00000-of-00001.parquet", repo_type="dataset")
train = pd.read_parquet(p)
ONLY_LABELLED = os.environ.get("ONLY_LABELLED", "1" if not torch.cuda.is_available() else "") == "1"
LABELLED = {(r.model_id, r.problem) for r in train.itertuples() if r.is_robust is not None}
# Largest model first: a session that runs out of time loses the cheap model, not the costly one.
MODELS = sorted(MODELS, key=lambda m: "4B" in m)
probs = list(dict.fromkeys(list(pd.read_parquet(p).problem) + list(pd.read_parquet(q).original_problem)))
print(f"{len(probs)} problems, transformers {transformers.__version__}, {DEVICE}", flush=True)
meta = {"module_sha256": HASHES, "transformers": transformers.__version__, "device": DEVICE,
        "rejected": {i: variants(t)[1] for i, t in enumerate(probs) if variants(t)[1]}, "models": {}}
with open(f"{OUT}/features.jsonl", "w") as fh:
    for model_id in MODELS:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
        load_s = time.time() - t0
        times = []
        todo = probs if not ONLY_LABELLED else [t for t in probs if (model_id, t) in LABELLED]
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
