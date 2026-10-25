"""AIMO prompt-only feature extraction (built by kaggle/make_feat.py)."""
SOURCES = {"perturb.py": "\"\"\"Deterministic, math-preserving prompt variants.\n\nOnly prose changes: text inside $...$, \\\\(...\\\\), \\\\[...\\\\] and every digit stay byte for byte,\nand a variant whose digit sequence or math spans differ from the original is rejected (the\noriginal is used in its place and the rejection is reported). The organisers' variants\n(typos, distract, rename, rephrase, domain, expert) are unknown; these are proxies.\n\"\"\"\n\nimport hashlib\nimport random\nimport re\n\nMATH = re.compile(r\"(\\$\\$.*?\\$\\$|\\$[^$]*\\$|\\\\\\(.*?\\\\\\)|\\\\\\[.*?\\\\\\])\", re.S)\nNEUTRAL = (\n    \"The problem was first posed in a regional mathematics circle.\",\n    \"Take your time and read every condition carefully.\",\n    \"This question appeared in a practice set for a competition.\",\n)\n\n\ndef _seed(text: str) -> int:\n    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], \"little\")\n\n\ndef _math_and_digits(text: str) -> tuple:\n    return tuple(MATH.findall(text)), re.sub(r\"\\D\", \"\", text)\n\n\ndef typos(text: str) -> str:\n    \"\"\"Swap two inner letters in about one prose word in eight.\"\"\"\n    rng = random.Random(_seed(text))\n    out = []\n    for part in MATH.split(text):\n        if MATH.fullmatch(part):\n            out.append(part)\n            continue\n        words = part.split(\" \")\n        for i, w in enumerate(words):\n            if len(w) > 4 and w.isalpha() and rng.random() < 0.125:\n                j = rng.randrange(1, len(w) - 2)\n                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]\n        out.append(\" \".join(words))\n    return \"\".join(out)\n\n\ndef distract(text: str) -> str:\n    \"\"\"Append one irrelevant sentence that carries no number.\"\"\"\n    return f\"{text.rstrip()} {NEUTRAL[_seed(text) % len(NEUTRAL)]}\"\n\n\nVARIANTS = {\"typos\": typos, \"distract\": distract}\n\n\ndef variants(text: str) -> tuple:\n    \"\"\"({name: text}, [rejected names]).\"\"\"\n    want = _math_and_digits(text)\n    out, rejected = {}, []\n    for name, fn in VARIANTS.items():\n        v = fn(text)\n        if _math_and_digits(v) != want:\n            rejected.append(name)\n            v = text\n        out[name] = v\n    return out, rejected\n", "features.py": "\"\"\"Prompt-only robustness features: one forward pass over a problem and its variants.\n\nNo generation. Per problem, the original and each variant from perturb.py run as one padded\nbatch; from it come eight dimensionless scalars:\n\n- nll: mean next-token NLL of the original prompt (how familiar the text is to the model);\n- log_tokens: log of the original prompt length;\n- dnll_<v>: the variant's mean NLL minus the original's;\n- dcos50_<v>, dcos75_<v>: 1 - cosine between mask-mean-pooled hidden states of the original\n  and the variant at 50% and 75% of the depth (same fractions for every architecture).\n\"\"\"\n\nimport math\nimport time\n\nimport torch\n\nfrom perturb import VARIANTS, variants\n\nSUFFIX = \"\\n\\nPlease reason step by step, and put your final answer within \\\\boxed{}.\"\nFRACTIONS = (0.5, 0.75)\nFEATURES = [\"nll\", \"log_tokens\"] + [f\"{k}_{v}\" for v in VARIANTS for k in (\"dnll\", \"dcos50\", \"dcos75\")]\nMAX_PROMPT_TOKENS = 2048\n\n\ndef prompt(tok, text: str) -> str:\n    msgs = [{\"role\": \"user\", \"content\": text + SUFFIX}]\n    try:\n        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)\n    except Exception:  # a tokenizer without a chat template\n        return text + SUFFIX\n\n\n@torch.no_grad()\ndef problem_features(model, tok, text: str) -> dict:\n    texts = {\"original\": text, **variants(text)[0]}\n    names = list(texts)\n    if tok.pad_token is None:\n        tok.pad_token = tok.eos_token\n    tok.padding_side = \"right\"\n    batch = tok([prompt(tok, texts[n]) for n in names], return_tensors=\"pt\", padding=True,\n                truncation=True, max_length=MAX_PROMPT_TOKENS, add_special_tokens=False)\n    batch = {k: v.to(model.device) for k, v in batch.items()}\n    out = model(**batch, output_hidden_states=True, use_cache=False)\n    mask = batch[\"attention_mask\"].float()\n    ids = batch[\"input_ids\"]\n    logp = torch.log_softmax(out.logits[:, :-1].float(), -1)\n    nll_tok = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)\n    nll = (nll_tok * mask[:, 1:]).sum(1) / mask[:, 1:].sum(1).clamp(min=1)\n    depth = len(out.hidden_states) - 1\n    pooled = {}\n    for f in FRACTIONS:\n        h = out.hidden_states[max(1, round(f * depth))].float()\n        pooled[f] = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)\n    row = {\"nll\": float(nll[0]), \"log_tokens\": math.log(float(mask[0].sum()))}\n    for i, n in enumerate(names[1:], 1):\n        row[f\"dnll_{n}\"] = float(nll[i] - nll[0])\n        for f in FRACTIONS:\n            cos = torch.nn.functional.cosine_similarity(pooled[f][0], pooled[f][i], dim=0)\n            row[f\"dcos{int(f * 100)}_{n}\"] = float(1 - cos)\n    return row\n\n\ndef extract(model, tok, problems: list, deadline: float = math.inf) -> list:\n    \"\"\"One feature dict per problem, None for problems reached after the deadline.\"\"\"\n    rows = []\n    for text in problems:\n        if time.monotonic() >= deadline:\n            rows.append(None)\n            continue\n        try:\n            rows.append(problem_features(model, tok, text))\n        except torch.cuda.OutOfMemoryError:\n            torch.cuda.empty_cache()\n            rows.append(None)\n    return rows\n", "trainset.py": "\"\"\"Which train-main-v2 file a script reads, and proof of which one it read.\n\nTwo files are known: \"pinned\" (the revision the report was built on, 82 rows) and \"1015\" (the file of\n2026-10-15, the same labelled rows plus nine more Skywork rows). A choice is one of those names or a\npath; the file's sha256 must match a known entry. AIMO_TRAIN sets the default when no flag is given.\n\"\"\"\n\nimport hashlib\nimport os\nfrom pathlib import Path\n\nROOT = Path(__file__).resolve().parent\nKNOWN = {\n    \"pinned\": (\"data/train-main-v2/train-00000-of-00001.parquet\",\n               \"bedfb551ea0f4d95d5e70e1df7cf61395ce2d36eeba2869aba6a90fc3c198824\"),\n    \"1015\": (\"data/train-main-v2/train-2026-10-15.parquet\",\n             \"e18eb44207e7987781ec2ff214793196038442bdc1466fdf5498f3e87ec58285\"),\n}\nENV = \"AIMO_TRAIN\"\n\n\ndef default():\n    return os.environ.get(ENV) or \"pinned\"\n\n\ndef resolve(choice=None, root=None):\n    \"\"\"Return (path, name, sha256) for a choice; the hash must belong to a known entry.\"\"\"\n    root = Path(root) if root else ROOT\n    choice = choice or default()\n    if choice in KNOWN:\n        path = root / KNOWN[choice][0]\n    else:\n        path = Path(choice)\n        if not path.is_absolute():\n            path = root / path\n    sha = hashlib.sha256(path.read_bytes()).hexdigest()\n    names = [n for n, (_, h) in KNOWN.items() if h == sha]\n    if not names:\n        raise ValueError(f\"train file {path} has sha256 {sha}, which matches no known entry\")\n    if choice in KNOWN and names[0] != choice:\n        raise ValueError(f\"train file {path} has sha256 {sha}, expected {KNOWN[choice][1]} for {choice!r}\")\n    return path, names[0], sha\n\n\ndef read(choice=None, root=None):\n    \"\"\"The chosen file as a DataFrame, with its name and sha256.\"\"\"\n    import pandas as pd\n\n    path, name, sha = resolve(choice, root)\n    return pd.read_parquet(path), name, sha\n\n\ndef provenance(name, sha):\n    return {\"train_file\": name, \"train_sha256\": sha}\n\n\ndef add_argument(ap):\n    ap.add_argument(\"--train\", default=None,\n                    help=f\"pinned, 1015 or a path to a known train file (default: ${ENV}, else pinned)\")\n    return ap\n\n\ndef strip_flag(argv):\n    \"\"\"Pull `--train X` or `--train=X` out of a positional argv; return (choice or None, rest).\"\"\"\n    rest, choice, i = [], None, 0\n    while i < len(argv):\n        if argv[i] == \"--train\":\n            choice, i = argv[i + 1], i + 2\n        elif argv[i].startswith(\"--train=\"):\n            choice, i = argv[i].split(\"=\", 1)[1], i + 1\n        else:\n            rest.append(argv[i])\n            i += 1\n    return choice, rest\n"}
HASHES = {"perturb.py": "fc0ed3885b6cc355f4ff492f95162475163b1582f6b15457df2cc786cfffb62f", "features.py": "989fd16d1b11605cb21d01fd8da60bd4dd3150d5955e42b9912cc856cf0966b5"}

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
