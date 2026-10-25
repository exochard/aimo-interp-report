"""Answer flips under answer-preserving perturbations, the quantity the robustness label counts.

A configuration is labelled robust when the model's accuracy does not decay under perturbed
versions of the problem. Instead of features of the prompt or of a fixed reasoning prefix, this
measures the answer itself: the model solves the original and a few perturbed copies K times
each, and the flip rate is how often a perturbed copy's answers disagree with the original's
majority answer. Perturbations are deterministic and need no second model: letter-swap typos
outside math, an irrelevant distractor sentence, and person names swapped.

  python answer_flips.py generate --model-dir DIR --model-id ID --rows rows.parquet --out flips.jsonl
  python answer_flips.py evaluate --flips flips.jsonl [...] --out verdict.json
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
DISTRACTORS = (
    "Note that the building where this problem was written has 7 floors.",
    "A student solving this had 3 pencils and 12 sheets of paper on the desk.",
)
NAMES = ("Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "John", "Mary", "Tom", "Sarah")
SWAP = ("Rosa", "Ivan", "Leila", "Omar", "Nadia", "Pablo", "Yuki", "Kofi", "Ingrid", "Tariq", "Mei", "Anika")


def _outside_math(text):
    """Spans of text outside $...$ and \\( \\) / \\[ \\] math."""
    spans, pos = [], 0
    for m in re.finditer(r"\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]", text, flags=re.S):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    return spans


def typos(text, seed, n=2):
    """Swap two inner letters in up to n words of at least 5 letters, outside math."""
    rng = random.Random(seed)
    words = [(a + m.start(), a + m.end()) for a, b in _outside_math(text)
             for m in re.finditer(r"[A-Za-z]{5,}", text[a:b])]
    out = list(text)
    for s, e in rng.sample(words, min(n, len(words))):
        i = rng.randrange(s + 1, e - 2)
        out[i], out[i + 1] = out[i + 1], out[i]
    return "".join(out)


def distract(text, k):
    return text.rstrip() + " " + DISTRACTORS[k % len(DISTRACTORS)]


def rename(text):
    """Swap known first names; None when the problem names nobody."""
    hits = [n for n in NAMES if re.search(r"\b%s\b" % n, text)]
    if not hits:
        return None
    for n in hits:
        text = re.sub(r"\b%s\b" % n, SWAP[NAMES.index(n)], text)
    return text


def variants(problem, seed=0):
    out = {"typos": typos(problem, seed), "distract": distract(problem, seed)}
    r = rename(problem)
    if r is not None:
        out["rename"] = r
    return out


def boxed(text):
    """Content of the last \\boxed{...}, braces matched; None when there is none."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        k += 1
    if depth:
        return None
    ans = text[j:k - 1]
    ans = re.sub(r"\\[,!;: ]|\\left|\\right|\s|\$", "", ans).rstrip(".")
    ans = re.sub(r"^\\text\{(.*)\}$", r"\1", ans)
    return ans.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")


def flip_features(answers):
    """answers: {"original": [...], variant: [...]} (None for no answer). Features of one row."""
    orig = [a for a in answers["original"] if a is not None]
    major = Counter(orig).most_common(1)[0][0] if orig else None
    agree = {}
    for name, ans in answers.items():
        if name == "original":
            continue
        agree[name] = (sum(a == major for a in ans) / len(ans)) if (ans and major is not None) else 0.0
    n_all = sum(len(v) for v in answers.values())
    return dict(
        orig_consistency=(Counter(orig).most_common(1)[0][1] / len(answers["original"])) if orig else 0.0,
        flip_max=1.0 - min(agree.values()) if agree else 0.0,
        flip_mean=1.0 - sum(agree.values()) / len(agree) if agree else 0.0,
        no_answer=sum(a is None for v in answers.values() for a in v) / max(1, n_all),
    )


FORCE = "\n</think>\n\nThe final answer is \\boxed{"


def force_answers(llm, prompts, texts, max_tokens=24):
    """Samples that ran out of budget before a \\boxed{} answer get the reasoning closed and the
    answer decoded greedily, so every sample yields the answer the model had committed to."""
    from vllm import SamplingParams
    answers = [[boxed(t) for t in ts] for ts in texts]
    todo = [(i, j) for i, ts in enumerate(texts) for j, t in enumerate(ts) if answers[i][j] is None]
    if todo:
        cont = [prompts[i] + texts[i][j].split("</think>")[0] + FORCE for i, j in todo]
        outs = llm.generate(cont, SamplingParams(n=1, temperature=0.0, max_tokens=max_tokens))
        for (i, j), o in zip(todo, outs):
            answers[i][j] = boxed("\\boxed{" + o.outputs[0].text)
    return answers


def generate(a):
    import pandas as pd
    from vllm import LLM, SamplingParams
    rows = pd.read_parquet(a.rows)
    rows = rows[rows.model_id == a.model_id]
    if a.limit:
        rows = rows.head(a.limit)
    kw = dict(model=a.model_dir, max_model_len=a.max_tokens + 1024, gpu_memory_utilization=0.88,
              enforce_eager=True,  # torch.compile of the FP8 path fails on Ampere (vLLM 0.30)
              seed=0, trust_remote_code=False)
    if a.quantize:
        kw["quantization"] = "fp8"  # weight-only FP8 (Marlin) on Ampere; vLLM 0.30 has no bitsandbytes
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=a.k, temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, seed=0)
    prompts, index = [], []
    for r in rows.itertuples():
        texts = {"original": r.problem, **variants(r.problem)}
        for name, t in texts.items():
            msg = [{"role": "user", "content": PROMPT.format(problem=t)}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            index.append((r.key, name))
    outs = llm.generate(prompts, sp)
    texts = [[c.text for c in o.outputs] for o in outs]
    answers = force_answers(llm, prompts, texts) if a.force else [[boxed(t) for t in ts] for ts in texts]
    by_row = {}
    for (key, name), ans in zip(index, answers):
        by_row.setdefault(key, {})[name] = ans
    with open(a.out, "w") as f:
        for r in rows.itertuples():
            ans = by_row[r.key]
            f.write(json.dumps(dict(key=r.key, model_id=r.model_id, problem_id=r.problem_id, robust=bool(r.robust),
                                    answers=ans, **flip_features(ans))) + "\n")


def generate_hf(a):
    """Transformers fallback for cards vLLM cannot quantize on: 4-bit NF4 weights, left-padded
    batches of prompts, K samples each."""
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    rows = pd.read_parquet(a.rows)
    rows = rows[rows.model_id == a.model_id]
    if a.limit:
        rows = rows.head(a.limit)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, quantization_config=q, device_map="cuda",
                                                 dtype=torch.bfloat16).eval()
    torch.manual_seed(0)
    jobs = []
    for r in rows.itertuples():
        for name, t in {"original": r.problem, **variants(r.problem)}.items():
            msg = [{"role": "user", "content": PROMPT.format(problem=t)}]
            jobs.append((r.key, name, tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)))
    by_row = {}
    for i in range(0, len(jobs), a.batch):
        chunk = jobs[i:i + a.batch]
        enc = tok([c[2] for c in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95, max_new_tokens=a.max_tokens,
                                 num_return_sequences=a.k, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, (key, name, _) in enumerate(chunk):
            by_row.setdefault(key, {})[name] = [boxed(t) for t in texts[j * a.k:(j + 1) * a.k]]
        print("batch", i // a.batch + 1, "of", -(-len(jobs) // a.batch), flush=True)
    with open(a.out, "w") as f:
        for r in rows.itertuples():
            ans = by_row[r.key]
            f.write(json.dumps(dict(key=r.key, model_id=r.model_id, problem_id=r.problem_id, robust=bool(r.robust),
                                    answers=ans, **flip_features(ans))) + "\n")


def evaluate(a):
    """Problem-held-out choice of feature and threshold against the majority class."""
    rows = [json.loads(line) for p in a.flips for line in open(p)]
    feats = ("flip_max", "flip_mean", "orig_consistency", "no_answer")
    problems = sorted({r["problem_id"] for r in rows})
    y = [r["robust"] for r in rows]

    def fit(train):
        best = None
        for f in feats:
            for t in sorted({r[f] for r in train}):
                for sign in (1, -1):  # sign 1: robust when feature <= t
                    acc = sum(((r[f] <= t) if sign == 1 else (r[f] > t)) == r["robust"] for r in train)
                    if best is None or acc > best[0]:
                        best = (acc, f, t, sign)
        maj = Counter(r["robust"] for r in train).most_common(1)[0][0]
        return best, maj

    correct = maj_correct = 0
    for p in problems:
        train = [r for r in rows if r["problem_id"] != p]
        test = [r for r in rows if r["problem_id"] == p]
        (_, f, t, sign), maj = fit(train)
        for r in test:
            pred = (r[f] <= t) if sign == 1 else (r[f] > t)
            correct += pred == r["robust"]
            maj_correct += maj == r["robust"]
    full, _ = fit(rows)
    pos = [r for r in rows if r["robust"]]
    neg = [r for r in rows if not r["robust"]]

    def auc(f):
        pairs = [(x[f] < z[f]) + 0.5 * (x[f] == z[f]) for x in pos for z in neg]
        return round(sum(pairs) / len(pairs), 3) if pairs else None
    verdict = dict(n=len(rows), robust_rate=round(sum(y) / len(y), 3), heldout_correct=correct,
                   majority_correct=maj_correct, gate_three_over_majority=correct >= maj_correct + 3,
                   full_rule=dict(feature=full[1], threshold=full[2], sign=full[3]),
                   auc_lower_is_robust={f: auc(f) for f in feats})
    Path(a.out).write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model-dir", required=True)
    g.add_argument("--model-id", required=True)
    g.add_argument("--rows", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--k", type=int, default=3)
    g.add_argument("--max-tokens", type=int, default=3072)
    g.add_argument("--quantize", action="store_true")
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--no-force", dest="force", action="store_false", help="keep unfinished samples unanswered")
    e = sub.add_parser("evaluate")
    e.add_argument("--flips", nargs="+", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "evaluate":
        evaluate(a)
    else:
        (generate_hf if a.backend == "hf" else generate)(a)
