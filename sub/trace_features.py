"""Counterfactual trace sensitivity: one short reasoning prefix, teacher-forced under verified variants.

Per problem:

1. Generate a deterministic 256-token reasoning prefix from the original prompt (the model's own
   chat template and reasoning effort). The prefix is a probe, not a solution.
2. Take the first verified variant of up to four answer-preserving families (variable renaming,
   equivalent numeric expression, equation-side swap, reordering of independent givens), and one
   condition change that alters the problem (an integer constant plus one) as a negative control.
   A rename also renames the same letter in the trace; a trace whose renamed letter is also an
   English word ("a", "I") drops that family instead of teacher-forcing a wrong symbol.
3. Teacher-force the trace after the original prompt and after each variant prompt, in one padded
   batch. Per trace token: log-likelihood under each prompt, and at 50% and 75% of the depth the
   hidden state, captured by forward hooks for trace positions only.
4. Reduce to at most twelve fixed features (FEATURES). Activations are released per problem; no
   full-network hidden states or full-vocabulary distributions are kept.

The ratio feature compares answer-preserving sensitivity with the condition change's: a model
equally insensitive to both is not shown robust. Missing families and a missing control are
explicit features, never silently imputed as zero sensitivity.
"""

import math
import re
import time

from invariance import _number_sites, _apply_edits, constants, eqswap, rename, reorder, verify

PREFIX_TOKENS = 256
FRACTIONS = (0.5, 0.75)
FAMILIES = ("rename", "constants", "eqswap", "reorder")
TAIL = 0.2  # share of least-likely trace tokens the concentration feature looks at
FEATURES = ["conf", "mean_dll", "worst_dll", "var_dll", "tail_conc", "dcos50_mean", "dcos75_q90",
            "dcos_entity", "neg_ratio", "neg_missing", "missing_frac"]
_WORDLIKE = {"a", "A", "I"}


def condition_change(text: str):
    """Negative control: one integer constant in a math span raised by one. Returns (text, edit) or None."""
    sites = [(a, b, v) for a, b, v in _number_sites(text) if v >= 2]
    if not sites:
        return None
    a, b, v = sites[len(sites) // 2]
    new = _apply_edits(text, [(a, b, str(v + 1))])
    return (new, (a, b, str(v), str(v + 1))) if new != text else None


def variants(text: str) -> dict:
    """{family: (variant text, meta)} for the first verified variant of each family, plus "condition"."""
    out = {}
    makers = {"rename": lambda t: rename(t, "rename_a"), "constants": lambda t: constants(t, "const_sum"),
              "eqswap": eqswap, "reorder": reorder}
    for fam in FAMILIES:
        try:
            p = makers[fam](text)
        except Exception:  # a transform that cannot parse the problem is a missing family
            p = None
        if p is not None and verify(text, p):
            out[fam] = (p.text, p.meta)
    neg = condition_change(text)
    if neg:
        out["condition"] = (neg[0], {"edits": [neg[1]]})
    return out


def map_trace(trace: str, family: str, meta: dict):
    """The trace as it would read under the variant, or None when it cannot be mapped safely."""
    if family != "rename":
        return trace
    old, new = meta["from"], meta["to"]
    if old in _WORDLIKE:
        return None
    return re.sub(r"(?<![A-Za-z\\])" + re.escape(old) + r"(?![A-Za-z])", new, trace)


def entity_strings(family: str, meta: dict) -> list:
    """Text pieces of the trace that name what the variant changed."""
    if family == "rename":
        return [meta["to"]]
    return [e[2] for e in meta.get("edits", []) if re.search(r"\d", str(e[2]))]


def _quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def reduce(orig_lp: list, fam_lp: dict, fam_cos: dict, fam_entity_cos: dict, neg_lp) -> dict:
    """Fixed features from per-token trace log-likelihoods and cosine changes.

    orig_lp: trace-token log-probs after the original prompt; fam_lp[f]: the same after variant f
    (aligned token for token, or a different length after a rename); fam_cos[f]: {fraction: [1 - cos]}
    per aligned token; fam_entity_cos[f]: 1 - cos at 75% depth on tokens naming changed entities.
    """
    n = len(orig_lp)
    row = {"conf": sum(orig_lp) / max(1, n)}
    dlls, pooled = [], []
    order = sorted(range(n), key=lambda i: orig_lp[i])[:max(1, int(TAIL * n))]
    tail_share = []
    for f, lp in fam_lp.items():
        if len(lp) == n:
            d = [b - a for a, b in zip(orig_lp, lp)]
            pooled += d
            tot = sum(abs(x) for x in d)
            tail_share.append(sum(abs(d[i]) for i in order) / tot if tot > 0 else 0.0)
            dlls.append(abs(sum(d) / max(1, n)))
        else:  # renamed trace: length-normalised totals only
            dlls.append(abs(sum(lp) / max(1, len(lp)) - row["conf"]))
    row["mean_dll"] = sum(dlls) / len(dlls) if dlls else float("nan")
    row["worst_dll"] = max(dlls) if dlls else float("nan")
    if pooled:
        mu = sum(pooled) / len(pooled)
        row["var_dll"] = sum((x - mu) ** 2 for x in pooled) / len(pooled)
    else:
        row["var_dll"] = float("nan")
    row["tail_conc"] = sum(tail_share) / len(tail_share) if tail_share else float("nan")
    c50 = [x for f in fam_cos for x in fam_cos[f].get(0.5, [])]
    c75 = [x for f in fam_cos for x in fam_cos[f].get(0.75, [])]
    row["dcos50_mean"] = sum(c50) / len(c50) if c50 else float("nan")
    row["dcos75_q90"] = _quantile(c75, 0.9)
    ent = [x for f in fam_entity_cos for x in fam_entity_cos[f]]
    row["dcos_entity"] = sum(ent) / len(ent) if ent else float("nan")
    if neg_lp is not None and len(neg_lp) == n and dlls:
        neg = abs(sum(b - a for a, b in zip(orig_lp, neg_lp)) / max(1, n))
        row["neg_ratio"] = row["mean_dll"] / (neg + 1e-3)
        row["neg_missing"] = 0.0
    else:
        row["neg_ratio"], row["neg_missing"] = float("nan"), 1.0
    row["missing_frac"] = 1.0 - len(fam_lp) / len(FAMILIES)
    return row


# ---- model side ---------------------------------------------------------------------------

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def chat(tok, text: str, effort: str) -> str:
    msgs = [{"role": "user", "content": text + SUFFIX}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       reasoning_effort=effort if effort != "default" else None)
    except Exception:  # a tokenizer without a chat template
        return text + SUFFIX


def _layers(model):
    for path in ("model.layers", "model.model.layers", "transformer.h", "model.language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("no decoder layer list found")


def problem_features(model, tok, text: str, effort: str = "default") -> dict:
    import torch

    prompt = chat(tok, text, effort)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=PREFIX_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    trace = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    rows = {"original": (prompt, trace, [])}
    for fam, (vtext, meta) in variants(text).items():
        t = map_trace(trace, fam, meta)
        if t is not None:
            rows[fam] = (chat(tok, vtext, effort), t, entity_strings(fam, meta))
    names = list(rows)
    p_ids = [tok(rows[n][0], add_special_tokens=False)["input_ids"] for n in names]
    t_ids = [tok(rows[n][1], add_special_tokens=False)["input_ids"] for n in names]
    seqs = [p + t for p, t in zip(p_ids, t_ids)]
    width = max(map(len, seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=model.device)
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=model.device)
    layers = _layers(model)
    picks = {f: layers[max(0, round(f * len(layers)) - 1)] for f in FRACTIONS}
    caught = {}

    def hook(f):
        def run(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            caught[f] = [h[i, len(p_ids[i]):len(seqs[i])].float().cpu() for i in range(len(seqs))]
        return run

    handles = [picks[f].register_forward_hook(hook(f)) for f in FRACTIONS]
    try:
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        lps = []
        for i in range(len(seqs)):
            a, b = len(p_ids[i]), len(seqs[i])
            lp = torch.log_softmax(logits[i, a - 1:b - 1].float(), -1)
            lps.append(lp.gather(-1, ids[i, a:b, None]).squeeze(-1).cpu().tolist())
        del logits
    finally:
        for h in handles:
            h.remove()
    orig = lps[0]
    fam_lp, fam_cos, fam_ent, neg = {}, {}, {}, None
    for i, n in enumerate(names[1:], 1):
        if n == "condition":
            neg = lps[i]
            continue
        fam_lp[n] = lps[i]
        if len(t_ids[i]) == len(t_ids[0]):
            fam_cos[n] = {f: (1 - torch.nn.functional.cosine_similarity(caught[f][0], caught[f][i], dim=-1)).tolist()
                          for f in FRACTIONS}
            pieces = rows[n][2]
            toks = [tok.decode([t]) for t in t_ids[0]]
            near = [k for k, s in enumerate(toks) if any(p and p.strip() and p.strip() in s for p in pieces)]
            fam_ent[n] = [fam_cos[n][0.75][k] for k in near]
    caught.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    row = reduce(orig, fam_lp, fam_cos, fam_ent, neg)
    row["trace_tokens"] = len(t_ids[0])
    return row


def extract(model, tok, problems: list, effort: str = "default", deadline: float = math.inf) -> list:
    """One feature dict per problem; None for a problem reached after the deadline or out of memory."""
    import torch

    out = []
    for text in problems:
        if time.monotonic() >= deadline:
            out.append(None)
            continue
        try:
            out.append(problem_features(model, tok, text, effort))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out.append(None)
    return out
