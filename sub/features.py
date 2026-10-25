"""Prompt-only robustness features: one forward pass over a problem and its variants.

No generation. Per problem, the original and each variant from perturb.py run as one padded
batch; from it come eight dimensionless scalars:

- nll: mean next-token NLL of the original prompt (how familiar the text is to the model);
- log_tokens: log of the original prompt length;
- dnll_<v>: the variant's mean NLL minus the original's;
- dcos50_<v>, dcos75_<v>: 1 - cosine between mask-mean-pooled hidden states of the original
  and the variant at 50% and 75% of the depth (same fractions for every architecture).
"""

import math
import time

import torch

from perturb import VARIANTS, variants

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
FRACTIONS = (0.5, 0.75)
FEATURES = ["nll", "log_tokens"] + [f"{k}_{v}" for v in VARIANTS for k in ("dnll", "dcos50", "dcos75")]
MAX_PROMPT_TOKENS = 2048


def prompt(tok, text: str) -> str:
    msgs = [{"role": "user", "content": text + SUFFIX}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:  # a tokenizer without a chat template
        return text + SUFFIX


@torch.no_grad()
def problem_features(model, tok, text: str) -> dict:
    texts = {"original": text, **variants(text)[0]}
    names = list(texts)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    batch = tok([prompt(tok, texts[n]) for n in names], return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_PROMPT_TOKENS, add_special_tokens=False)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model(**batch, output_hidden_states=True, use_cache=False)
    mask = batch["attention_mask"].float()
    ids = batch["input_ids"]
    logp = torch.log_softmax(out.logits[:, :-1].float(), -1)
    nll_tok = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
    nll = (nll_tok * mask[:, 1:]).sum(1) / mask[:, 1:].sum(1).clamp(min=1)
    depth = len(out.hidden_states) - 1
    pooled = {}
    for f in FRACTIONS:
        h = out.hidden_states[max(1, round(f * depth))].float()
        pooled[f] = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
    row = {"nll": float(nll[0]), "log_tokens": math.log(float(mask[0].sum()))}
    for i, n in enumerate(names[1:], 1):
        row[f"dnll_{n}"] = float(nll[i] - nll[0])
        for f in FRACTIONS:
            cos = torch.nn.functional.cosine_similarity(pooled[f][0], pooled[f][i], dim=0)
            row[f"dcos{int(f * 100)}_{n}"] = float(1 - cos)
    return row


def extract(model, tok, problems: list, deadline: float = math.inf) -> list:
    """One feature dict per problem, None for problems reached after the deadline."""
    rows = []
    for text in problems:
        if time.monotonic() >= deadline:
            rows.append(None)
            continue
        try:
            rows.append(problem_features(model, tok, text))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append(None)
    return rows
