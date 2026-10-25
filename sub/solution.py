"""Not part of any submission: it expects a probe.json that was never built.

AIMO Interpretability entry sketch: prompt-only perturbation-sensitivity predictor.

are_robust loads the evaluated model offline, runs one forward pass per problem over the
original and two math-safe variants (features.py), and applies a logistic artifact fit on
train-main-v2 (probe.json). One monotonic deadline covers the whole process: 600 s stay in
reserve, model work stops at 2,700 s in all, and each call gets an equal share of what is
left across the model groups not yet seen, so a late group still gets its turn. A problem the
budget does not reach gets the artifact's per-model fallback; the list is always complete.
"""

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

START = time.monotonic()
MODEL_BUDGET_S = 2700
EXPECTED_GROUPS = 4
_STATE = {"seen": set(), "model_id": None, "model": None, "tok": None}

with open(os.path.join(HERE, "probe.json")) as fh:
    ARTIFACT = json.load(fh)


def _fallback(model_id: str) -> bool:
    return bool(ARTIFACT["fallback"].get(model_id, ARTIFACT["default"]))


def _predict(row: dict) -> bool:
    z = ARTIFACT["intercept"]
    for name, mean, scale, coef in zip(ARTIFACT["features"], ARTIFACT["mean"], ARTIFACT["scale"], ARTIFACT["coef"]):
        z += coef * (row[name] - mean) / scale
    return 1 / (1 + math.exp(-z)) >= ARTIFACT["threshold"]


def _load(model_id: str):
    if _STATE["model_id"] == model_id:
        return _STATE["model"], _STATE["tok"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _STATE.update(model=None, tok=None, model_id=None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto",
                                                 local_files_only=True).eval()
    _STATE.update(model=model, tok=tok, model_id=model_id)
    return model, tok


def _call_deadline(model_id: str) -> float:
    _STATE["seen"].add(model_id)
    left = START + MODEL_BUDGET_S - time.monotonic()
    groups_left = max(1, EXPECTED_GROUPS - len(_STATE["seen"]) + 1)
    return time.monotonic() + max(0.0, left / groups_left)


def are_robust(model_id: str, reasoning_effort: str = "default", problems: list = ()) -> list:
    problems = list(problems)
    preds = [_fallback(model_id)] * len(problems)
    if model_id not in ARTIFACT["models"] or not problems:
        return [bool(p) for p in preds]
    deadline = _call_deadline(model_id)
    try:
        from features import extract
        model, tok = _load(model_id)
        rows = extract(model, tok, problems, deadline)
        preds = [_predict(r) if r else p for r, p in zip(rows, preds)]
    except Exception as e:  # any failure returns the fallback for every problem
        print(f"[aimo] {model_id}: {type(e).__name__}: {e}", file=sys.stderr)
    return [bool(p) for p in preds]
