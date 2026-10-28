"""Training-only prior for the small track (v3).

Each (model_id, reasoning_effort) maps to a beta-binomial posterior robust rate computed offline by
`entry/fit_training_prior.py` from the organisers' `train-main-v2` file (sha256 and counts in
training-prior.json). A problem is robust when its configuration's posterior is at least 0.5; an
unknown configuration, or one with fewer than five labelled training rows, gets the track's global
training posterior. The small-track training data hold Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think and
DeepSeek-R1-0528-Qwen3-8B, all with effort "default"; any other small-track configuration, including
a known model with another effort, takes the global prior. No public-set label, problem
or score is used, the answer does not depend on the problem text, and the decision is never adapted to
the batch. The function reads only the bundled training-prior.json, never raises, and returns exactly
one bool per problem in input order. If the JSON is unreadable it returns FALLBACK, the global
training decision (training prevalence is above one half).
"""

import json
from pathlib import Path

TRACK = "small"
FALLBACK = True
THRESHOLD = 0.5


def _load():
    try:
        data = json.loads((Path(__file__).resolve().parent / "training-prior.json").read_text())
        track = data["tracks"][TRACK]
        decide_global = bool(track["global_posterior"] >= THRESHOLD)
        if track["selected_rule"] == "global":
            return {}, decide_global
        table = {(c["model_id"], c["reasoning_effort"]): bool(c["posterior"] >= THRESHOLD)
                 for c in track["configurations"] if not c["used_global"]}
        return table, decide_global
    except Exception:
        return {}, FALLBACK


_TABLE, _GLOBAL = _load()


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    try:
        n = len(list(problems))
    except Exception:
        return []
    try:
        decision = _TABLE.get((model_id, reasoning_effort), _GLOBAL)
    except Exception:
        decision = _GLOBAL
    return [bool(decision)] * n
