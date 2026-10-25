"""Group rule plus the kit's uncertainty profile for DeepSeek, guarded against the run limit.

Configurations in ROBUST are net robust on the public competition set; DeepSeek is judged per
problem from its own greedy trace (kit baseline, bundled artifact) in chunks, under a deadline
measured from import, and stops early when the slowest chunk so far would overrun the run
limit. After an error or once the deadline passes, the remaining DeepSeek
problems get FALLBACK. The function never raises and returns exactly one bool per problem.
"""

import time

_clock = time.monotonic
_T0 = _clock()  # before the kit import, so torch and transformers load time counts

try:
    from uncertainty_profile import inference as _inference
    from uncertainty_profile.inference import predict_robustness
except Exception:  # missing kit package: every DeepSeek problem takes the fallback
    _inference = None

    def predict_robustness(model_id, problem_texts):
        raise RuntimeError("uncertainty_profile unavailable")

ROBUST = {("openai/gpt-oss-120b", "medium")}
FALLBACK = False
PROFILED = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
CHUNK = 8
DEADLINE_S = 3000.0
RUN_LIMIT_S = 3600.0  # kit README: the limit covers the entire prediction run
MARGIN_S = 300.0


class _ModelCache:
    """Keeps the loaded model across predict_robustness calls; release_model waits for close()."""

    def __init__(self, inference, load, release):
        self._inference, self._load, self._release = inference, load, release
        self._loaded = {}

    def load(self, model_id, **kwargs):
        key = (model_id, tuple(sorted(kwargs.items())))
        if key not in self._loaded:
            self._loaded[key] = self._load(model_id, **kwargs)
        return self._loaded[key]

    def close(self):
        self._loaded.clear()
        self._inference.load_model_and_tokenizer = self._load
        self._inference.release_model = self._release
        self._release()


def _install_model_cache(inference):
    cache = _ModelCache(inference, inference.load_model_and_tokenizer, inference.release_model)
    inference.load_model_and_tokenizer = cache.load
    inference.release_model = lambda: None
    return cache


def _profile(problems: list[str]) -> list[bool]:
    out: list[bool] = []
    cache = _install_model_cache(_inference) if _inference is not None else None
    slowest = 0.0
    try:
        for i in range(0, len(problems), CHUNK):
            chunk = problems[i : i + CHUNK]
            try:
                elapsed = _clock() - _T0
                if elapsed > DEADLINE_S or elapsed + 1.5 * slowest > RUN_LIMIT_S - MARGIN_S:
                    break
                start = _clock()
                preds = predict_robustness(model_id=PROFILED, problem_texts=chunk)
                slowest = max(slowest, _clock() - start)
                if len(preds) != len(chunk):
                    break
                out.extend(bool(p) for p in preds)
            except Exception:
                break
    finally:
        if cache is not None:
            try:
                cache.close()
            except Exception:
                pass
    return out + [FALLBACK] * (len(problems) - len(out))


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    try:
        problems = list(problems)
        if model_id == PROFILED:
            return _profile(problems)
        robust = (model_id, reasoning_effort) in ROBUST or (model_id, "*") in ROBUST
        return [bool(robust)] * len(problems)
    except Exception:
        return [FALLBACK] * len(list(problems))
