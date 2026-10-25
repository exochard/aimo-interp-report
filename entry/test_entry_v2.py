"""Guarded v2 entries: chunking, deadline, fallback, model cache (no model loads).

  CUDA_VISIBLE_DEVICES="" python3 -m pytest -q entry/test_entry_v2.py
"""

import importlib.util
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
TRACKS = {
    "main-hybrid-v2": dict(robust=("openai/gpt-oss-120b", "medium"), fallback=False,
                           other=("some/other", "low")),
    "small-hybrid-v2": dict(robust=("Qwen/Qwen3.5-4B", "high"), fallback=True,
                            other=("some/other", "low")),
}


def load(track):
    spec = importlib.util.spec_from_file_location(f"sol_{track.replace('-', '_')}",
                                                  HERE / track / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=list(TRACKS))
def sol(request):
    mod = load(request.param)
    mod.track = request.param
    mod.cfg = TRACKS[request.param]
    return mod


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def fake(sol, monkeypatch, calls, fail_on=None, clock=None, step=0.0):
    def predict(model_id, problem_texts):
        calls.append(list(problem_texts))
        if clock is not None:
            clock.t += step
        if fail_on is not None and len(calls) == fail_on:
            raise RuntimeError("boom")
        return [True] * len(problem_texts)
    monkeypatch.setattr(sol, "predict_robustness", predict)


def test_normal_path_chunks_of_eight(sol, monkeypatch):
    calls = []
    fake(sol, monkeypatch, calls)
    problems = [f"p{i}" for i in range(19)]
    out = sol.are_robust(DEEPSEEK, "medium", problems)
    assert out == [True] * 19
    assert [len(c) for c in calls] == [8, 8, 3]
    assert [p for c in calls for p in c] == problems


def test_chunk_size_never_above_eight(sol, monkeypatch):
    calls = []
    fake(sol, monkeypatch, calls)
    sol.are_robust(DEEPSEEK, "medium", ["x"] * 100)
    assert max(len(c) for c in calls) <= 8 and sum(len(c) for c in calls) == 100


def test_exception_on_second_chunk_falls_back(sol, monkeypatch):
    calls = []
    fake(sol, monkeypatch, calls, fail_on=2)
    out = sol.are_robust(DEEPSEEK, "medium", ["x"] * 20)
    fb = sol.cfg["fallback"]
    assert out == [True] * 8 + [fb] * 12
    assert len(calls) == 2  # no further calls after a failure


def test_past_deadline_falls_back(sol, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(sol, "_clock", clock)
    monkeypatch.setattr(sol, "_T0", 0.0)
    monkeypatch.setattr(sol, "DEADLINE_S", 100.0)
    calls = []
    fake(sol, monkeypatch, calls, clock=clock, step=60.0)
    out = sol.are_robust(DEEPSEEK, "medium", ["x"] * 40)
    # chunk 1 at t=0, chunk 2 at t=60, chunk 3 check at t=120 > 100
    assert len(calls) == 2
    assert out == [True] * 16 + [sol.cfg["fallback"]] * 24


def test_predicted_overrun_stops_before_chunk(sol, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(sol, "_clock", clock)
    monkeypatch.setattr(sol, "_T0", 0.0)
    monkeypatch.setattr(sol, "DEADLINE_S", 1e9)
    calls = []
    fake(sol, monkeypatch, calls, clock=clock, step=1000.0)
    out = sol.are_robust(DEEPSEEK, "medium", ["x"] * 40)
    # before chunk 2: 1000 + 1.5 * 1000 = 2500 <= 3300; before chunk 3: 2000 + 1500 = 3500 > 3300
    assert len(calls) == 2
    assert out == [True] * 16 + [sol.cfg["fallback"]] * 24


def test_clock_starts_before_kit_import(sol):
    src = (HERE / sol.track / "solution.py").read_text()
    assert src.index("_T0 = _clock()") < src.index("from uncertainty_profile")


def test_deadline_already_passed_makes_no_call(sol, monkeypatch):
    clock = Clock()
    clock.t = 5000.0
    monkeypatch.setattr(sol, "_clock", clock)
    monkeypatch.setattr(sol, "_T0", 0.0)
    calls = []
    fake(sol, monkeypatch, calls)
    out = sol.are_robust(DEEPSEEK, "medium", ["x"] * 5)
    assert calls == [] and out == [sol.cfg["fallback"]] * 5


def test_malformed_prediction_falls_back(sol, monkeypatch):
    monkeypatch.setattr(sol, "predict_robustness", lambda model_id, problem_texts: [True])
    out = sol.are_robust(DEEPSEEK, "medium", ["x"] * 3)
    assert out == [sol.cfg["fallback"]] * 3


def test_output_length_and_bool_type(sol, monkeypatch):
    import numpy as np
    monkeypatch.setattr(sol, "predict_robustness",
                        lambda model_id, problem_texts: [np.bool_(True)] * len(problem_texts))
    for n in (0, 1, 8, 9, 17):
        out = sol.are_robust(DEEPSEEK, "medium", ["x"] * n)
        assert type(out) is list and len(out) == n
        assert all(type(v) is bool for v in out)
    assert sol.are_robust("some/other", "low", ["x"] * 3) == [False] * 3


def test_non_deepseek_follows_group_rule(sol, monkeypatch):
    calls = []
    fake(sol, monkeypatch, calls)
    model, effort = sol.cfg["robust"]
    assert sol.are_robust(model, effort, ["a", "b"]) == [True, True]
    assert sol.are_robust(*sol.cfg["other"][:1], sol.cfg["other"][1], ["a"]) == [False]
    assert calls == []
    assert sol.ROBUST == {sol.cfg["robust"] if sol.track.startswith("main") else ("Qwen/Qwen3.5-4B", "*")}


def test_model_cache_loads_once_and_releases_at_end():
    mod = load("main-hybrid-v2")
    loads, releases = [], []
    inf = types.SimpleNamespace(
        load_model_and_tokenizer=lambda model_id, **kw: (loads.append(model_id), ("m", "t"))[1],
        release_model=lambda: releases.append(1),
    )
    cache = mod._install_model_cache(inf)
    assert inf.load_model_and_tokenizer(DEEPSEEK, local_files_only=True) == ("m", "t")
    assert inf.load_model_and_tokenizer(DEEPSEEK, local_files_only=True) == ("m", "t")
    inf.release_model()  # per-call release is deferred
    assert loads == [DEEPSEEK] and releases == []
    cache.close()
    assert releases == [1]


def test_small_marker_present():
    assert (HERE / "small-hybrid-v2" / "small.txt").exists()
    assert not (HERE / "main-hybrid-v2" / "small.txt").exists()
