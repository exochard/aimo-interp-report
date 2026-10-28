"""Training-only v3 entries: contract, fallback, isolation, provenance (no model loads, no network).

  python -m pytest -q entry/test_entry_v3.py
"""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
JSON = HERE / "training-prior.json"
TRAIN_SHA = "e18eb44207e7987781ec2ff214793196038442bdc1466fdf5498f3e87ec58285"
TRACKS = {"main-training-v3": ("main", "openai/gpt-oss-120b", "medium"),
          "small-training-v3": ("small", "Qwen/Qwen3.5-4B", "default")}
PARQUET_NAME = "train-2026-10-15.parquet"


def load(name, directory=None):
    path = (directory or HERE / name) / "solution.py"
    spec = importlib.util.spec_from_file_location(f"sol_{name.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=list(TRACKS))
def sol(request):
    mod = load(request.param)
    mod.name = request.param
    mod.track, mod.known_model, mod.known_effort = TRACKS[request.param]
    return mod


def global_decision(track):
    return json.loads(JSON.read_text())["tracks"][track]["global_posterior"] >= 0.5


def test_one_bool_per_problem_in_order(sol):
    for n in (1, 2, 7, 50):
        out = sol.are_robust(sol.known_model, sol.known_effort, [f"p{i}" for i in range(n)])
        assert len(out) == n and all(type(b) is bool for b in out)


def test_empty_and_duplicate_problems(sol):
    assert sol.are_robust(sol.known_model, sol.known_effort, []) == []
    assert len(sol.are_robust(sol.known_model, sol.known_effort, ["same"] * 5)) == 5


def test_decision_ignores_problem_text_and_order(sol):
    a = sol.are_robust(sol.known_model, sol.known_effort, ["x", "y", "z"])
    b = sol.are_robust(sol.known_model, sol.known_effort, ["z", "y", "x"])
    assert a == b and len(set(a)) == 1


def test_known_configuration_uses_its_posterior(sol):
    track = json.loads(JSON.read_text())["tracks"][sol.track]
    for c in track["configurations"]:
        want = c["posterior"] >= 0.5 if not c["used_global"] else global_decision(sol.track)
        if track["selected_rule"] == "global":
            want = global_decision(sol.track)
        assert sol.are_robust(c["model_id"], c["reasoning_effort"], ["p"]) == [want]


def test_unknown_configuration_gets_global(sol):
    want = [global_decision(sol.track)] * 3
    assert sol.are_robust("some/other", "low", ["a", "b", "c"]) == want
    assert sol.are_robust(sol.known_model, "no-such-effort", ["a", "b", "c"]) == want


def test_bad_inputs_do_not_raise(sol):
    assert sol.are_robust(None, None, ["a"]) == [global_decision(sol.track)]
    assert sol.are_robust("m", "e", None) == []
    assert sol.are_robust("m", "e", ("a", "b")) == [global_decision(sol.track)] * 2


def test_missing_json_falls_back_to_global_decision(sol, tmp_path):
    d = tmp_path / sol.name
    d.mkdir()
    shutil.copy(HERE / sol.name / "solution.py", d / "solution.py")
    mod = load(sol.name, d)
    assert mod.are_robust(sol.known_model, sol.known_effort, ["a", "b"]) == [mod.FALLBACK] * 2
    assert mod.FALLBACK is global_decision(sol.track)


def test_corrupt_json_falls_back(sol, tmp_path):
    d = tmp_path / sol.name
    d.mkdir()
    shutil.copy(HERE / sol.name / "solution.py", d / "solution.py")
    (d / "training-prior.json").write_text("{not json")
    assert load(sol.name, d).are_robust("m", "e", ["a"]) == [global_decision(sol.track)]


def test_no_torch_or_network_imports(sol):
    src = (HERE / sol.name / "solution.py").read_text()
    imported = set(re.findall(r"^\s*(?:from|import)\s+([\w.]+)", src, re.M))
    banned = {"torch", "transformers", "requests", "urllib", "urllib3", "http", "socket", "httpx",
              "huggingface_hub", "numpy", "pandas", "joblib", "sklearn", "subprocess", "os"}
    assert not {m.split(".")[0] for m in imported} & banned
    assert imported <= {"json", "pathlib"}


def test_json_hash_matches_training_parquet():
    data = json.loads(JSON.read_text())
    assert data["input_sha256"] == TRAIN_SHA
    candidates = [Path(p) / "train-main-v2" / PARQUET_NAME
                  for p in (os.environ.get("AIMO_DATA_DIR"), HERE.parent / "data",
                            Path.home() / "Projects/Athlon/contests/aimo-interp/data") if p]
    parquet = next((p for p in candidates if p.exists()), None)
    if parquet is None:
        pytest.skip("training parquet not on this machine; the pinned hash constant was checked above")
    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == data["input_sha256"]


def test_json_has_no_problem_text():
    data = json.loads(JSON.read_text())
    assert set(data) == {"input_sha256", "input_file", "pinned_sha256", "grid", "min_n", "threshold", "tracks"}
    assert max(len(s) for s in re.findall(r'"([^"]*)"', JSON.read_text())) < 80


def test_source_has_no_public_set_text_or_ids():
    pd = pytest.importorskip("pandas")
    roots = [p for p in (os.environ.get("AIMO_DATA_DIR"), HERE.parent / "data",
                         Path.home() / "Projects/Athlon/contests/aimo-interp/data") if p]
    files = [Path(r) / d / "validation-00000-of-00001.parquet" for r in roots
             for d in ("aimo-interp-challenge-sample-full", "val-sample")]
    files = [f for f in files if f.exists()]
    if not files:
        pytest.skip("public sample parquets not on this machine (REPRODUCE.md section 2 lists them)")
    texts = [HERE / n / "solution.py" for n in TRACKS] + [JSON]
    blob = "\n".join(t.read_text() for t in texts).lower()
    needles = set()
    for f in files:
        df = pd.read_parquet(f)
        for t in df.original_problem.dropna().unique():
            norm = re.sub(r"\s+", " ", t).strip().lower()
            needles.add(norm[:60])
            needles.add(hashlib.sha256(t.encode()).hexdigest())
            needles.add(hashlib.sha256(t.encode()).hexdigest()[:16])
        needles.update(str(i) for i in df.problem_id.dropna().unique() if len(str(i)) >= 6)
    assert needles and not [n for n in needles if n in blob]


def test_extracted_zip_runs_from_another_cwd(tmp_path):
    if shutil.which("zip") is None:
        pytest.skip("zip binary not available")
    dist = tmp_path / "dist"
    env = dict(os.environ, OUT_DIR=str(dist))
    subprocess.run(["bash", str(HERE / "build.sh"), "v3"], check=True, env=env, capture_output=True)
    for name, (track, model, effort) in TRACKS.items():
        with zipfile.ZipFile(dist / f"{name}.zip") as z:
            # the kit requires small.txt at the root of a small-track zip and forbids it in a main-track one
            marker = ["small.txt"] if track == "small" else []
            assert sorted(z.namelist()) == sorted(["solution.py", "training-prior.json"] + marker)
            out = tmp_path / f"x-{name}"
            z.extractall(out)
        code = ("import importlib.util as u,sys;s=u.spec_from_file_location('sol',sys.argv[1]+'/solution.py');"
                "m=u.module_from_spec(s);s.loader.exec_module(m);"
                "print(m.are_robust(sys.argv[2],sys.argv[3],['a','b','c']))")
        res = subprocess.run([sys.executable, "-I", "-c", code, str(out), model, effort],
                             cwd=tmp_path, capture_output=True, text=True, check=True)
        cfg = {(c["model_id"], c["reasoning_effort"]): c for c in
               json.loads(JSON.read_text())["tracks"][track]["configurations"]}[(model, effort)]
        assert res.stdout.strip() == str([cfg["posterior"] >= 0.5] * 3)


def test_zip_build_is_deterministic(tmp_path):
    if shutil.which("zip") is None:
        pytest.skip("zip binary not available")
    hashes = []
    for i in range(2):
        out = tmp_path / str(i)
        subprocess.run(["bash", str(HERE / "build.sh"), "v3"], check=True,
                       env=dict(os.environ, OUT_DIR=str(out)), capture_output=True)
        hashes.append([hashlib.sha256((out / f"{n}.zip").read_bytes()).hexdigest() for n in TRACKS])
    assert hashes[0] == hashes[1]
