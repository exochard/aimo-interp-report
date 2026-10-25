"""Train-file choice, and the producers and fitters that use it (tiny fixture parquets, no model, no data folder).

  uv run --no-project --with pytest --with pandas --with pyarrow --with scikit-learn --with numpy --with torch --with transformers python -m pytest test_trainset.py -q
"""

import hashlib
import json
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sub"))
sys.path.insert(0, str(HERE / "kaggle"))
import fit_activation_probe as FA  # noqa: E402
import fit_probe as FP  # noqa: E402
import fit_trace_probe as FT  # noqa: E402
import make_feat as MF  # noqa: E402
import run_activation_probe as RAP  # noqa: E402
import run_trace_pilot as RTP  # noqa: E402
import trainset  # noqa: E402
from features import FEATURES  # noqa: E402

QWEN, SKY = "Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B"


def rows(sky_problems):
    out = [{"model_id": QWEN, "problem": f"q{i}", "reasoning_effort": "low", "is_robust": bool(i % 2)} for i in range(3)]
    out += [{"model_id": SKY, "problem": f"s{i}", "reasoning_effort": "low", "is_robust": bool(i % 2)} for i in range(sky_problems)]
    return pd.DataFrame(out)


@pytest.fixture
def files(tmp_path, monkeypatch):
    """A 'pinned' file with 5 rows and a '1015' file with the same 5 plus 2 Skywork rows."""
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    rows(2).to_parquet(a)
    rows(4).to_parquet(b)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    monkeypatch.setattr(trainset, "KNOWN", {"pinned": (str(a), sha(a)), "1015": (str(b), sha(b))})
    monkeypatch.delenv(trainset.ENV, raising=False)
    return a, b


def test_resolve_names_paths_and_default(files, tmp_path, monkeypatch):
    a, b = files
    assert trainset.resolve()[1] == "pinned"
    assert trainset.resolve("1015") == (Path(str(b)), "1015", trainset.KNOWN["1015"][1])
    assert trainset.resolve(str(b))[1] == "1015"
    monkeypatch.setenv(trainset.ENV, "1015")
    assert trainset.resolve()[1] == "1015"
    assert trainset.resolve("pinned")[1] == "pinned"
    df, name, sha = trainset.read()
    assert name == "1015" and len(df) == 7 and sha == trainset.KNOWN["1015"][1]


def test_relative_path_is_taken_from_the_given_root(files, tmp_path):
    assert trainset.resolve("b.parquet", root=tmp_path)[1] == "1015"


def test_unknown_hash_is_an_error_naming_it(files, tmp_path):
    other = tmp_path / "other.parquet"
    rows(1).to_parquet(other)
    sha = hashlib.sha256(other.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=sha):
        trainset.resolve(str(other))


def test_named_choice_with_wrong_content_is_an_error(files, monkeypatch):
    a, b = files
    monkeypatch.setattr(trainset, "KNOWN", {"pinned": (str(b), trainset.KNOWN["pinned"][1]), "1015": trainset.KNOWN["1015"]})
    with pytest.raises(ValueError, match="expected"):
        trainset.resolve("pinned")


def test_strip_flag_keeps_positional_args():
    assert trainset.strip_flag(["rows.jsonl", "out.json"]) == (None, ["rows.jsonl", "out.json"])
    assert trainset.strip_flag(["rows.jsonl", "--train", "1015", "out.json"]) == ("1015", ["rows.jsonl", "out.json"])
    assert trainset.strip_flag(["--train=x", "r"]) == ("x", ["r"])


# --- fitters ---------------------------------------------------------------------------------------

def feature_file(tmp_path, kind):
    path = tmp_path / "feats.jsonl"
    with path.open("w") as fh:
        for r in rows(4).itertuples():
            fh.write(json.dumps({"model_id": r.model_id, "problem": r.problem, "reasoning_effort": r.reasoning_effort,
                                 "features": {k: 0.5 for k in kind}}) + "\n")
    return path


def test_fitters_read_the_chosen_file(files, tmp_path):
    f = feature_file(tmp_path, FEATURES)
    assert len(FP.table(f)) == 5 and len(FP.table(f, "pinned")) == 5
    assert len(FP.table(f, "1015")) == 7
    t = feature_file(tmp_path, FT.FEATURES)
    assert len(FT.table(t)) == 5 and len(FT.table(t, train="1015")) == 7


def test_calibration_load_follows_the_choice(files, tmp_path):
    import calibration as C
    f = feature_file(tmp_path, FEATURES)
    assert len(C.load(f)) == 5 and len(C.load(f, "1015")) == 7


def test_activation_fit_compares_the_manifest_with_the_chosen_file(files, tmp_path):
    a, b = files
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"complete": True, "label_sha256": trainset.KNOWN["pinned"][1], "completed_rows": 5, "models": {}}))
    with pytest.raises(ValueError, match="labels changed"):
        FA.evaluate(tmp_path, "1015")
    with pytest.raises(FileNotFoundError):  # past the hash check, the vectors are missing
        FA.evaluate(tmp_path)


# --- run_trace_pilot ---------------------------------------------------------------------------------

@pytest.fixture
def trace(monkeypatch):
    loaded = []
    monkeypatch.setattr(RTP, "load", lambda m: loaded.append(m) or (object(), object(), "bf16", ["A"]))
    monkeypatch.setattr(RTP, "extract", lambda model, tok, problems, effort, deadline: [{"x": 1.0} for _ in problems])
    for fn, val in (("max_memory_allocated", 0), ("empty_cache", None), ("reset_peak_memory_stats", None)):
        monkeypatch.setattr(RTP.torch.cuda, fn, lambda *a, val=val: val)
    return loaded


def keys(path):
    return [(r["model_id"], r["reasoning_effort"], r["problem"]) for r in map(json.loads, path.read_text().splitlines())]


def test_trace_default_writes_everything_and_a_train_record(files, trace, tmp_path):
    out = tmp_path / "rows.jsonl"
    RTP.main([str(out), "1"])
    assert len(keys(out)) == 5 and trace == [QWEN, SKY]
    assert json.loads((tmp_path / "rows.jsonl.train.json").read_text()) == {
        "train_file": "pinned", "train_sha256": trainset.KNOWN["pinned"][1]}


def test_trace_append_adds_only_missing_rows_and_skips_loaded_models(files, trace, tmp_path):
    out = tmp_path / "rows.jsonl"
    RTP.main([str(out), "1"])
    trace.clear()
    RTP.main([str(out), "1", "--train", "1015", "--append", "--models", SKY])
    assert trace == [SKY] and len(keys(out)) == 7 and len(set(keys(out))) == 7
    rec = json.loads((tmp_path / "rows.jsonl.train.json").read_text())
    assert rec["train_file"] == "1015" and rec["earlier"][0]["train_file"] == "pinned"
    trace.clear()
    RTP.main([str(out), "1", "--train", "1015", "--append"])
    assert trace == [] and len(keys(out)) == 7  # nothing missing: no model is loaded


def test_trace_models_filter_without_append(files, trace, tmp_path):
    out = tmp_path / "rows.jsonl"
    RTP.main([str(out), "1", "--models", QWEN])
    assert trace == [QWEN] and {k[0] for k in keys(out)} == {QWEN}


def test_trace_train_comes_from_the_environment(files, trace, tmp_path, monkeypatch):
    monkeypatch.setenv(trainset.ENV, "1015")
    out = tmp_path / "rows.jsonl"
    RTP.main([str(out), "1", "--models", SKY])
    assert len(keys(out)) == 4


# --- run_activation_probe ----------------------------------------------------------------------------

@pytest.fixture
def act(monkeypatch):
    loaded = []
    monkeypatch.setattr(RAP, "load_model", lambda m, root: loaded.append(m) or (None, None, {"revision": "r", "precision": "p"}))
    monkeypatch.setattr(RAP, "capture", lambda model, tok, problem, effort: (np.ones((2, 3)), {"layers": 2}))
    return loaded


def run_act(out, *args):
    RAP.main(["--out", str(out), *args])
    return json.loads((out / "manifest.json").read_text())


def test_activation_default_manifest_names_the_file(files, act, tmp_path):
    m = run_act(tmp_path / "o")
    assert m["label_sha256"] == trainset.KNOWN["pinned"][1] and m["train_file"] == "pinned"
    assert m["complete"] and m["completed_rows"] == 5 and "label_files" not in m
    assert len(list((tmp_path / "o").glob("*.npz"))) == 5 and act == [QWEN, SKY]
    with pytest.raises(FileExistsError):
        run_act(tmp_path / "o")


def test_activation_append_extracts_only_the_new_keys_and_records_both_files(files, act, tmp_path):
    out = tmp_path / "o"
    run_act(out)
    act.clear()
    m = run_act(out, "--train", "1015", "--append", "--only", SKY)
    assert act == [SKY]
    assert len(list(out.glob("*.npz"))) == 7 and len(json.loads((out / "rows.json").read_text())) == 7
    assert m["label_sha256"] == trainset.KNOWN["1015"][1] and m["complete"] and m["expected_rows"] == 7
    assert [f["train_file"] for f in m["label_files"]] == ["pinned", "1015"]
    assert set(m["models"]) == {QWEN, SKY}
    act.clear()
    run_act(out, "--train", "1015", "--append")
    assert act == []  # every npz exists: no model is loaded


def test_activation_only_filter(files, act, tmp_path):
    m = run_act(tmp_path / "o", "--only", SKY)
    assert act == [SKY] and m["completed_rows"] == 2 and not m["complete"]


# --- kaggle/make_feat.py driver ------------------------------------------------------------------------

@pytest.fixture
def driver(files, tmp_path, monkeypatch):
    run_py = tmp_path / "feat" / "run.py"
    MF.build(tmp_path / "feat")
    work = tmp_path / "work"
    work.mkdir()
    loaded = []
    feats = types.ModuleType("features")
    feats.FEATURES = ["nll"]
    feats.problem_features = lambda model, tok, text: {"nll": 1.0}
    pert = types.ModuleType("perturb")
    pert.variants = lambda t: ({}, [])
    monkeypatch.setitem(sys.modules, "features", feats)
    monkeypatch.setitem(sys.modules, "perturb", pert)
    import huggingface_hub
    import transformers

    def fail(*a, **k):
        raise AssertionError("no Hub download expected")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, m, **k: object()))

    class M:
        def eval(self):
            return self
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        classmethod(lambda cls, m, **k: loaded.append(m) or M()))
    monkeypatch.chdir(work)
    monkeypatch.setenv("SKIP_INSTALL", "1")
    monkeypatch.setenv("OUT", str(tmp_path / "out"))
    monkeypatch.setenv("MODELS", f"{QWEN},{SKY}")
    monkeypatch.setenv("ONLY_LABELLED", "1")
    (tmp_path / "out").mkdir()

    def run(*args):
        monkeypatch.setattr(sys, "argv", [str(run_py), *args])
        runpy.run_path(str(run_py), run_name="__main__")
        out = tmp_path / "out"
        return ([json.loads(x) for x in (out / "features.jsonl").read_text().splitlines()],
                json.loads((out / "meta.json").read_text()))
    return run, loaded


def test_make_feat_build_adds_the_helper_and_keeps_the_module_hashes(tmp_path):
    hashes = MF.build(tmp_path / "f")
    assert set(hashes) == {"perturb.py", "features.py"}
    assert '"trainset.py"' in (tmp_path / "f/run.py").read_text()


def test_make_feat_local_train_then_append_adds_only_missing_rows(driver):
    run, loaded = driver
    rowsout, meta = run("--train", "pinned")
    assert len(rowsout) == 5 and meta["train_file"] == "pinned"
    assert meta["train_sha256"] == trainset.KNOWN["pinned"][1]
    loaded.clear()
    rowsout, meta = run("--train", "1015", "--append", "--models", SKY)
    assert loaded == [SKY] and len(rowsout) == 7 and len({(r["model_id"], r["problem"]) for r in rowsout}) == 7
    assert meta["train_file"] == "1015" and meta["earlier_train"][0]["train_file"] == "pinned"
    assert set(meta["models"]) == {QWEN, SKY}
    loaded.clear()
    rowsout, _ = run("--train", "1015", "--append")
    assert loaded == [] and len(rowsout) == 7
