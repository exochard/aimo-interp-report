"""Leave-one-group-out helpers of the probe fits, and the prompt-only feature code (synthetic rows, no data, no model).

  uv run --no-project --with pytest --with pandas --with scikit-learn --with numpy --with torch python -m pytest test_probe_fits.py -q
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sub"))
import fit_probe as FP  # noqa: E402
import fit_trace_probe as FT  # noqa: E402
import perturb as P  # noqa: E402

PROSE = "Find the number of positive integers $n$ below 100 such that $n^2 + 1$ is divisible by 5. Explain every step."


def frame():
    return pd.DataFrame({"problem": ["a", "a", "b", "c", "c", "c"], "y": [True, False, True, True, False, False]})


def test_oof_hands_each_group_a_training_set_without_it():
    seen = []

    def predict(train, test):
        seen.append((set(test.problem), set(train.problem)))
        return np.full(len(test), True)

    out = FP.oof(frame(), predict)
    assert out.dtype == bool and out.all() and len(out) == 6
    assert len(seen) == 3
    for held, rest in seen:
        assert len(held) == 1 and not held & rest and len(rest) == 2


def test_oof_writes_predictions_back_in_row_order():
    df = frame()
    out = FP.oof(df, lambda train, test: np.asarray(test.index % 2 == 0))
    assert list(out) == [i % 2 == 0 for i in range(6)]


def test_logo_matches_oof_and_honours_the_grouping_column():
    df = frame()
    df["fam"] = ["x", "x", "x", "y", "y", "y"]
    majority = lambda train, test: np.full(len(test), bool(train.y.mean() >= 0.5))  # noqa: E731
    assert (FT.logo(df, majority) == FP.oof(df, majority)).all()
    seen = []
    FT.logo(df, lambda train, test: seen.append(set(train.fam)) or np.zeros(len(test), bool), by="fam")
    assert seen == [{"y"}, {"x"}]


def test_variants_keep_math_and_digits_and_are_deterministic():
    v, rejected = P.variants(PROSE)
    assert rejected == [] and set(v) == set(P.VARIANTS)
    for text in v.values():
        assert P._math_and_digits(text) == P._math_and_digits(PROSE)
    assert P.variants(PROSE) == (v, rejected)
    assert v["distract"].startswith(PROSE.rstrip()) and v["distract"] != PROSE


def test_typos_leave_math_spans_untouched_and_change_only_prose():
    text = "Consider $\\alpha + \\beta$ where everything otherwise remains unchanged, therefore compute carefully."
    out = P.typos(text)
    assert "$\\alpha + \\beta$" in out
    assert sorted(out) == sorted(text)


def test_a_variant_that_breaks_the_digits_is_rejected_and_replaced(monkeypatch):
    monkeypatch.setitem(P.VARIANTS, "typos", lambda t: t + " 7")
    v, rejected = P.variants(PROSE)
    assert rejected == ["typos"] and v["typos"] == PROSE


def test_feature_names_cover_both_variants():
    F = pytest.importorskip("features")
    assert F.FEATURES[:2] == ["nll", "log_tokens"]
    assert len(F.FEATURES) == 2 + 3 * len(P.VARIANTS)
    assert len(set(F.FEATURES)) == len(F.FEATURES)


def test_prompt_falls_back_without_a_chat_template():
    F = pytest.importorskip("features")

    class Plain:
        def apply_chat_template(self, *a, **k):
            raise ValueError("no template")

    class Chat:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            return "<u>" + msgs[0]["content"] + "<a>"

    assert F.prompt(Plain(), "Q") == "Q" + F.SUFFIX
    assert F.prompt(Chat(), "Q") == "<u>Q" + F.SUFFIX + "<a>"


def test_extract_returns_none_for_problems_after_the_deadline():
    F = pytest.importorskip("features")
    assert F.extract(None, None, ["a", "b"], deadline=-math.inf) == [None, None]
