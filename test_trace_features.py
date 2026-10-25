"""Trace-sensitivity helpers on fixtures (no model needed).

  uv run --no-project --with pytest --with sympy python -m pytest test_trace_features.py -q
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sub"))
import invariance as I  # noqa: E402
import trace_features as T  # noqa: E402

GEOM = "Triangle $ABC$ has $AB = 20$, $AC = 13$, and $BC = 7$. Let $n$ be the area of $ABC$ and $m = 2$. Find $n + 3m$."


def test_variants_verify_and_condition_changes_the_problem():
    v = T.variants(GEOM)
    for fam in T.FAMILIES:
        if fam in v:
            assert v[fam][0] != GEOM
    assert "condition" in v
    new, edit = T.condition_change(GEOM)
    a, b, old, rep = edit
    assert int(rep) == int(old) + 1 and new != GEOM


def test_rename_maps_the_trace_and_word_letters_are_refused():
    assert T.map_trace("so n = 5 and n^2 = 25", "rename", {"from": "n", "to": "t"}) == "so t = 5 and t^2 = 25"
    assert T.map_trace("let a be 3", "rename", {"from": "a", "to": "t"}) is None
    assert T.map_trace("unchanged", "eqswap", {}) == "unchanged"
    # letters inside words and LaTeX commands stay
    assert T.map_trace("\\sin n and nine", "rename", {"from": "n", "to": "t"}) == "\\sin t and nine"


def test_reduce_features_and_missing_indicators():
    orig = [-1.0, -0.5, -3.0, -0.2, -2.0]
    fam_lp = {"constants": [-1.1, -0.5, -3.5, -0.2, -2.0], "eqswap": list(orig)}
    cos = {"constants": {0.5: [0.0, 0.1, 0.2, 0.0, 0.0], 0.75: [0.1] * 5}, "eqswap": {0.5: [0.0] * 5, 0.75: [0.0] * 5}}
    row = T.reduce(orig, fam_lp, cos, {"constants": [0.3]}, neg_lp=[-2.0, -1.5, -4.0, -1.2, -3.0])
    assert set(T.FEATURES) <= set(row)
    assert math.isclose(row["conf"], sum(orig) / 5)
    assert math.isclose(row["worst_dll"], 0.6 / 5)
    assert row["missing_frac"] == 0.5 and row["neg_missing"] == 0.0
    assert row["neg_ratio"] < 1  # the condition change moved the trace far more than the safe variants
    assert row["tail_conc"] > 0  # the changed tokens include the least likely ones
    none = T.reduce(orig, {}, {}, {}, None)
    assert none["missing_frac"] == 1.0 and none["neg_missing"] == 1.0 and math.isnan(none["mean_dll"])


def test_entity_strings():
    assert T.entity_strings("rename", {"from": "n", "to": "t"}) == ["t"]
    assert T.entity_strings("constants", {"edits": [(0, 2, "20", "(19+1)")]}) == ["20"]
