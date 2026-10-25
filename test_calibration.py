"""Selection happens inside training problem groups (synthetic rows, no data directory).

  uv run --no-project --with pytest --with pandas --with scikit-learn --with numpy python -m pytest test_calibration.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "sub"))
import calibration as C  # noqa: E402


def frame(seed=0, flip_problem=None):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(6):
        for m, fam in enumerate(("a", "b", "c")):
            y = bool((p + m) % 2)
            if flip_problem == p:
                y = not y
            rows.append({"model_id": f"m{m}", "family": fam, "problem": f"p{p}", "y": y, "nll": rng.normal(),
                         "log_tokens": rng.normal(), **{c: rng.normal() for c in C.PROMPT[2:]}})
    return pd.DataFrame(rows)


def test_best_breaks_ties_toward_earlier_arm_and_smaller_grid():
    table = {(a, k, c): 5 for a in C.ARMS for k in C.GRID_K for c in C.GRID_C}
    assert C.best(table) == (next(iter(C.ARMS)), C.GRID_K[0], C.GRID_C[0])
    table[("ability+difficulty", 4.0, 1.0)] = 6
    assert C.best(table) == ("ability+difficulty", 4.0, 1.0)
    assert C.best(table, "ability")[0] == "ability"


def test_choice_for_a_fold_ignores_the_held_out_problem_labels():
    base, flipped = frame(), frame(flip_problem=2)
    _, _, a = C.outer(base, "problem", False)
    _, _, b = C.outer(flipped, "problem", False)
    fa = next(c for c in a if c["held_out"] == "p2")
    fb = next(c for c in b if c["held_out"] == "p2")
    assert (fa["arm"], fa["k"], fa["C"], fa["inner_correct"]) == (fb["arm"], fb["k"], fb["C"], fb["inner_correct"])


def test_every_fold_records_a_choice_and_prediction_per_row():
    df = frame()
    sel, arms, chosen = C.outer(df, "problem", False)
    assert len(sel) == len(df) and len(chosen) == df.problem.nunique()
    assert set(arms) == set(C.ARMS) and all(len(v) == len(df) for v in arms.values())
    assert all(c["inner_rows"] == len(df) - 3 for c in chosen)


def test_paired_sign_test():
    y = np.array([True] * 10)
    p, q = np.array([True] * 10), np.array([False] * 3 + [True] * 7)
    r = C.paired(y, p, q)
    assert (r["wins"], r["losses"]) == (3, 0) and r["p_two_sided"] == 0.25
    assert C.paired(y, p, p)["p_two_sided"] == 1.0
