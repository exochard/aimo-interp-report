"""Leave-one-problem-out evaluation of the training-only prior on the 34- and 43-row sets.

  python entry/eval_training_prior.py --train PARQUET_1015 --pinned PARQUET_PINNED [--out FILE]

34 rows: small-track labelled rows of the pinned revision; 43 rows: those of the 2026-10-15 file. The
main track (gpt-oss-120b) has 37 labelled rows in both files. Every fit and every choice of the prior
strength sits inside the training folds. Gate: the prior is within one row of always-robust and of the
per-model majority overall and for each model; otherwise the track ships the global training majority.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_training_prior as fp  # noqa: E402


def table(ev):
    lines = [f"{'set':20} {'rows':>4} {'prior':>6} {'always':>7} {'per-model':>10}  gate"]
    for name, v in ev.items():
        s = v["scores"]
        lines.append(f"{name:20} {s['prior']['n']:4} {s['prior']['correct']:6} {s['always_robust']['correct']:7} "
                     f"{s['per_model_majority']['correct']:10}  {'pass' if v['gate_passed'] else 'FAIL'}")
        for m, (c, n) in s["prior"]["by_model"].items():
            lines.append(f"  {m:38} n={n:2} prior {c:2} always {s['always_robust']['by_model'][m][0]:2} "
                         f"per-model {s['per_model_majority']['by_model'][m][0]:2}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--pinned", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "training-prior-eval.json"))
    a = ap.parse_args(argv)
    prior, ev = fp.fit({"1015": a.train, "pinned": a.pinned})
    out = {"input_sha256": prior["input_sha256"], "pinned_sha256": prior["pinned_sha256"],
           "grid": fp.GRID, "evaluation": ev,
           "selected_rule": {t: v["selected_rule"] for t, v in prior["tracks"].items()},
           "chosen_prior_strength": {t: v["prior_strength"] for t, v in prior["tracks"].items()}}
    Path(a.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(table(ev))
    print("selected:", out["selected_rule"], "k:", out["chosen_prior_strength"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
