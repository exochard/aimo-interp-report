"""Forest plot of each signal's difference against the majority baseline (Section 4).

  python report/forest.py [report/intervals.json] [report/forest.png]

Reads the paired differences and 95 % problem-bootstrap intervals that intervals.py saved. Black on
white with distinct marker shapes, so it reads in grayscale.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = {
    "prompt:capability": "Prompt-only: capability",
    "prompt:sensitivity": "Prompt-only: sensitivity",
    "prompt:combined": "Prompt-only: combined",
    "calibration (nested)": "Calibration (nested)",
    "trace sensitivity": "Counterfactual trace",
    "hidden-activation probe": "Hidden-activation probe",
}
MARKERS = ["o", "s", "D", "^", "v", "P"]


def rows(path):
    """[(label, difference, low, high)] in table order."""
    vs = json.loads(Path(path).read_text())["vs_majority"]
    return [(LABELS[k], v["difference"], v["interval95"][0], v["interval95"][1]) for k, v in vs.items() if k in LABELS]


def plot(data, out):
    fig, ax = plt.subplots(figsize=(6.0, 2.0), dpi=200)
    n = len(data)
    for i, (label, d, lo, hi) in enumerate(data):
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color="black", lw=1.2, solid_capstyle="butt")
        ax.plot([lo, lo], [y - 0.15, y + 0.15], color="black", lw=1.2)
        ax.plot([hi, hi], [y - 0.15, y + 0.15], color="black", lw=1.2)
        ax.plot([d], [y], MARKERS[i % len(MARKERS)], color="black", mfc="white", mew=1.2, ms=5.5, zorder=3)
    ax.axvline(0, color="black", lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(data)], fontsize=7.5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(min(r[2] for r in data) - 1, max(r[3] for r in data) + 1)
    ax.set_xlabel("Correct rows minus majority (of 34), 95 % interval", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else HERE / "intervals.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else HERE / "forest.png"
    plot(rows(src), dst)
