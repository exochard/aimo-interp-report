"""Which train-main-v2 file a script reads, and proof of which one it read.

Two files are known: "pinned" (the revision the report was built on, 82 rows) and "1015" (the file of
2026-10-15, the same labelled rows plus nine more Skywork rows). A choice is one of those names or a
path; the file's sha256 must match a known entry. AIMO_TRAIN sets the default when no flag is given.
"""

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KNOWN = {
    "pinned": ("data/train-main-v2/train-00000-of-00001.parquet",
               "bedfb551ea0f4d95d5e70e1df7cf61395ce2d36eeba2869aba6a90fc3c198824"),
    "1015": ("data/train-main-v2/train-2026-10-15.parquet",
             "e18eb44207e7987781ec2ff214793196038442bdc1466fdf5498f3e87ec58285"),
}
ENV = "AIMO_TRAIN"


def default():
    return os.environ.get(ENV) or "pinned"


def resolve(choice=None, root=None):
    """Return (path, name, sha256) for a choice; the hash must belong to a known entry."""
    root = Path(root) if root else ROOT
    choice = choice or default()
    if choice in KNOWN:
        path = root / KNOWN[choice][0]
    else:
        path = Path(choice)
        if not path.is_absolute():
            path = root / path
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    names = [n for n, (_, h) in KNOWN.items() if h == sha]
    if not names:
        raise ValueError(f"train file {path} has sha256 {sha}, which matches no known entry")
    if choice in KNOWN and names[0] != choice:
        raise ValueError(f"train file {path} has sha256 {sha}, expected {KNOWN[choice][1]} for {choice!r}")
    return path, names[0], sha


def read(choice=None, root=None):
    """The chosen file as a DataFrame, with its name and sha256."""
    import pandas as pd

    path, name, sha = resolve(choice, root)
    return pd.read_parquet(path), name, sha


def provenance(name, sha):
    return {"train_file": name, "train_sha256": sha}


def add_argument(ap):
    ap.add_argument("--train", default=None,
                    help=f"pinned, 1015 or a path to a known train file (default: ${ENV}, else pinned)")
    return ap


def strip_flag(argv):
    """Pull `--train X` or `--train=X` out of a positional argv; return (choice or None, rest)."""
    rest, choice, i = [], None, 0
    while i < len(argv):
        if argv[i] == "--train":
            choice, i = argv[i + 1], i + 2
        elif argv[i].startswith("--train="):
            choice, i = argv[i].split("=", 1)[1], i + 1
        else:
            rest.append(argv[i])
            i += 1
    return choice, rest
