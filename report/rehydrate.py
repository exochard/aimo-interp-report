"""Swap the organisers' problem text in the saved run files for a hash, and back.

The training set (aimo-interp/train-main-v2) carries licence "other", so the public archive does not
redistribute its problem statements. The saved features key rows by problem text; `strip` replaces
every JSON string equal to a problem statement with "sha256:<first 16 hex>", and `restore` puts the
text back from the parquet the reader downloads at the pinned revision (see REPRODUCE.md).

  python report/rehydrate.py strip   <root> <parquet>
  python report/rehydrate.py restore <root> <parquet>
"""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

FILES = ["runs/feat/features.jsonl", "runs/trace-pilot/rows.jsonl", "runs/calibration-nested/report.json"]


def token(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def walk(x, table):
    if isinstance(x, dict):
        return {table.get(k, k): walk(v, table) for k, v in x.items()}
    if isinstance(x, list):
        return [walk(v, table) for v in x]
    return table.get(x, x) if isinstance(x, str) else x


def strings(x):
    if isinstance(x, dict):
        return [s for k, v in x.items() for s in [k, *strings(v)]]
    if isinstance(x, list):
        return [s for v in x for s in strings(v)]
    return [x] if isinstance(x, str) else []


def main(mode: str, root: str, parquet: str) -> None:
    problems = set(pq.read_table(parquet, columns=["problem"]).column("problem").to_pylist())
    forward = {p: token(p) for p in problems}
    table = forward if mode == "strip" else {v: k for k, v in forward.items()}
    for name in FILES:
        path = Path(root) / name
        text = path.read_text()
        if name.endswith(".jsonl"):
            data = [walk(json.loads(l), table) for l in text.splitlines() if l.strip()]
            out = "".join(json.dumps(d) + "\n" for d in data)
        else:
            data = walk(json.loads(text), table)
            out = json.dumps(data, indent=1) + "\n"
        path.write_text(out)
        if mode == "strip":
            seen = strings(data)
            leaked = [p[:40] for p in problems if any(p[:40] in s for s in seen)]
            if leaked:
                sys.exit(f"{name}: problem text left after strip: {leaked[:2]}")
        print(f"{mode} {name}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
