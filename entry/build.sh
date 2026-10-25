#!/usr/bin/env bash
# Assemble dist/entry/{main,small}-hybrid-v2.zip from the tracked solution.py files plus the
# kit's uncertainty_profile/ package and uncertainty_artifacts/*.joblib. Prints each sha256.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
lane="$(dirname "$here")"
kit="${KIT_DIR:-$HOME/Projects/Athlon/contests/aimo-interp/kit}/solutions/uncertainty-profiling"
out="$lane/dist/entry"
mkdir -p "$out"
stage="$(mktemp -d "$out/tmp.XXXXXXXXXX")"
trap 'rm -rf "$stage"' EXIT
for name in main-hybrid-v2 small-hybrid-v2; do
  d="$stage/$name"
  mkdir -p "$d/uncertainty_artifacts"
  cp -r "$kit/uncertainty_profile" "$d/"
  find "$d" -name __pycache__ -type d -prune -exec rm -rf {} +
  cp "$kit"/uncertainty_artifacts/*.joblib "$d/uncertainty_artifacts/"
  cp "$here/$name/solution.py" "$d/"
  [ -f "$here/$name/small.txt" ] && cp "$here/$name/small.txt" "$d/"
  rm -f "$out/$name.zip"
  (cd "$d" && zip -qrX "$out/$name.zip" .)
  sha256sum "$out/$name.zip"
done
