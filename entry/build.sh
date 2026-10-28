#!/usr/bin/env bash
# Assemble dist/entry/{main,small}-hybrid-v2.zip from the tracked solution.py files plus the
# kit's uncertainty_profile/ package and uncertainty_artifacts/*.joblib. Prints each sha256.
# `build.sh v3` instead builds dist/entry/{main,small}-training-v3.zip (solution.py and
# training-prior.json only, fixed timestamps and file order) and leaves every other zip alone;
# OUT_DIR overrides the output folder.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
lane="$(dirname "$here")"
if [ "${1:-}" = "v3" ]; then
  out="${OUT_DIR:-$lane/dist/entry}"
  mkdir -p "$out"
  for name in main-training-v3 small-training-v3; do
    d="$(mktemp -d)"
    cp "$here/$name/solution.py" "$here/training-prior.json" "$d/"
    files="solution.py training-prior.json"
    # the kit requires a root-level small.txt in a small-track zip and forbids it in a main-track one
    if [ -f "$here/$name/small.txt" ]; then cp "$here/$name/small.txt" "$d/"; files="$files small.txt"; fi
    (cd "$d" && touch -d "2026-01-01 00:00:00 UTC" $files)
    rm -f "$out/$name.zip"
    (cd "$d" && printf '%s\n' $files | TZ=UTC zip -qX -@ "$out/$name.zip")
    rm -rf "$d"
    sha256sum "$out/$name.zip"
  done
  exit 0
fi
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
