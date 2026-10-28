#!/usr/bin/env bash
# Build the public code archive for the technical report from the committed tree.
#   [DIST=<lane dist/ in the main checkout>] report/package.sh [OUT_DIR]
#                                   (default OUT_DIR ~/AI/var/staging/aimo-report/<date>)
# Tracked code, the report, the saved run files with problem text replaced by hashes, the
# submission bundles, and MANIFEST.sha256. Refuses to write into an existing folder.
# Run it in the main checkout: runs/ is gitignored, so a worktree has none of the run files.
set -euo pipefail
# --- identity scrub and gate (anonymous review) ---
# The patterns name the authors, so they live in an untracked file and never ship:
# one `pattern<TAB>replacement` per line; replacement `-` means check only.
ids=${IDENTITY_FILE:-$(cd "$(dirname "$0")" && pwd)/identity.tsv}
[ -s "$ids" ] || { echo "missing $ids"; exit 1; }
scrub_identity() {  # $1 = staged root; rewrites text files only (binary files and zips are left byte-exact)
  local p r f
  while IFS=$'\t' read -r p r; do
    [ "$r" = - ] && continue
    { rg -l -I --no-ignore --hidden -F -e "$p" "$1" || true; } |
    while IFS= read -r f; do sed -i "s|$p|$r|g" "$f"; done
  done < "$ids"
}
identity_gate() {  # $1 = staged root; non-zero if an identity string survives in text, a PDF or a zip member
  local pat bad=0 f
  pat=$(cut -f1 "$ids" | paste -sd '|')
  rg -i -I --no-ignore --hidden -n --max-columns 160 -e "$pat" "$1" && bad=1
  while IFS= read -r -d '' f; do
    case $f in
      *.pdf) pdftotext "$f" - | rg -i -n -e "$pat" && { echo "identity in PDF: $f"; bad=1; } ;;
      *.zip) unzip -p "$f" | rg -a -i -q -e "$pat" && { echo "identity inside zip: $f"; bad=1; } ;;
    esac
  done < <(find "$1" -type f \( -name '*.pdf' -o -name '*.zip' \) -print0)
  [ "$bad" = 0 ] || { echo "identity gate failed"; return 1; }
}
py=${PY:?set PY to a python with pyarrow, e.g. PY=$HOME/AI/var/venvs/k12-vl/bin/python}
lane=$(cd "$(dirname "$0")/.." && pwd)
repo=$(git -C "$lane" rev-parse --show-toplevel)
rel=${lane#"$repo"/}
out=${1:-$HOME/AI/var/staging/aimo-report/$(date +%F)}
name=supplementary
[ -e "$out/$name" ] && { echo "exists: $out/$name"; exit 1; }
git -C "$repo" diff --quiet HEAD -- "$rel" || { echo "uncommitted changes in $rel"; exit 1; }
[ -f "$lane/report/report.tex" ] || { echo "missing $lane/report/report.tex"; exit 1; }
grep -qF '⟦' "$lane/report/report.tex" && { echo "report/report.tex still has unfilled ⟦ markers"; exit 1; }
[ "$lane/report/report.pdf" -nt "$lane/report/report.tex" ] || { echo "report/report.pdf is missing or older than report/report.tex: run report/build_pdf.sh"; exit 1; }
runs=(runs/feat/features.jsonl runs/feat/meta.json runs/probe-gate.json runs/probe.json
      runs/calibration-nested/report.json runs/trace-pilot/rows.jsonl runs/trace-pilot/probe.json
      runs/agreement-vs-labels.csv runs/agreement-vs-labels.json runs/activation-r3-yuris/probe.json
      runs/force-r4/Qwen3.5-4B.jsonl runs/force-r4/Skywork-OR1-Math-7B.jsonl
      runs/force-r4/DeepSeek-R1-0528-Qwen3-8B.jsonl runs/force-r4/eval-3models.json
      runs/artifact-gate/features.jsonl runs/artifact-gate/manifest.json runs/artifact-gate/score.json)
for f in "${runs[@]}"; do
  [ -e "$lane/$f" ] || { echo "missing $lane/$f: package.sh must run in the main checkout because runs/ is gitignored"; exit 1; }
done
mkdir -p "$out/$name"
git -C "$repo" archive HEAD "$rel" | tar -x -C "$out/$name" --strip-components=$(($(tr -cd / <<<"$rel" | wc -c) + 1))
for f in "${runs[@]}"; do
  mkdir -p "$out/$name/$(dirname "$f")" && cp "$lane/$f" "$out/$name/$f"
done
# Submission zips: both constant baselines, the single-configuration probes and the hybrids.
# DIST points at the lane's dist/ in the main checkout (zips are untracked, so a worktree has none).
dist=${DIST:-$lane/dist}
shopt -s nullglob
zips=("$dist"/always-true-main.zip "$dist"/always-true-small.zip "$dist"/entry/*.zip "$dist"/probe/*.zip)
[ ${#zips[@]} -ge 4 ] || { echo "no submission zips under $dist"; exit 1; }
for z in "${zips[@]}"; do
  rel=${z#"$dist"/}
  mkdir -p "$out/$name/dist/$(dirname "$rel")" && cp "$z" "$out/$name/dist/$rel"
done
parquet=$lane/data/train-main-v2/train-00000-of-00001.parquet
"$py" "$lane/report/rehydrate.py" strip "$out/$name" "$parquet"
# No problem statement may remain anywhere in the archive, in raw or JSON-escaped form.
"$py" - "$out/$name" "$parquet" <<'EOF'
import json, sys, pathlib, pyarrow.parquet as pq
root, probs = pathlib.Path(sys.argv[1]), pq.read_table(sys.argv[2], columns=["problem"]).column("problem").to_pylist()
needles = {p[:40] for p in probs} | {json.dumps(p)[1:41] for p in probs}
hits = [(str(f.relative_to(root)), n[:20]) for f in root.rglob("*") if f.is_file() and f.suffix != ".zip"
        for n in needles if n in f.read_text(errors="ignore")]
sys.exit(f"problem text found: {hits[:3]}") if hits else print("no problem text in archive")
EOF
scrub_identity "$out/$name"
identity_gate "$out/$name"
(cd "$out/$name" && find . -type f ! -name MANIFEST.sha256 -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum > MANIFEST.sha256)
(cd "$out" && find "$name" -exec touch -d 2026-11-01T00:00:00Z {} + && LC_ALL=C find "$name" -type f | LC_ALL=C sort | zip -q -X -@ "$name.zip")
sha256sum "$out/$name.zip"
echo "files: $(wc -l < "$out/$name/MANIFEST.sha256"), commit $(git -C "$repo" rev-parse --short HEAD)"
