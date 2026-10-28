#!/usr/bin/env bash
# Compile report.tex (NeurIPS 2026 template, anonymous) to report.pdf with tectonic and print the page count.
# Fails if report.tex still has unfilled markers or the PDF text carries an identity.
set -euo pipefail
cd "$(dirname "$0")"
tectonic=${TECTONIC:-$HOME/AI/var/tools/tectonic-0.17.0/tectonic}
grep -qF '⟦' report.tex && { echo "report.tex still has unfilled ⟦ markers"; exit 1; }
systemd-run --user --scope -q -p CollectMode=inactive-or-failed -p MemoryMax=3G -p MemorySwapMax=0 -p CPUQuota=200% nice -n 19 \
  "$tectonic" -X compile report.tex
# identity.tsv is untracked (it names the authors); without it only home paths are checked.
pat='/home/'
[ -s identity.tsv ] && pat+="|$(cut -f1 identity.tsv | paste -sd '|')"
if pdftotext report.pdf - | grep -iE "$pat"; then echo "identity found in report.pdf"; exit 1; fi
pdfinfo report.pdf | grep -E '^(Pages|Title|Author|Creator|Producer):'
