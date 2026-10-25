#!/usr/bin/env bash
# Render report.md to report.pdf with report.css (A4, 11 pt, 2.5 cm margins) and print the page count.
set -euo pipefail
cd "$(dirname "$0")"
pandoc report.md -o report.pdf --pdf-engine=weasyprint --css report.css --metadata title=" " --standalone
pdfinfo report.pdf | grep '^Pages:'
