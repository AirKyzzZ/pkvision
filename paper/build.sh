#!/usr/bin/env bash
# Build the PkVision paper end-to-end.
# Run from anywhere; resolves to the paper/ directory.
set -e
cd "$(dirname "$0")"

PAPER_DIR="$(pwd)"
mkdir -p build

# BibTeX runs from inside build/, so BIBINPUTS / BSTINPUTS must be absolute.
export TEXINPUTS=".:${PAPER_DIR}:${PAPER_DIR}/template:${TEXINPUTS:-}"
export BSTINPUTS=".:${PAPER_DIR}:${PAPER_DIR}/template:${BSTINPUTS:-}"
export BIBINPUTS=".:${PAPER_DIR}:${BIBINPUTS:-}"

pdflatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null
(cd build && bibtex main)
pdflatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null
pdflatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null

echo
echo "=== build/main.pdf ==="
pdfinfo build/main.pdf 2>/dev/null | grep -E "Pages|File size"
