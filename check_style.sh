#!/usr/bin/env bash
# Run the full local check: format, lint, type-check, and tests. Fails on the
# first tool that reports a problem. CI runs everything here except pylint.
# `python -m` keeps every tool on the active environment's interpreter.
set -euo pipefail

python -m ruff format .
python -m ruff check .
python -m mypy src/csiphon
python -m pylint src/csiphon
python -m pytest
