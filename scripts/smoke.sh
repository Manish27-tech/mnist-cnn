#!/usr/bin/env bash
set -euo pipefail
python -m ruff check src tests dashboard run.py
python -m pytest -q
python run.py check
