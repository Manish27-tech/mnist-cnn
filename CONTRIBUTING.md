# Contributing

Thanks for taking a look. Small, focused pull requests are welcome.

1. Create a virtual environment and install `requirements-dev.txt`.
2. Run `ruff check src tests dashboard run.py` and `pytest -q`.
3. Keep new modules covered by at least one test.
4. Update `docs/METHODOLOGY.md` if you change training or evaluation.

Please do not commit the raw dataset or large binary files other than the small
artefacts required by the dashboard.
