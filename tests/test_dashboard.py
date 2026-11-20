"""Dashboard smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"
HAS_FEED = (ROOT / "artifacts" / "dashboard" / "kpis.json").exists()

pytestmark = pytest.mark.skipif(not HAS_FEED, reason="run the pipeline first")


def test_default_page():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception


def test_performance_page():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.sidebar.radio[0].set_value("Performance").run()
    assert not at.exception


def test_about_page():
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    at.sidebar.radio[0].set_value("About").run()
    assert not at.exception
