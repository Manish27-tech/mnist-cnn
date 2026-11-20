"""Artefact checks."""

from __future__ import annotations

from mnistvision import config


def check_artifacts() -> dict:
    problems = []
    if not config.CHECKPOINT.exists():
        problems.append("missing checkpoint")
    if not (config.DASHBOARD_FEED / "kpis.json").exists():
        problems.append("missing dashboard feed")
    figures = sorted(p.name for p in config.FIGURES.glob("*.png"))
    if len(figures) < 8:
        problems.append(f"only {len(figures)} figures")
    return {"status": "ok" if not problems else "fail", "problems": problems, "figures": figures}
