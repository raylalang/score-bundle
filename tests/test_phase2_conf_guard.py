"""The confirmation-mode guard must refuse without the explicit consent var.

The frozen 13-piece pool is one-shot (registered 2026-08-17): flipping
PHASE2_SPLIT=confirmation without PHASE2_CONFIRMATION_I_AM_SURE=yes must fail
at import time, before any stage could touch a confirmation file.
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")


def _import_probe(extra_env: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = SCRIPTS + os.pathsep + os.path.join(REPO, "src") \
        + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("PHASE2_SPLIT", None)
    env.pop("PHASE2_CONFIRMATION_I_AM_SURE", None)
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", "import eval_phase2_real; print('imported')"],
        capture_output=True, text=True, env=env, cwd=REPO)


def test_confirmation_mode_refused_without_consent():
    out = _import_probe({"PHASE2_SPLIT": "confirmation"})
    assert out.returncode != 0
    assert "REFUSED" in out.stderr


def test_dev_mode_imports_fine():
    out = _import_probe({})
    assert out.returncode == 0
    assert "imported" in out.stdout


def test_consented_confirmation_mode_imports():
    out = _import_probe({"PHASE2_SPLIT": "confirmation",
                         "PHASE2_CONFIRMATION_I_AM_SURE": "yes"})
    assert out.returncode == 0


def test_driver_script_refuses_without_env():
    out = subprocess.run(
        ["bash", os.path.join(SCRIPTS, "run_phase2_confirmation.sh")],
        capture_output=True, text=True,
        env={k: v for k, v in os.environ.items()
             if k not in ("PHASE2_SPLIT", "PHASE2_CONFIRMATION_I_AM_SURE")},
        cwd=REPO)
    assert out.returncode != 0
    assert "REFUSED" in out.stdout
