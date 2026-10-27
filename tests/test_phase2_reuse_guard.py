"""The re-use one-shots (registered 2026-10-27) must refuse without consent
and must never reach the spent 2026-08-27 artifacts.

Every probe runs in a subprocess so the import-time gates are exercised; no
probe needs the confirmation cache (every gate precedes the cache load).
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
VARS = ("PHASE2_SPLIT", "PHASE2_CONFIRMATION_I_AM_SURE",
        "PHASE2_REUSE_I_AM_SURE")
CONF = {"PHASE2_SPLIT": "confirmation",
        "PHASE2_CONFIRMATION_I_AM_SURE": "yes"}
REUSE = dict(CONF, PHASE2_REUSE_I_AM_SURE="yes")


def _probe(code: str, extra_env: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in VARS}
    env["PYTHONPATH"] = SCRIPTS + os.pathsep + os.path.join(REPO, "src") \
        + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra_env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, cwd=REPO)


def test_tonal_refused_in_confirmation_mode_without_reuse_consent():
    out = _probe("import eval_phase2_real as e; e.stage_run('0/1', tonal=True)",
                 CONF)
    assert out.returncode != 0 and "REFUSED" in out.stderr


def test_corrnoise_refused_in_confirmation_mode_without_reuse_consent():
    out = _probe("import eval_corrnoise_tau", CONF)
    assert out.returncode != 0 and "REFUSED" in out.stderr


def test_reuse_mode_routes_tonal_to_its_own_artifacts():
    out = _probe("import eval_phase2_real as e; "
                 "print(e._cells_dir(True)); print(e.TONAL_OUT_MD); "
                 "print(e.TONAL_CELLS); print(e._cells_dir(False))", REUSE)
    assert out.returncode == 0, out.stderr
    got = out.stdout.split()
    assert got[0].endswith(os.path.join("phase2_cells", "conf_tonal"))
    assert got[1] == "results/phase2_tonal_confirmation_results.md"
    assert got[2] == "results/phase2_tonal_confirmation_cells.pkl"
    assert got[3].endswith(os.path.join("phase2_cells", "conf"))


def test_reuse_mode_locks_the_spent_plain_confirmation():
    for code in ("import eval_phase2_real as e; e.stage_run('0/1', tonal=False)",
                 "import eval_phase2_real as e; e.stage_report(tonal=False)",
                 "import eval_phase2_real as e; e.stage_run('0/1', tonal='chain')"):
        out = _probe(code, REUSE)
        assert out.returncode != 0 and "REFUSED" in out.stderr, code


def test_reuse_mode_dispatch_locks_plain_verbs():
    for verb in ("extract", "run", "report", "delta"):
        out = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "eval_phase2_real.py"), verb],
            capture_output=True, text=True, cwd=REPO,
            env=dict({k: v for k, v in os.environ.items() if k not in VARS},
                     PYTHONPATH=SCRIPTS + os.pathsep + os.path.join(REPO, "src"),
                     **REUSE))
        assert out.returncode != 0 and "REFUSED" in out.stderr, verb


def test_corrnoise_refuses_robustness_switches_in_confirmation_mode():
    out = _probe("import sys; sys.argv += ['run', '--joint']; "
                 "import eval_corrnoise_tau", REUSE)
    assert out.returncode != 0 and "REFUSED" in out.stderr


def test_dev_mode_unchanged():
    out = _probe("import eval_phase2_real as e, eval_corrnoise_tau as c; "
                 "print(e._cells_dir(True), e.TONAL_OUT_MD, c.OUT_DIR)", {})
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == [os.path.join("results", "phase2_cells", "tonal"),
                                  "results/phase2_tonal_dev.md",
                                  "results/corrnoise_tau"]


def test_reuse_runner_refuses_without_env():
    out = subprocess.run(
        ["bash", os.path.join(SCRIPTS, "run_phase2_reuse_confirmation.sh"),
         "tonal"], capture_output=True, text=True, cwd=REPO,
        env={k: v for k, v in os.environ.items() if k not in VARS})
    assert out.returncode != 0 and "REFUSED" in out.stdout
