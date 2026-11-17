"""The phrase-scale one-shot (docs/phrase_prior_prereg_design.md) must refuse
without consent, route only to its own artifacts under consent, and leave the
development path unchanged.  Probes run in subprocesses (import-time gates)."""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
VARS = ("PHASE2_SPLIT", "PHASE2_CONFIRMATION_I_AM_SURE", "PHASE2_REUSE_I_AM_SURE")
CONF = {"PHASE2_SPLIT": "confirmation", "PHASE2_CONFIRMATION_I_AM_SURE": "yes"}
REUSE = dict(CONF, PHASE2_REUSE_I_AM_SURE="yes")
CODE = ("import eval_prior_value_phrase_dev as m; "
        "print(m.CELLS_DIR, m.PREFIX, m.OUT_MD, m.CACHE)")


def _probe(code, extra_env):
    env = {k: v for k, v in os.environ.items() if k not in VARS}
    env["PYTHONPATH"] = SCRIPTS + os.pathsep + os.path.join(REPO, "src")
    env.update(extra_env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, cwd=REPO)


def test_refused_in_confirmation_mode_without_both_consents():
    for env in ({"PHASE2_SPLIT": "confirmation"}, CONF,
                {"PHASE2_SPLIT": "confirmation", "PHASE2_REUSE_I_AM_SURE": "yes"}):
        out = _probe(CODE, env)
        assert out.returncode != 0 and "REFUSED" in out.stderr, env


def test_consent_routes_to_own_artifacts_only():
    out = _probe(CODE, REUSE)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["results/phase3_cells/conf_phrase", "priorval_phrase_conf",
                                  "results/phrase_prior_confirmation_results.md",
                                  ".cache/urmp_targets_conf.pkl"]


def test_dev_mode_unchanged():
    out = _probe(CODE, {})
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["results/phase3_cells", "priorval_phrase",
                                  "results/prior_value_phrase_dev.md",
                                  ".cache/urmp_targets_dev.pkl"]


def test_runner_refuses_without_env():
    out = subprocess.run(["bash", os.path.join(SCRIPTS, "run_phrase_prior_confirmation.sh")],
                         capture_output=True, text=True, cwd=REPO,
                         env={k: v for k, v in os.environ.items() if k not in VARS})
    assert out.returncode != 0 and "REFUSED" in out.stdout
