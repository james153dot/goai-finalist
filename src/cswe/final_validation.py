"""One-shot, post-development independent OpenFOAM validation.

The algorithm, constants, discovery criteria, and evaluation metrics were
frozen before the final validation set was generated.

This module must never substitute atlas predictions for a claimed independent
final CFD validation. If OpenFOAM is unavailable the result is PENDING.
"""

from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import qmc
import numpy as np

from cswe.agent import LevelSetAgent
from cswe.baseline import LatinHypercubeBaseline, RandomBaseline
from cswe.environment import ExplorationEnv
from cswe.metrics import campaign_diagnostics, score_against_test
from cswe.openfoam import openfoam_available, run_mixer
from cswe.physics import PARAM_BOUNDS, PARAM_NAMES, analog_constant_record, _acoustics
from cswe.runmeta import run_metadata

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "artifacts" / "final_validation.json"
HOLDOUT_PATH = ROOT / "artifacts" / "final_validation_holdout.json"

FROZEN_STATEMENT = (
    "The algorithm, constants, discovery criteria, and evaluation metrics "
    "were frozen before the final validation set was generated."
)

# Independent of the live n=8 seeds and of of_test.json seed 123.
DEFAULT_SEED = 101
DEFAULT_HOLDOUT_SIZE = 48
DEFAULT_BUDGET = 16
DEFAULT_N_INIT = 5
DEFAULT_N_ITER = 90

ACQUISITION_CONFIG = {
    "class": "LevelSetAgent",
    "policy": "full",
    "n_init": DEFAULT_N_INIT,
    "n_candidates": 700,
    "epsilon": 0.10,
    "straddle": "1.96 * std - |mu - 0|",
    "missing_class_hunt": True,
}


def frozen_record(*, seed: int = DEFAULT_SEED, holdout_size: int = DEFAULT_HOLDOUT_SIZE) -> dict:
    return {
        "frozen_statement": FROZEN_STATEMENT,
        "analog_constants": analog_constant_record(),
        "acquisition": ACQUISITION_CONFIG,
        "parameter_bounds": {k: list(v) for k, v in PARAM_BOUNDS.items()},
        "final_validation_seed": int(seed),
        "holdout_size": int(holdout_size),
        "budget": DEFAULT_BUDGET,
        "n_iter": DEFAULT_N_ITER,
        "methods": ["LevelSetAgent(policy=full)", "LatinHypercubeBaseline", "RandomBaseline"],
        "scoring": (
            "Identical GP scoring as metrics.score_against_test on the new hold-out. "
            "No tuning after the hold-out is generated."
        ),
        "holdout_path": str(HOLDOUT_PATH.relative_to(ROOT)),
        "existing_holdout_not_reused": "artifacts/of_test.json is not the final-validation set.",
    }


def _build_independent_holdout(n: int, seed: int, n_iter: int) -> list[dict]:
    """New OpenFOAM hold-out. Writes a new file; never overwrites of_test.json."""
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit = sampler.random(n=n)
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES])
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES])
    pts = [{k: float(row[i]) for i, k in enumerate(PARAM_NAMES)} for row in qmc.scale(unit, lo, hi)]
    rows = []
    for i, x in enumerate(pts):
        report = run_mixer(x, n_iter=n_iter)
        row = {**x, "tau": report.tau, "Um": report.Um, "R_spatial": report.R_spatial,
               "compactness": report.compactness, "x_q": report.x_q,
               "Cvalid": report.Cvalid, "Cconv": report.Cvalid, "notes": report.notes}
        if report.Cvalid:
            n_index, omega, rayleigh, sigma, S, phase = _acoustics(
                report.tau, report.Um, x["o"], report.R_spatial, report.compactness
            )
            row.update({"n_index": n_index, "sigma": sigma, "sigma_analog": sigma,
                        "S": S, "phase": phase, "stable": int(S < 1.0), "R": rayleigh})
        rows.append(row)
        print(f"[final-holdout {i+1}/{n}] Cvalid={row['Cvalid']}", flush=True)
    HOLDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_PATH.write_text(
        json.dumps({"rows": rows, **run_metadata(openfoam_executed=True, extra={"seed": seed, "n": n})}, indent=2),
        encoding="utf-8",
    )
    return rows


def run_final_validation(
    *,
    seed: int = DEFAULT_SEED,
    holdout_size: int = DEFAULT_HOLDOUT_SIZE,
    budget: int = DEFAULT_BUDGET,
    n_init: int = DEFAULT_N_INIT,
    n_iter: int = DEFAULT_N_ITER,
    out: Path = OUT_PATH,
) -> dict:
    frozen = frozen_record(seed=seed, holdout_size=holdout_size)
    frozen["budget"] = int(budget)
    frozen["n_iter"] = int(n_iter)
    meta = run_metadata(openfoam_executed=False)
    if not openfoam_available():
        payload = {
            "status": "PENDING",
            "reason": (
                "OpenFOAM 14 is not available. Independent final CFD hold-out was not generated. "
                "Atlas predictions are not substituted for this validation."
            ),
            "frozen": frozen,
            "results": None,
            **meta,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    holdout = _build_independent_holdout(holdout_size, seed, n_iter)
    env_ai = ExplorationEnv(seed=seed, backend="openfoam", n_iter=n_iter)
    ai = LevelSetAgent(env_ai, n_init=n_init, policy="full").run(budget=budget)
    env_lhs = ExplorationEnv(seed=seed + 10_000, backend="openfoam", n_iter=n_iter)
    lhs = LatinHypercubeBaseline(env_lhs).run(budget=budget)
    env_rand = ExplorationEnv(seed=seed + 20_000, backend="openfoam", n_iter=n_iter)
    rand = RandomBaseline(env_rand).run(budget=budget)
    results = {
        "ai": {
            "diagnostics": campaign_diagnostics(ai.evaluations),
            "holdout": score_against_test(ai.evaluations, holdout),
        },
        "lhs": {
            "diagnostics": campaign_diagnostics(lhs.evaluations),
            "holdout": score_against_test(lhs.evaluations, holdout),
        },
        "random": {
            "diagnostics": campaign_diagnostics(rand.evaluations),
            "holdout": score_against_test(rand.evaluations, holdout),
        },
        "holdout_n": len(holdout),
        "holdout_n_valid": sum(1 for r in holdout if r.get("Cvalid")),
        "holdout_n_unstable": sum(1 for r in holdout if r.get("Cvalid") and r.get("sigma", 0) > 0),
        "note": "Frozen methods were evaluated once. Do not retune from this result.",
    }
    payload = {
        "status": "COMPLETE",
        "frozen": frozen,
        "results": results,
        **run_metadata(openfoam_executed=True),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
