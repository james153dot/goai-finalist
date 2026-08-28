"""Scientist-facing metrics.

Two families:

1. Campaign diagnostics that do not need a second model: how fast the
   minority unstable class is found, how many unstable evaluations the
   budget bought, how many points sit near σ = 0.
2. Hold-out OpenFOAM scoring: a GP fit on the campaign's σ values is
   compared to independent CFD test rows (unstable recall, boundary MAE).

Volume accuracy against a smooth interpolator is not the claim. The claim
is that adaptive search spends expensive solver calls on the window edge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import warnings

from cswe.geometry import CLASSICAL_INJECTORS
from cswe.physics import PARAM_NAMES, STABILITY_THRESHOLD

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SIGMA_THRESHOLD = 0.0
EDGE_BAND = 0.12


def _sigma(row: dict) -> float:
    if row.get("sigma") is not None and row.get("sigma") == row.get("sigma"):
        return float(row["sigma"])
    return float(row["S"]) - 1.0


def is_unstable(row: dict) -> bool:
    return _sigma(row) > SIGMA_THRESHOLD


def campaign_diagnostics(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("Cconv") in (1, True) and _sigma(r) == _sigma(r)]
    n_u = int(sum(is_unstable(r) for r in valid))
    first = None
    for r in valid:
        if is_unstable(r):
            first = int(r.get("t", valid.index(r)))
            break
    near = [r for r in valid if abs(_sigma(r)) < EDGE_BAND]
    return {
        "n_valid": len(valid),
        "n_unstable_found": n_u,
        "unstable_fraction": (n_u / len(valid)) if valid else 0.0,
        "time_to_first_unstable": first,
        "n_near_boundary": len(near),
        "frac_evals_near_edge": (len(near) / len(valid)) if valid else 0.0,
        "mean_abs_sigma": float(np.mean([abs(_sigma(r)) for r in valid])) if valid else None,
    }


def _gp(rows: list[dict]) -> GaussianProcessRegressor | None:
    valid = [r for r in rows if r.get("Cconv") in (1, True) and _sigma(r) == _sigma(r)]
    if len(valid) < 4:
        return None
    X = np.array([[r[n] for n in PARAM_NAMES] for r in valid], dtype=float)
    y = np.array([_sigma(r) for r in valid], dtype=float)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(0.12, 8.0)) + WhiteKernel(noise_level=0.03),
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0,
    )
    gp.fit(X, y)
    return gp


def score_against_test(campaign_rows: list[dict], test_rows: list[dict]) -> dict:
    """Compare a campaign's GP (fit on σ_analog) to independent OpenFOAM tests.

    Unstable recall
        Recall_U = TP_U / (TP_U + FN_U)
        on the hold-out set, where unstable means σ_analog > 0.
        Overall accuracy can stay high by predicting the majority stable class;
        Recall_U asks whether the reconstructed map recovers the minority regime.

    Boundary MAE
        Let B = { i in hold-out : |σ_i| < 0.20 }. Then
            E_B = (1/|B|) Σ_{i in B} |σ̂(x_i) − σ_i|
        where σ̂ is the GP fit on campaign evaluations. This is the error in
        predicted growth rate on hold-out points that already lie near the
        analog threshold, not a Euclidean contour distance in parameter space.
    """
    test = [r for r in test_rows if r.get("Cconv") in (1, True) and _sigma(r) == _sigma(r)]
    gp = _gp(campaign_rows)
    out = campaign_diagnostics(campaign_rows)
    out.update(
        {
            "n_campaign": out["n_valid"],
            "n_test": len(test),
            "volume_accuracy": None,
            "unstable_recall": None,
            "stable_recall": None,
            "boundary_mae": None,
            "brier": None,
        }
    )
    if gp is None or not test:
        return out
    Xt = np.array([[r[n] for n in PARAM_NAMES] for r in test], dtype=float)
    yt = np.array([_sigma(r) for r in test], dtype=float)
    ybin = yt > SIGMA_THRESHOLD
    mu = gp.predict(Xt)
    pred_bin = mu > SIGMA_THRESHOLD
    out["volume_accuracy"] = float(np.mean(pred_bin == ybin))
    if ybin.any():
        out["unstable_recall"] = float(np.mean(pred_bin[ybin]))
    if (~ybin).any():
        out["stable_recall"] = float(np.mean(~pred_bin[~ybin]))
    edge = np.abs(yt) < 0.20
    if edge.any():
        out["boundary_mae"] = float(np.mean(np.abs(mu[edge] - yt[edge])))
    else:
        out["boundary_mae"] = float(np.mean(np.abs(mu - yt)))
    # Logistic around σ = 0.
    prob_u = 1.0 / (1.0 + np.exp(-mu / 0.08))
    out["brier"] = float(np.mean((prob_u - ybin.astype(float)) ** 2))
    return out


def learning_curve(campaign_rows: list[dict], test_rows: list[dict], steps: list[int] | None = None) -> list[dict]:
    n = len(campaign_rows)
    if steps is None:
        steps = [k for k in range(5, n + 1, 2)]
        if n not in steps:
            steps.append(n)
    return [{"budget": k, **score_against_test(campaign_rows[:k], test_rows)} for k in steps if k <= n]


def classical_truth() -> dict[str, dict]:
    from cswe.physics import simulate

    out = {}
    rng = np.random.default_rng(0)
    for name, x in CLASSICAL_INJECTORS.items():
        r = simulate(x, rng=rng, backend="atlas")
        out[name] = {
            "S": r.S,
            "sigma": r.sigma,
            "stable": r.stable,
            "tau": r.tau,
            "R_spatial": r.R_spatial,
            "compactness": r.compactness,
            "x_q": r.x_q,
            "phase": r.phase,
        }
    return out


def load_test_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]
