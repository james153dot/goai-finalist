"""JSONL / JSON campaign logs and comparison metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import warnings

from cswe.agent import CampaignRecord
from cswe.physics import STABILITY_THRESHOLD, true_stability

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def save_campaign(record: CampaignRecord, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "exploration.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for row in record.evaluations:
            fh.write(json.dumps(row) + "\n")
    meta = {
        "method": record.method,
        "seed": record.seed,
        "budget": record.budget,
        "n_evaluations": len(record.evaluations),
        "n_converged": sum(1 for r in record.evaluations if r["Cconv"] == 1),
        "hypotheses": record.hypotheses,
        "discoveries": record.discoveries,
        "stability_threshold": STABILITY_THRESHOLD,
        "fixed_environment": {
            "chamber_mode": "first_longitudinal_n_tau",
            "injector_mixing": "openfoam14_2d_laminar_dual_jet",
            "public_outputs": "abstract_design_principles_only",
        },
    }
    (out_dir / "campaign.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_evaluations(log_path: Path) -> list[dict]:
    rows = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def boundary_error(rows: list[dict], n_probe: int = 800, seed: int = 0) -> dict:
    """Hold-out reconstruction error of the stability classifier vs the true map."""
    valid = [r for r in rows if r["Cconv"] == 1]
    if len(valid) < 5:
        return {"n_valid": len(valid), "holdout_accuracy": None, "brier": None}

    X = np.array([[r["g"], r["d"], r["a"], r["s"], r["o"]] for r in valid], dtype=float)
    y = np.array([r["S"] for r in valid], dtype=float)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=0.05),
        normalize_y=True,
        random_state=seed,
    )
    gp.fit(X, y)

    rng = np.random.default_rng(seed)
    probes = rng.random((n_probe, 5))
    probes[:, 0] *= 2.0
    mu = gp.predict(probes)
    pred_stable = mu < STABILITY_THRESHOLD
    true_stable = np.array(
        [
            true_stability({"g": p[0], "d": p[1], "a": p[2], "s": p[3], "o": p[4]})
            for p in probes
        ]
    )
    acc = float(np.mean(pred_stable == true_stable))
    # Soft score: map S prediction to a probability via a logistic around the threshold.
    prob_unstable = 1.0 / (1.0 + np.exp(-(mu - STABILITY_THRESHOLD) / 0.25))
    true_unstable = (~true_stable).astype(float)
    brier = float(np.mean((prob_unstable - true_unstable) ** 2))
    return {
        "n_valid": len(valid),
        "n_probe": n_probe,
        "holdout_accuracy": acc,
        "brier": brier,
        "pred_stable_fraction": float(np.mean(pred_stable)),
        "true_stable_fraction": float(np.mean(true_stable)),
    }


def first_boundary_step(rows: list[dict], band: float = 0.35) -> int | None:
    for r in rows:
        if r["Cconv"] == 1 and abs(r["S"] - STABILITY_THRESHOLD) < band:
            return int(r["t"])
    return None


def compare_campaigns(ai: CampaignRecord, baseline: CampaignRecord) -> dict:
    ai_rows = ai.evaluations
    base_rows = baseline.evaluations
    return {
        "budget": ai.budget,
        "seed": ai.seed,
        "ai": {
            "method": ai.method,
            "n_discoveries": len(ai.discoveries),
            "first_boundary_step": first_boundary_step(ai_rows),
            "reconstruction": boundary_error(ai_rows, seed=ai.seed),
            "hypotheses": ai.hypotheses,
        },
        "baseline": {
            "method": baseline.method,
            "n_discoveries": len(baseline.discoveries),
            "first_boundary_step": first_boundary_step(base_rows),
            "reconstruction": boundary_error(base_rows, seed=baseline.seed),
            "hypotheses": baseline.hypotheses,
        },
    }
