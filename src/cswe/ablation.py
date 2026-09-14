"""Policy-component ablation on the committed inexpensive atlas.

Compares the current LevelSetAgent against stripped acquisition variants and
the non-adaptive baselines under an identical budget, bounds, hold-out,
seed set, and GP scoring procedure.

This is a policy-component ablation, not a causal identification of a single
mechanism. Do not overclaim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cswe.agent import LevelSetAgent
from cswe.baseline import LatinHypercubeBaseline, RandomBaseline
from cswe.environment import ExplorationEnv
from cswe.metrics import campaign_diagnostics, load_test_set, score_against_test
from cswe.mixing import TEST_PATH
from cswe.physics import PARAM_BOUNDS, analog_constant_record
from cswe.runmeta import run_metadata

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "artifacts" / "ablation_study.json"
FIG_PATH = ROOT / "artifacts" / "figures" / "ablation_study.png"

DEFAULT_SEEDS = list(range(16))
DEFAULT_BUDGET = 16
DEFAULT_N_INIT = 5

METHODS = (
    ("full", "LevelSetAgent full"),
    ("straddle", "straddle only"),
    ("uncertainty", "uncertainty only"),
    ("no_regime_hunt", "no missing-regime hunt"),
    ("lhs", "Latin hypercube"),
    ("random", "uniform random"),
)

METRIC_KEYS = (
    "unstable_recall",
    "unstable_precision",
    "unstable_f1",
    "n_unstable_found",
    "time_to_first_unstable",
    "near_boundary_sigma_mae",
    "volume_accuracy",
)


def _run_method(method: str, seed: int, budget: int, n_init: int):
    if method == "lhs":
        return LatinHypercubeBaseline(ExplorationEnv(seed=seed + 10_000)).run(budget)
    if method == "random":
        return RandomBaseline(ExplorationEnv(seed=seed + 20_000)).run(budget)
    env = ExplorationEnv(seed=seed)
    return LevelSetAgent(env, n_init=n_init, policy=method).run(budget)


def _censor_ttf(value, budget: int) -> float:
    return float(budget) if value is None else float(value)


def run_ablation(
    *,
    seeds: list[int] | None = None,
    budget: int = DEFAULT_BUDGET,
    n_init: int = DEFAULT_N_INIT,
    out: Path = OUT_PATH,
) -> dict:
    seeds = list(seeds or DEFAULT_SEEDS)
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing committed hold-out {TEST_PATH}")
    test_rows = load_test_set(TEST_PATH)
    records = []
    for seed in seeds:
        rec = {"seed": int(seed), "methods": {}}
        for method, _label in METHODS:
            campaign = _run_method(method, seed, budget, n_init)
            scored = score_against_test(campaign.evaluations, test_rows)
            diag = campaign_diagnostics(campaign.evaluations)
            rec["methods"][method] = {
                "diagnostics": diag,
                "test": scored,
                "unstable_recall": scored.get("unstable_recall"),
                "unstable_precision": scored.get("unstable_precision"),
                "unstable_f1": scored.get("unstable_f1"),
                "n_unstable_found": diag.get("n_unstable_found"),
                "time_to_first_unstable": diag.get("time_to_first_unstable"),
                "near_boundary_sigma_mae": scored.get("near_boundary_sigma_mae") or scored.get("boundary_mae"),
                "volume_accuracy": scored.get("volume_accuracy"),
            }
        records.append(rec)
        print(f"ablation seed {seed} done", flush=True)

    summary = {"n_seeds": len(seeds), "budget": budget, "methods": {}}
    for method, label in METHODS:
        block = {}
        for key in METRIC_KEYS:
            raw = [r["methods"][method][key] for r in records]
            if key == "time_to_first_unstable":
                vals = np.array([_censor_ttf(v, budget) for v in raw], dtype=float)
            else:
                vals = np.array([v for v in raw if v is not None], dtype=float)
            block[key] = {
                "mean": float(vals.mean()) if len(vals) else None,
                "median": float(np.median(vals)) if len(vals) else None,
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
        summary["methods"][method] = {"label": label, **block}

    payload = {
        "status": "COMPLETE",
        "backend": "atlas",
        "openfoam_executed": False,
        "holdout": "artifacts/of_test.json",
        "seeds": seeds,
        "budget": budget,
        "n_init": n_init,
        "parameter_bounds": {k: list(v) for k, v in PARAM_BOUNDS.items()},
        "analog_constants": analog_constant_record(),
        "scoring": "metrics.score_against_test on the committed OpenFOAM hold-out",
        "methods": [{"id": m, "label": lab} for m, lab in METHODS],
        "summary": summary,
        "records": records,
        "interpretation": (
            "Policy-component ablation on the inexpensive atlas. "
            "Differences indicate which acquisition ingredients are associated "
            "with the observed advantage under this protocol; they do not "
            "identify a unique causal mechanism."
        ),
        **run_metadata(openfoam_executed=False, extra={"random_seed": seeds[0] if seeds else None}),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_figure(payload: dict, path: Path = FIG_PATH) -> Path | None:
    if not payload.get("summary"):
        return None
    import matplotlib.pyplot as plt

    methods = [m["id"] for m in payload["methods"]]
    labels = [m["label"] for m in payload["methods"]]
    panels = (
        ("unstable_recall", "Unstable recall"),
        ("unstable_precision", "Unstable precision"),
        ("unstable_f1", "Unstable F1"),
        ("n_unstable_found", "Unstable evaluations found"),
        ("time_to_first_unstable", "Time to first unstable"),
        ("near_boundary_sigma_mae", r"Near-boundary $\sigma$ MAE"),
        ("volume_accuracy", "Volume accuracy"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.6))
    axes = axes.ravel()
    colors = ["#1f618d", "#5dade2", "#48c9b0", "#f4d03f", "#7f8c8d", "#b2babb"]
    x = np.arange(len(methods))
    for ax, (key, title) in zip(axes, panels):
        means = [payload["summary"]["methods"][m][key]["mean"] or 0.0 for m in methods]
        sds = [payload["summary"]["methods"][m][key]["sd"] or 0.0 for m in methods]
        ax.bar(x, means, yerr=sds, color=colors, ecolor="#444", capsize=3)
        ax.set_xticks(x, [lab.replace(" ", "\n") for lab in labels], fontsize=7)
        ax.set_title(title, fontsize=10)
    axes[-1].axis("off")
    n = payload["summary"]["n_seeds"]
    fig.suptitle(
        f"Atlas policy-component ablation (n={n} seeds, budget {payload['budget']}). "
        "Error bars: sample sd. Not a causal claim."
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
