"""Sensitivity of the Rayleigh analog to damping and base frequency.

OpenFOAM mixing fields are held fixed. Only analog constants change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cswe.agent import LevelSetAgent
from cswe.baseline import LatinHypercubeBaseline
from cswe.environment import ExplorationEnv
from cswe.geometry import CLASSICAL_INJECTORS
from cswe.metrics import campaign_diagnostics, score_against_test
from cswe.mixing import ATLAS_PATH, TEST_PATH
from cswe.physics import (
    ACOUSTIC_DAMPING,
    CHAMBER_OMEGA,
    analog_constants,
    relabel_mixing_row,
    unstable_runs_1d,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "artifacts" / "sensitivity.json"

GRID = [
    {"name": "D=0.8 D0", "damping_scale": 0.8, "omega_scale": 1.0},
    {"name": "nominal", "damping_scale": 1.0, "omega_scale": 1.0},
    {"name": "D=1.2 D0", "damping_scale": 1.2, "omega_scale": 1.0},
    {"name": "ω=0.9 ω0", "damping_scale": 1.0, "omega_scale": 0.9},
    {"name": "ω=1.1 ω0", "damping_scale": 1.0, "omega_scale": 1.1},
]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _topology(g_rows: list[dict], damping: float, omega0: float) -> dict:
    labeled = [relabel_mixing_row(r, damping=damping, omega0=omega0) for r in g_rows]
    runs = unstable_runs_1d([r["g"] for r in labeled], [r["sigma"] for r in labeled])
    return {
        "n_unstable_stations": int(sum(r["sigma"] > 0 for r in labeled)),
        "n_unstable_intervals": len(runs),
        "intervals": [{"g_lo": a, "g_hi": b} for a, b in runs],
        "second_interval_present": len(runs) >= 2,
        "nonmonotonic": len(runs) >= 2 or (
            len(runs) == 1 and runs[0][0] > min(r["g"] for r in labeled) + 0.05
            and runs[0][1] < max(r["g"] for r in labeled) - 0.05
        ),
    }


def _classical(damping: float, omega0: float) -> dict:
    atlas = _load_json(ATLAS_PATH)
    out = {}
    if not atlas:
        return out
    by_label = {r.get("label"): r for r in atlas["rows"] if r.get("label")}
    for name, x in CLASSICAL_INJECTORS.items():
        row = by_label.get(name)
        if row is None:
            continue
        lab = relabel_mixing_row({**row, "o": x["o"]}, damping=damping, omega0=omega0)
        out[name] = {"sigma_analog": lab["sigma"], "stable": bool(lab["stable"]), "tau": lab["tau"]}
    return out


def _atlas_fraction(damping: float, omega0: float) -> dict:
    atlas = _load_json(ATLAS_PATH)
    rows = [relabel_mixing_row(r, damping=damping, omega0=omega0) for r in atlas["rows"] if r.get("Cconv")]
    n_u = sum(r["sigma"] > 0 for r in rows)
    return {"n": len(rows), "n_unstable": n_u, "unstable_fraction": n_u / len(rows)}


def _seed_study(damping: float, omega0: float, n_seeds: int, budget: int, n_init: int) -> dict:
    test_raw = _load_json(TEST_PATH)
    test_rows = []
    if test_raw:
        test_rows = [
            relabel_mixing_row(r, damping=damping, omega0=omega0) for r in test_raw["rows"]
        ]
    recs = []
    with analog_constants(damping=damping, omega0=omega0):
        for i in range(n_seeds):
            seed = 11 + 3 * i
            ai = LevelSetAgent(ExplorationEnv(seed=seed), n_init=n_init).run(budget=budget)
            base = LatinHypercubeBaseline(ExplorationEnv(seed=seed + 10_000)).run(budget=budget)
            rec = {
                "seed": seed,
                "ai": campaign_diagnostics(ai.evaluations),
                "baseline": campaign_diagnostics(base.evaluations),
            }
            if test_rows:
                rec["ai_test"] = score_against_test(ai.evaluations, test_rows)
                rec["baseline_test"] = score_against_test(base.evaluations, test_rows)
            recs.append(rec)

    def mean_field(which: str, field: str) -> float | None:
        vals = [r[which][field] for r in recs if r[which].get(field) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "ai_mean_n_unstable": mean_field("ai", "n_unstable_found"),
        "lhs_mean_n_unstable": mean_field("baseline", "n_unstable_found"),
        "ai_mean_unstable_recall": mean_field("ai_test", "unstable_recall") if test_rows else None,
        "lhs_mean_unstable_recall": mean_field("baseline_test", "unstable_recall") if test_rows else None,
        "ai_better_n_unstable": float(np.mean([r["ai"]["n_unstable_found"] > r["baseline"]["n_unstable_found"] for r in recs])),
    }
    if test_rows:
        summary["ai_better_recall"] = float(
            np.mean(
                [
                    (r["ai_test"].get("unstable_recall") or 0)
                    >= (r["baseline_test"].get("unstable_recall") or 0)
                    for r in recs
                ]
            )
        )
    return {"summary": summary, "n_seeds": n_seeds, "budget": budget}


def run_sensitivity(n_seeds: int = 16, budget: int = 16, n_init: int = 5) -> dict:
    gdata = _load_json(ROOT / "artifacts" / "g_sweep.json")
    g_rows = gdata["rows"] if gdata else []
    cases = []
    for spec in GRID:
        damping = ACOUSTIC_DAMPING * spec["damping_scale"]
        omega0 = CHAMBER_OMEGA * spec["omega_scale"]
        case = {
            "name": spec["name"],
            "damping": damping,
            "omega0": omega0,
            "damping_scale": spec["damping_scale"],
            "omega_scale": spec["omega_scale"],
            "atlas": _atlas_fraction(damping, omega0),
            "classical": _classical(damping, omega0),
        }
        if g_rows:
            case["g_slice"] = _topology(g_rows, damping, omega0)
        case["seed_study"] = _seed_study(damping, omega0, n_seeds=n_seeds, budget=budget, n_init=n_init)
        cases.append(case)

    nominal = next(c for c in cases if c["name"] == "nominal")
    topology_persistent = all(
        c.get("g_slice", {}).get("second_interval_present") for c in cases if "g_slice" in c
    )
    ai_still_ahead = all(
        (c["seed_study"]["summary"]["ai_mean_n_unstable"] or 0)
        > (c["seed_study"]["summary"]["lhs_mean_n_unstable"] or 0)
        for c in cases
    )
    payload = {
        "nominal_damping": ACOUSTIC_DAMPING,
        "nominal_omega0": CHAMBER_OMEGA,
        "note": (
            "Mixing fields are frozen OpenFOAM outputs. Only analog damping D and "
            "base frequency ω0 change. The exact σ_analog=0 contour moves; the "
            "question is whether qualitative topology and the adaptive-search "
            "advantage persist."
        ),
        "second_unstable_interval_persists_on_g_slice": topology_persistent,
        "ai_finds_more_unstables_in_every_setting": ai_still_ahead,
        "cases": cases,
        "nominal_g_intervals": nominal.get("g_slice"),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
