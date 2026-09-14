"""Numerical verification of analog classifications vs mesh / iteration settings.

This is V&V of the *declared analog* under the existing OpenFOAM mixer.
It does not validate the analog against a real combustor.

Representative conditions are the three committed classical injector analogs:
like-on-like (clearly analog-unstable), unlike-impinging (near σ_analog = 0),
and swirl-coaxial (clearly analog-stable). No new physical configurations
are invented for this study.
"""

from __future__ import annotations

import json
from pathlib import Path

from cswe.geometry import CLASSICAL_INJECTORS, jet_layout
from cswe.openfoam import mesh_plan, openfoam_available, run_mixer
from cswe.physics import analog_constant_record, _acoustics
from cswe.runmeta import run_metadata

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "artifacts" / "verification.json"
FIG_PATH = ROOT / "artifacts" / "figures" / "numerical_verification.png"

# Existing committed analogs. Roles follow the committed classical table.
REPRESENTATIVE_CONDITIONS: dict[str, dict] = {
    "like_on_like": {
        "role": "clearly_analog_unstable",
        "x": dict(CLASSICAL_INJECTORS["like_on_like"]),
    },
    "unlike_impinging": {
        "role": "near_sigma_analog_zero",
        "x": dict(CLASSICAL_INJECTORS["unlike_impinging"]),
    },
    "swirl_coaxial": {
        "role": "clearly_analog_stable",
        "x": dict(CLASSICAL_INJECTORS["swirl_coaxial"]),
    },
}

DEFAULT_MESH_SCALES = (0.70, 1.00, 1.40)
# 90 is the live-campaign iteration cap. Neighbors test under- and over-iteration.
DEFAULT_ITER_CAPS = (40, 90, 160)
DEFAULT_SEED = 0


def planned_configuration(
    mesh_scales: tuple[float, ...] = DEFAULT_MESH_SCALES,
    iter_caps: tuple[int, ...] = DEFAULT_ITER_CAPS,
) -> dict:
    conditions = []
    for name, spec in REPRESENTATIVE_CONDITIONS.items():
        layout = jet_layout(**spec["x"])
        conditions.append(
            {
                "name": name,
                "role": spec["role"],
                "x": spec["x"],
                "mesh_plans": {str(s): mesh_plan(layout, s) for s in mesh_scales},
            }
        )
    return {
        "conditions": conditions,
        "mesh_scales": list(mesh_scales),
        "iteration_caps": list(iter_caps),
        "default_mesh_scale": 1.0,
        "default_nxs": 48,
        "note": (
            "mesh_scale=1.0 reproduces the committed default (nxs=48, _ny unchanged). "
            "No new injector geometries are introduced."
        ),
    }


def _row_from_report(name: str, spec: dict, report, mesh_scale: float, n_iter: int) -> dict:
    layout = jet_layout(**spec["x"])
    plan = mesh_plan(layout, mesh_scale)
    if report.Cvalid:
        *_, sigma, S, _ = _acoustics(
            report.tau, report.Um, spec["x"]["o"], report.R_spatial, report.compactness
        )
        classification = "unstable" if sigma > 0 else "stable"
    else:
        sigma = float("nan")
        S = float("nan")
        classification = "invalid"
    return {
        "condition": name,
        "role": spec["role"],
        "x": spec["x"],
        "mesh_scale": float(mesh_scale),
        "n_iter": int(n_iter),
        "nxs": plan["nxs"],
        "n_cells": plan["n_cells"],
        "tau": report.tau,
        "R_spatial": report.R_spatial,
        "compactness": report.compactness,
        "Um": report.Um,
        "sigma_analog": sigma,
        "S": S,
        "classification": classification,
        "Cvalid": bool(report.Cvalid),
        "Cconv": bool(report.Cvalid),
        "notes": report.notes,
    }


def run_verification(
    *,
    mesh_scales: tuple[float, ...] = DEFAULT_MESH_SCALES,
    iter_caps: tuple[int, ...] = DEFAULT_ITER_CAPS,
    seed: int = DEFAULT_SEED,
    out: Path = OUT_PATH,
) -> dict:
    config = planned_configuration(mesh_scales, iter_caps)
    analog = analog_constant_record()
    if not openfoam_available():
        payload = {
            "status": "PENDING",
            "reason": "OpenFOAM 14 is not available; no numerical verification cases were executed.",
            "configuration": config,
            "analog_constants": analog,
            "random_seed": int(seed),
            "results": [],
            "classification_robustness": None,
            **run_metadata(openfoam_executed=False),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    results = []
    for name, spec in REPRESENTATIVE_CONDITIONS.items():
        for mesh_scale in mesh_scales:
            for n_iter in iter_caps:
                report = run_mixer(spec["x"], n_iter=n_iter, mesh_scale=mesh_scale)
                results.append(_row_from_report(name, spec, report, mesh_scale, n_iter))

    robustness = _assess_robustness(results)
    payload = {
        "status": "COMPLETE",
        "configuration": config,
        "analog_constants": analog,
        "random_seed": int(seed),
        "results": results,
        "classification_robustness": robustness,
        **run_metadata(openfoam_executed=True),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _assess_robustness(results: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = {}
    for row in results:
        by_cond.setdefault(row["condition"], []).append(row)
    out = {}
    for name, rows in by_cond.items():
        valid = [r for r in rows if r.get("Cvalid")]
        classes = {r["classification"] for r in valid}
        sigmas = [r["sigma_analog"] for r in valid if r["sigma_analog"] == r["sigma_analog"]]
        out[name] = {
            "n_valid": len(valid),
            "n_total": len(rows),
            "unique_classifications": sorted(classes),
            "classification_constant": len(classes) == 1,
            "sigma_analog_range": [float(min(sigmas)), float(max(sigmas))] if sigmas else None,
        }
    all_constant = bool(out) and all(v["classification_constant"] for v in out.values())
    return {
        "all_conditions_classification_constant": all_constant,
        "per_condition": out,
        "note": (
            "Classification is called robust here only if every valid mesh/iteration "
            "setting keeps the same analog-stable / analog-unstable label. "
            "This is numerical robustness of the analog, not validation against a combustor."
        ),
    }


def write_figure(payload: dict, path: Path = FIG_PATH) -> Path | None:
    if payload.get("status") != "COMPLETE" or not payload.get("results"):
        return None
    if not payload.get("openfoam_executed"):
        return None
    import matplotlib.pyplot as plt

    rows = payload["results"]
    names = list(REPRESENTATIVE_CONDITIONS)
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.0))
    quantities = (
        (axes[0, 0], "sigma_analog", r"$\sigma_{\mathrm{analog}}$"),
        (axes[0, 1], "tau", r"$\tau$ (s)"),
        (axes[0, 2], "R_spatial", r"$R_{\mathrm{spatial}}$"),
        (axes[1, 0], "compactness", "compactness $C$"),
        (axes[1, 1], "Um", r"$U_m$"),
        (axes[1, 2], "Cvalid", r"$C_{\mathrm{valid}}$"),
    )
    colors = {"like_on_like": "#c0392b", "unlike_impinging": "#d4ac0d", "swirl_coaxial": "#1f618d"}
    for ax, field, title in quantities:
        for name in names:
            sub = [r for r in rows if r["condition"] == name]
            xs = [r["mesh_scale"] + 0.02 * (r["n_iter"] - 90) / 120.0 for r in sub]
            ys = [1.0 if r[field] else 0.0 for r in sub] if field == "Cvalid" else [r[field] for r in sub]
            ax.plot(xs, ys, "o-", color=colors[name], label=name.replace("_", " "), alpha=0.85)
        if field == "sigma_analog":
            ax.axhline(0.0, color="#444", ls="--", lw=1)
        ax.set_xlabel("mesh scale (jittered by iteration cap)")
        ax.set_ylabel(title)
        ax.set_title(title)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Numerical verification of analog quantities vs mesh / iteration (OpenFOAM executed)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
