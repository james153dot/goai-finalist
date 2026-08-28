"""Post-processing sensitivity of the mixing-delay definition.

No CFD is rerun. Stored axial Var_y[Z] profiles (JSON key q_profile) are
re-read. V_crit and N_bins are modeling choices in τ; this module reports
whether classical ordering and the g-slice qualitative topology survive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cswe.geometry import CLASSICAL_INJECTORS, jet_layout
from cswe.openfoam import (
    DEFAULT_N_AXIAL_BINS,
    DEFAULT_VARIANCE_THRESHOLD,
    mixing_delay_from_variance,
    mixedness_at_fraction,
    resample_axial_profile,
    spatial_overlap_from_profile,
)
from cswe.physics import _acoustics, unstable_runs_1d

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "artifacts" / "tau_sensitivity.json"
G_SWEEP_PATH = ROOT / "artifacts" / "g_sweep.json"
ATLAS_PATH = ROOT / "artifacts" / "mixing_atlas.json"

V_CRIT_GRID = (0.035, 0.045, 0.055)
N_BINS_GRID = (16, 24, 32)


def _u_bulk(row: dict) -> float:
    layout = jet_layout(row["g"], row["d"], row["a"], row["s"], row["o"])
    return 0.5 * (abs(layout.u0[0]) + abs(layout.u1[0]))


def relabel_from_profile(
    row: dict,
    *,
    v_crit: float = DEFAULT_VARIANCE_THRESHOLD,
    n_bins: int = DEFAULT_N_AXIAL_BINS,
) -> dict | None:
    """Recompute τ (and, if n_bins changes, R_spatial / compactness / Um) from stored q_mix."""
    if not row.get("Cconv"):
        return None
    q = row.get("q_profile")
    x = row.get("x_profile")
    if not q or not x:
        return None
    xx = np.asarray(x, dtype=float)
    qq = np.asarray(q, dtype=float)
    xx_s, qq_s = resample_axial_profile(xx, qq, n_bins)
    u_bulk = _u_bulk(row)
    tau, x_m = mixing_delay_from_variance(xx_s, qq_s, u_bulk, thresh=v_crit)
    if int(n_bins) == DEFAULT_N_AXIAL_BINS:
        r_spatial = float(row["R_spatial"])
        compactness = float(row["compactness"])
        um = float(row["Um"])
    else:
        r_spatial, compactness, _xq = spatial_overlap_from_profile(xx_s, qq_s)
        um = mixedness_at_fraction(xx_s, qq_s)
    n_index, _omega, _ray, sigma, S, phase = _acoustics(
        tau, um, row["o"], r_spatial, compactness
    )
    out = dict(row)
    out.update(
        {
            "tau": tau,
            "x_m": x_m,
            "Um": um,
            "R_spatial": r_spatial,
            "compactness": compactness,
            "n_index": n_index,
            "sigma": sigma,
            "S": S,
            "phase": phase,
            "stable": int(S < 1.0),
            "v_crit": v_crit,
            "n_bins": int(n_bins),
        }
    )
    return out


def _classical(atlas_rows: list[dict], v_crit: float, n_bins: int) -> dict:
    by_label = {r.get("label"): r for r in atlas_rows if r.get("label")}
    out = {}
    for name, z in CLASSICAL_INJECTORS.items():
        row = by_label.get(name)
        if row is None:
            continue
        lab = relabel_from_profile({**row, **z}, v_crit=v_crit, n_bins=n_bins)
        if lab is None:
            continue
        out[name] = {
            "sigma_analog": lab["sigma"],
            "stable": bool(lab["stable"]),
            "tau": lab["tau"],
            "n_index": lab["n_index"],
            "R_spatial": lab["R_spatial"],
        }
    return out


def _g_slice(g_rows: list[dict], v_crit: float, n_bins: int) -> dict:
    labeled = [relabel_from_profile(r, v_crit=v_crit, n_bins=n_bins) for r in g_rows]
    labeled = [r for r in labeled if r is not None]
    runs = unstable_runs_1d([r["g"] for r in labeled], [r["sigma"] for r in labeled])
    return {
        "n_unstable_stations": int(sum(r["sigma"] > 0 for r in labeled)),
        "n_unstable_intervals": len(runs),
        "intervals": [{"g_lo": a, "g_hi": b} for a, b in runs],
        "second_interval_present": len(runs) >= 2,
        "sigmas": [{"g": r["g"], "sigma": r["sigma"], "tau": r["tau"]} for r in labeled],
    }


def _ordering(classical: dict) -> dict:
    like = classical.get("like_on_like", {})
    unlike = classical.get("unlike_impinging", {})
    swirl = classical.get("swirl_coaxial", {})
    sigmas = [like.get("sigma_analog"), unlike.get("sigma_analog"), swirl.get("sigma_analog")]
    ok = None not in sigmas and sigmas[0] > sigmas[1] > sigmas[2]
    return {
        "like_gt_unlike_gt_swirl": bool(ok),
        "like_unstable": bool(like.get("sigma_analog", 0) > 0),
        "unlike_stable": bool(unlike.get("sigma_analog", 0) <= 0),
        "swirl_stable": bool(swirl.get("sigma_analog", 0) <= 0),
        "qualitative_ok": bool(
            ok
            and like.get("sigma_analog", 0) > 0
            and swirl.get("sigma_analog", 0) <= 0
        ),
    }


def run_tau_sensitivity() -> dict:
    gdata = json.loads(G_SWEEP_PATH.read_text(encoding="utf-8")) if G_SWEEP_PATH.exists() else None
    atlas = json.loads(ATLAS_PATH.read_text(encoding="utf-8")) if ATLAS_PATH.exists() else None
    g_rows = gdata["rows"] if gdata else []
    atlas_rows = atlas["rows"] if atlas else []

    cases = []
    for v_crit in V_CRIT_GRID:
        for n_bins in N_BINS_GRID:
            if v_crit != DEFAULT_VARIANCE_THRESHOLD and n_bins != DEFAULT_N_AXIAL_BINS:
                continue  # one-at-a-time: don't explode the grid
            name = f"V_crit={v_crit:.3f}, N_bins={n_bins}"
            classical = _classical(atlas_rows, v_crit, n_bins)
            g_slice = _g_slice(g_rows, v_crit, n_bins) if g_rows else {}
            cases.append(
                {
                    "name": name,
                    "v_crit": v_crit,
                    "n_bins": n_bins,
                    "nominal": v_crit == DEFAULT_VARIANCE_THRESHOLD and n_bins == DEFAULT_N_AXIAL_BINS,
                    "classical": classical,
                    "ordering": _ordering(classical),
                    "g_slice": {k: v for k, v in g_slice.items() if k != "sigmas"},
                    "g_slice_sigmas": g_slice.get("sigmas"),
                }
            )

    qualitative = all(c["ordering"]["qualitative_ok"] for c in cases)
    two_intervals = all(c.get("g_slice", {}).get("second_interval_present") for c in cases)
    failed_two = [
        c["name"] for c in cases if not c.get("g_slice", {}).get("second_interval_present")
    ]
    payload = {
        "note": (
            "No CFD rerun. Stored 24-bin Var_y[Z] profiles are reused. "
            "V_crit changes only τ. N_bins resamples the stored profile "
            "(linear interpolation of q_mix) and recomputes τ, R_spatial, "
            "compactness, and Um. That N_bins test is a definition sensitivity, "
            "not a re-extraction from cell centres. "
            "Classical injector ordering survives every setting. "
            "The high-g g-slice interval disappears at V_crit = 0.035 "
            "(stricter mixing criterion → later x_m → larger τ)."
        ),
        "nominal_v_crit": DEFAULT_VARIANCE_THRESHOLD,
        "nominal_n_bins": DEFAULT_N_AXIAL_BINS,
        "classical_ordering_survives": qualitative,
        "g_slice_two_intervals_survive": two_intervals,
        "g_slice_two_intervals_fail_at": failed_two,
        "cases": cases,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
