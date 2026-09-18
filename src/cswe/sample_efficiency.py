"""Sample-efficiency of the eight committed live CFD campaigns.

Reads existing cfd_study_s*/cfd_comparison.json curves only.
Does not rerun or rewrite those historical campaign files.
Does not extrapolate beyond budget 16.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cswe.runmeta import run_metadata

ROOT = Path(__file__).resolve().parents[2]
FIG_PATH = ROOT / "artifacts" / "figures" / "sample_efficiency_live.png"
OUT_PATH = ROOT / "artifacts" / "sample_efficiency_live.json"
MAX_BUDGET = 16
RECALL_LEVELS = (0.25, 0.50, 0.67)


def _payloads() -> list[dict]:
    payloads = []
    for path in sorted((ROOT / "artifacts").glob("cfd_study_s*/cfd_comparison.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    payloads.sort(key=lambda d: d["seed"])
    return payloads


def _curve_to_grid(curve: list[dict], budgets: np.ndarray) -> np.ndarray:
    xs = np.array([r["budget"] for r in curve], dtype=float)
    ys = np.array([r.get("unstable_recall") if r.get("unstable_recall") is not None else np.nan for r in curve], dtype=float)
    out = np.full(len(budgets), np.nan)
    for i, b in enumerate(budgets):
        if b < xs.min() or b > xs.max():
            continue
        out[i] = float(np.interp(b, xs, ys))
    return out


def _first_budget_at_level(curve: list[dict], level: float) -> int | None:
    for row in sorted(curve, key=lambda r: r["budget"]):
        rec = row.get("unstable_recall")
        if rec is not None and rec >= level and row["budget"] <= MAX_BUDGET:
            return int(row["budget"])
    return None


def analyze_from_payloads(
    payloads: list[dict],
    *,
    source: str,
    out_path: Path,
    claim: str,
) -> dict:
    if not payloads:
        raise RuntimeError("no campaign payloads")
    budgets = np.arange(5, MAX_BUDGET + 1)
    ai_grid = []
    lhs_grid = []
    per_seed = []
    for p in payloads:
        ai_c = p["ai"]["curve"]
        lhs_c = p["baseline"]["curve"]
        ai_grid.append(_curve_to_grid(ai_c, budgets))
        lhs_grid.append(_curve_to_grid(lhs_c, budgets))
        reached = {}
        for level in RECALL_LEVELS:
            reached[str(level)] = {
                "ai": _first_budget_at_level(ai_c, level),
                "lhs": _first_budget_at_level(lhs_c, level),
            }
        row = {
            "seed": p["seed"],
            "ai_final_recall": p["ai"]["final"]["unstable_recall"],
            "lhs_final_recall": p["baseline"]["final"]["unstable_recall"],
            "evals_to_recall": reached,
        }
        if isinstance(p.get("random"), dict) and "final" in p["random"]:
            row["random_final_recall"] = p["random"]["final"].get("unstable_recall")
        per_seed.append(row)
    ai = np.vstack(ai_grid)
    lhs = np.vstack(lhs_grid)
    has_random = all(
        isinstance(p.get("random"), dict) and "curve" in (p.get("random") or {}) for p in payloads
    )
    rand_grid = []
    if has_random:
        for p in payloads:
            rand_grid.append(_curve_to_grid(p["random"]["curve"], budgets))
        rand = np.vstack(rand_grid)

    def _band(arr: np.ndarray) -> dict:
        ddof = 1 if arr.shape[0] > 1 else 0
        sd = np.nanstd(arr, axis=0, ddof=ddof)
        return {
            "budget": [int(b) for b in budgets],
            "mean": [float(v) for v in np.nanmean(arr, axis=0)],
            "median": [float(v) for v in np.nanmedian(arr, axis=0)],
            "sd": [float(v) if np.isfinite(v) else 0.0 for v in sd],
        }

    ai_band = _band(ai)
    lhs_band = _band(lhs)
    auc = {
        "ai": float(np.trapezoid(ai_band["mean"], budgets)),
        "lhs": float(np.trapezoid(lhs_band["mean"], budgets)),
        "integration": "trapezoid of mean recall vs budget on [5, 16]",
    }
    auc["difference_ai_minus_lhs"] = auc["ai"] - auc["lhs"]
    payload = {
        "status": "COMPLETE",
        "n_seeds": len(payloads),
        "seeds": [p["seed"] for p in payloads],
        "budget_max": MAX_BUDGET,
        "source": source,
        "historical_campaigns_rewritten": False,
        "ai": ai_band,
        "lhs": lhs_band,
        "area_under_mean_recall_curve": auc,
        "evals_to_recall_level": {},
        "per_seed": per_seed,
        "claim": claim,
        **run_metadata(openfoam_executed=False, extra={"random_seed": None}),
    }
    if has_random:
        rand_band = _band(rand)
        payload["random"] = rand_band
        payload["area_under_mean_recall_curve"]["random"] = float(np.trapezoid(rand_band["mean"], budgets))
        payload["area_under_mean_recall_curve"]["difference_ai_minus_random"] = (
            payload["area_under_mean_recall_curve"]["ai"] - payload["area_under_mean_recall_curve"]["random"]
        )
    for level in RECALL_LEVELS:
        ai_hits = [s["evals_to_recall"][str(level)]["ai"] for s in per_seed]
        lhs_hits = [s["evals_to_recall"][str(level)]["lhs"] for s in per_seed]
        ai_ok = [v for v in ai_hits if v is not None]
        lhs_ok = [v for v in lhs_hits if v is not None]
        payload["evals_to_recall_level"][str(level)] = {
            "ai_n_reached": len(ai_ok),
            "lhs_n_reached": len(lhs_ok),
            "n": len(per_seed),
            "ai_mean_evals_when_reached": float(np.mean(ai_ok)) if ai_ok else None,
            "lhs_mean_evals_when_reached": float(np.mean(lhs_ok)) if lhs_ok else None,
            "note": "Reported only for seeds that actually reached the level within budget 16. No extrapolation.",
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def analyze_live_sample_efficiency() -> dict:
    payloads = _payloads()
    if len(payloads) != 8:
        raise RuntimeError(f"expected 8 live CFD campaigns, found {len(payloads)}")
    return analyze_from_payloads(
        payloads,
        source="existing artifacts/cfd_study_s*/cfd_comparison.json curves; not rerun",
        out_path=OUT_PATH,
        claim=(
            "AI recovers the scientifically important minority regime using fewer "
            "expensive solver evaluations than Latin hypercube on these eight live campaigns."
        ),
    )


def write_figure(payload: dict | None = None, path: Path = FIG_PATH) -> Path | None:
    payload = payload or analyze_live_sample_efficiency()
    import matplotlib.pyplot as plt

    b = np.array(payload["ai"]["budget"])
    ai_m = np.array(payload["ai"]["mean"])
    ai_s = np.array(payload["ai"]["sd"])
    lhs_m = np.array(payload["lhs"]["mean"])
    lhs_s = np.array(payload["lhs"]["sd"])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.fill_between(b, ai_m - ai_s, ai_m + ai_s, color="#1f618d", alpha=0.18, label="AI ± 1 sd")
    ax.fill_between(b, lhs_m - lhs_s, lhs_m + lhs_s, color="#7f8c8d", alpha=0.18, label="LHS ± 1 sd")
    ax.plot(b, ai_m, "o-", color="#1f618d", lw=2.2, label="AI mean recall")
    ax.plot(b, np.array(payload["ai"]["median"]), "o--", color="#1f618d", alpha=0.55, label="AI median")
    ax.plot(b, lhs_m, "s-", color="#7f8c8d", lw=2.2, label="LHS mean recall")
    ax.plot(b, np.array(payload["lhs"]["median"]), "s--", color="#7f8c8d", alpha=0.55, label="LHS median")
    if payload.get("random") and payload["random"].get("mean"):
        rnd_m = np.array(payload["random"]["mean"])
        rnd_s = np.array(payload["random"]["sd"])
        ax.fill_between(b, rnd_m - rnd_s, rnd_m + rnd_s, color="#b2babb", alpha=0.18, label="Random ± 1 sd")
        ax.plot(b, rnd_m, "^-", color="#566573", lw=2.0, label="Random mean recall")
    ax.set_xlim(5, 16)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Live OpenFOAM evaluations")
    ax.set_ylabel("Hold-out unstable recall")
    ax.set_title(f"Sample efficiency on live CFD campaigns (n = {payload['n_seeds']})")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(
        0.02,
        0.96,
        f"n = {payload['n_seeds']} live seeds · no extrapolation past budget 16",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
