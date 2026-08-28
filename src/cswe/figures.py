"""Static figures a judge can put in the writeup without running Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "artifacts" / "figures"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def g_sweep_figure() -> Path | None:
    path = ROOT / "artifacts" / "g_sweep.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    g = [r["g"] for r in rows]
    sigma = [r["sigma"] for r in rows]
    tau = [r["tau"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(g, sigma, "o-", color="#c0392b", label="σ")
    ax.axhline(0.0, color="#444", ls="--", lw=1)
    ax.set_xlabel("pattern class g")
    ax.set_ylabel("growth rate σ")
    ax2 = ax.twinx()
    ax2.plot(g, tau, "s--", color="#2471a3", label="τ")
    ax2.set_ylabel("mixing delay τ (s)")
    ax.set_title(f"Live OpenFOAM pattern-class sweep (s = {data.get('s', 0.1):.2f})")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    return _save(fig, "g_sweep.png")


def swirl_sweep_figure() -> Path | None:
    path = ROOT / "artifacts" / "swirl_sweep.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    s = [r["s"] for r in rows]
    sigma = [r["sigma"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(s, sigma, "o-", color="#c0392b")
    ax.axhline(0.0, color="#444", ls="--", lw=1)
    ax.set_xlabel("swirl analog s")
    ax.set_ylabel("growth rate σ")
    ax.set_title(f"Swirl sweep inside like-on-like family (g = {data.get('g', 0.15):.2f})")
    return _save(fig, "swirl_sweep.png")


def live_cfd_learning_curves() -> Path | None:
    seeds = [11, 14, 19]
    payloads = []
    for seed in seeds:
        p = ROOT / "artifacts" / f"cfd_study_s{seed}" / "cfd_comparison.json"
        if p.exists():
            payloads.append(json.loads(p.read_text(encoding="utf-8")))
    if not payloads:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = {11: "#1f618d", 14: "#117a65", 19: "#6c3483"}
    for p in payloads:
        seed = p["seed"]
        ai = p["ai"]["curve"]
        base = p["baseline"]["curve"]
        ax.plot(
            [r["budget"] for r in ai],
            [r["unstable_recall"] for r in ai],
            "o-",
            color=colors[seed],
            label=f"AI seed {seed}",
        )
        ax.plot(
            [r["budget"] for r in base],
            [r["unstable_recall"] for r in base],
            "s--",
            color=colors[seed],
            alpha=0.55,
            label=f"LHS seed {seed}",
        )
    ax.set_xlabel("OpenFOAM evaluations")
    ax.set_ylabel("hold-out unstable recall")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Live foamRun campaigns vs independent OpenFOAM test set")
    ax.legend(ncol=2, fontsize=9)
    return _save(fig, "live_cfd_recall_curves.png")


def seed_study_figure() -> Path | None:
    path = ROOT / "artifacts" / "seed_study.json"
    if not path.exists():
        return None
    recs = json.loads(path.read_text(encoding="utf-8"))["records"]
    ai = np.array([r["ai"]["n_unstable_found"] for r in recs], dtype=float)
    lhs = np.array([r["baseline"]["n_unstable_found"] for r in recs], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(lhs, ai, c="#1f618d", s=36, zorder=3)
    lo, hi = 0, max(12, float(ai.max()), float(lhs.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="#888")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("LHS unstable evaluations (budget 16)")
    ax.set_ylabel("AI unstable evaluations (budget 16)")
    ax.set_title("24 atlas seeds: points above the diagonal favor the agent")
    ax.set_aspect("equal", adjustable="box")
    return _save(fig, "seed_study_unstable_counts.png")


def write_all() -> list[Path]:
    _style()
    written = []
    for fn in (g_sweep_figure, swirl_sweep_figure, live_cfd_learning_curves, seed_study_figure):
        p = fn()
        if p is not None:
            written.append(p)
    return written
