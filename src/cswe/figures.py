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


def _cfd_payloads() -> list[dict]:
    payloads = []
    for p in sorted((ROOT / "artifacts").glob("cfd_study_s*/cfd_comparison.json")):
        payloads.append(json.loads(p.read_text(encoding="utf-8")))
    return payloads


def live_cfd_learning_curves() -> Path | None:
    payloads = _cfd_payloads()
    if not payloads:
        return None
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i, p in enumerate(payloads):
        seed = p["seed"]
        color = cmap(i % 10)
        ai = p["ai"]["curve"]
        base = p["baseline"]["curve"]
        ax.plot(
            [r["budget"] for r in ai],
            [r["unstable_recall"] for r in ai],
            "o-",
            color=color,
            label=f"AI {seed}",
        )
        ax.plot(
            [r["budget"] for r in base],
            [r["unstable_recall"] for r in base],
            "s--",
            color=color,
            alpha=0.5,
            label=f"LHS {seed}",
        )
    ax.set_xlabel("OpenFOAM evaluations")
    ax.set_ylabel("hold-out unstable recall")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Live foamRun campaigns vs independent OpenFOAM test set")
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, "live_cfd_recall_curves.png")


def live_cfd_strip() -> Path | None:
    payloads = _cfd_payloads()
    if not payloads:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    for ax, field, title in (
        (axes[0], "unstable_recall", "Hold-out unstable recall"),
        (axes[1], "n_unstable_found", "Unstable evaluations found"),
    ):
        ai = [p["ai"]["final"][field] for p in payloads]
        lhs = [p["baseline"]["final"][field] for p in payloads]
        seeds = [p["seed"] for p in payloads]
        ax.scatter(np.zeros(len(ai)) + 0.0, ai, c="#1f618d", s=50, zorder=3, label="AI")
        ax.scatter(np.zeros(len(lhs)) + 1.0, lhs, c="#7f8c8d", s=50, zorder=3, label="LHS")
        for i, seed in enumerate(seeds):
            ax.plot([0, 1], [ai[i], lhs[i]], color="#bbb", lw=0.8, zorder=1)
            ax.annotate(str(seed), (0.0, ai[i]), textcoords="offset points", xytext=(-14, 0), fontsize=8, color="#1f618d")
        ax.set_xticks([0, 1], ["AI", "LHS"])
        ax.set_title(title)
        ax.set_xlim(-0.4, 1.4)
    fig.suptitle("Each marker is one live OpenFOAM campaign (budget 16)")
    fig.tight_layout()
    return _save(fig, "live_cfd_strip.png")


def seed14_both_bands() -> Path | None:
    path = ROOT / "artifacts" / "cfd_study_s14" / "ai" / "exploration.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for r in rows:
        g, s, t, sig = r["g"], r["s"], r["t"], r["sigma"]
        if sig > 0 and g < 0.7:
            style = dict(c="#c0392b", marker="o", s=70, label="unstable, low g")
        elif sig > 0 and g >= 1.3:
            style = dict(c="#8e44ad", marker="D", s=70, label="unstable, high g")
        else:
            style = dict(c="#27ae60", marker="s", s=40, label="stable")
        ax.scatter(t, g, **style, zorder=3)
    handles = {}
    for h in ax.collections:
        lab = h.get_label()
        if lab not in handles:
            handles[lab] = h
    ax.legend(handles.values(), handles.keys(), loc="best")
    ax.set_xlabel("live OpenFOAM evaluation t")
    ax.set_ylabel("pattern class g")
    ax.set_title("Seed 14 AI campaign: samples in both unstable intervals of g")
    ax.axhline(0.3, color="#c0392b", ls=":", lw=1, alpha=0.5)
    ax.axhline(1.5, color="#8e44ad", ls=":", lw=1, alpha=0.5)
    return _save(fig, "seed14_both_g_intervals.png")


def seed_study_figure() -> Path | None:
    path = ROOT / "artifacts" / "seed_study.json"
    if not path.exists():
        return None
    recs = json.loads(path.read_text(encoding="utf-8"))["records"]
    ai = np.array([r["ai"]["n_unstable_found"] for r in recs], dtype=float)
    lhs = np.array([r["baseline"]["n_unstable_found"] for r in recs], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    ax = axes[0]
    ax.scatter(lhs, ai, c="#1f618d", s=36, zorder=3)
    lo, hi = 0, max(12, float(ai.max()), float(lhs.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="#888")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("LHS unstable evaluations (budget 16)")
    ax.set_ylabel("AI unstable evaluations (budget 16)")
    ax.set_title("24 atlas seeds")
    ax.set_aspect("equal", adjustable="box")
    ax = axes[1]
    ax.boxplot([lhs, ai], tick_labels=["LHS", "AI"], widths=0.45)
    rng = np.random.default_rng(0)
    ax.scatter(1 + rng.uniform(-0.08, 0.08, len(lhs)), lhs, c="#7f8c8d", s=18, alpha=0.7, zorder=3)
    ax.scatter(2 + rng.uniform(-0.08, 0.08, len(ai)), ai, c="#1f618d", s=18, alpha=0.7, zorder=3)
    ax.set_ylabel("unstable evaluations found")
    ax.set_title("Distribution over 24 seeds")
    fig.tight_layout()
    return _save(fig, "seed_study_unstable_counts.png")


def sensitivity_figure() -> Path | None:
    path = ROOT / "artifacts" / "sensitivity.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    names = [c["name"] for c in data["cases"]]
    ai = [c["seed_study"]["summary"]["ai_mean_unstable_recall"] for c in data["cases"]]
    lhs = [c["seed_study"]["summary"]["lhs_mean_unstable_recall"] for c in data["cases"]]
    n_int = [c.get("g_slice", {}).get("n_unstable_intervals", 0) for c in data["cases"]]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(x - 0.18, ai, 0.36, label="AI mean recall", color="#1f618d")
    ax.bar(x + 0.18, lhs, 0.36, label="LHS mean recall", color="#95a5a6")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("atlas seed-study unstable recall")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax2 = ax.twinx()
    ax2.plot(x, n_int, "o--", color="#c0392b", label="g-slice unstable intervals")
    ax2.set_ylabel("unstable intervals on g-slice")
    ax2.set_ylim(0, 4)
    ax.set_title("Analog-constant sensitivity (OpenFOAM fields frozen)")
    return _save(fig, "sensitivity.png")


def write_all() -> list[Path]:
    _style()
    written = []
    for fn in (
        g_sweep_figure,
        swirl_sweep_figure,
        live_cfd_learning_curves,
        live_cfd_strip,
        seed14_both_bands,
        seed_study_figure,
        sensitivity_figure,
    ):
        p = fn()
        if p is not None:
            written.append(p)
    return written
