"""Post-development live OpenFOAM expansion.

The original eight campaigns in artifacts/cfd_study_s* stay frozen.
New campaigns are written under artifacts/cfd_expansion_s* so existing
n=8 figures, sample-efficiency, and cfd_live_summary.json are not mixed.

This expansion was not used to tune analog constants or the acquisition.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cswe.agent import CampaignRecord, LevelSetAgent
from cswe.baseline import LatinHypercubeBaseline, RandomBaseline
from cswe.environment import ExplorationEnv
from cswe.logging_utils import load_evaluations, save_campaign
from cswe.metrics import campaign_diagnostics, learning_curve, load_test_set, score_against_test
from cswe.mixing import TEST_PATH
from cswe.openfoam import openfoam_available, run_mixer
from cswe.physics import analog_constant_record, unstable_runs_1d, _acoustics
from cswe.runmeta import run_metadata

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_SEEDS = [8, 11, 14, 19, 23, 26, 32, 35]
# Disjoint from the frozen live set and from final-validation seed 101.
EXPANSION_SEEDS = [38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83]
DEFAULT_BUDGET = 16
DEFAULT_N_INIT = 5
DEFAULT_N_ITER = 90
DEFAULT_WORKERS = 2
SUMMARY_PATH = ROOT / "artifacts" / "cfd_live_expansion.json"
POOLED_PATH = ROOT / "artifacts" / "cfd_live_pooled.json"
GSWEEP_VERIFY_PATH = ROOT / "artifacts" / "g_sweep_mesh_verification.json"
PROGRESS_PATH = ROOT / "artifacts" / "cfd_expansion_progress.log"


def expansion_dir(seed: int) -> Path:
    return ROOT / "artifacts" / f"cfd_expansion_s{seed}"


def _stats(vals: list[float | None]) -> dict:
    a = np.array([v for v in vals if v is not None], dtype=float)
    if not len(a):
        return {"mean": None, "median": None, "sd": None, "range": None, "n": 0}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "range": [float(a.min()), float(a.max())],
        "n": int(len(a)),
    }


def _log(msg: str) -> None:
    line = msg.rstrip() + "\n"
    print(line, end="", flush=True)
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _try_load_campaign(path: Path, expected_budget: int) -> CampaignRecord | None:
    meta_path = path / "campaign.json"
    log_path = path / "exploration.jsonl"
    if not meta_path.exists() or not log_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    evaluations = load_evaluations(log_path)
    if len(evaluations) < expected_budget:
        return None
    return CampaignRecord(
        method=str(meta.get("method") or "unknown"),
        seed=int(meta.get("seed") or 0),
        budget=int(meta.get("budget") or expected_budget),
        evaluations=evaluations[:expected_budget],
        hypotheses=list(meta.get("hypotheses") or []),
        discoveries=list(meta.get("discoveries") or []),
    )


def _score_record(record: CampaignRecord, test_rows: list[dict]) -> dict:
    return {
        "final": score_against_test(record.evaluations, test_rows),
        "curve": learning_curve(record.evaluations, test_rows),
        "hypotheses": record.hypotheses,
    }


def run_one_seed(
    seed: int,
    *,
    budget: int = DEFAULT_BUDGET,
    n_init: int = DEFAULT_N_INIT,
    n_iter: int = DEFAULT_N_ITER,
    include_random: bool = True,
) -> dict:
    if not openfoam_available():
        raise RuntimeError("OpenFOAM 14 is required for live expansion.")
    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)
    out = expansion_dir(seed)
    out.mkdir(parents=True, exist_ok=True)
    cmp_path = out / "cfd_comparison.json"
    test_rows = load_test_set(TEST_PATH)

    ai = _try_load_campaign(out / "ai", budget)
    if ai is None:
        _log(f"=== expansion seed {seed}: AI ===")
        env_ai = ExplorationEnv(seed=seed, backend="openfoam", n_iter=n_iter)
        ai = LevelSetAgent(env_ai, n_init=n_init, policy="full").run(budget=budget)
        save_campaign(ai, out / "ai")
    else:
        _log(f"=== expansion seed {seed}: AI already complete ===")

    lhs = _try_load_campaign(out / "baseline", budget)
    if lhs is None:
        _log(f"=== expansion seed {seed}: LHS ===")
        env_lhs = ExplorationEnv(seed=seed + 10_000, backend="openfoam", n_iter=n_iter)
        lhs = LatinHypercubeBaseline(env_lhs).run(budget=budget)
        save_campaign(lhs, out / "baseline")
    else:
        _log(f"=== expansion seed {seed}: LHS already complete ===")

    rand: CampaignRecord | None = None
    if include_random:
        rand = _try_load_campaign(out / "random", budget)
        if rand is None:
            _log(f"=== expansion seed {seed}: Random ===")
            env_rand = ExplorationEnv(seed=seed + 20_000, backend="openfoam", n_iter=n_iter)
            rand = RandomBaseline(env_rand).run(budget=budget)
            save_campaign(rand, out / "random")
        else:
            _log(f"=== expansion seed {seed}: Random already complete ===")

    if cmp_path.exists() and ai is not None and lhs is not None and (not include_random or rand is not None):
        existing = json.loads(cmp_path.read_text(encoding="utf-8"))
        if existing.get("ai") and existing.get("baseline") and (not include_random or existing.get("random")):
            _log(f"=== expansion seed {seed}: comparison already complete ===")
            return existing

    payload: dict = {
        "budget": budget,
        "seed": seed,
        "n_init": n_init,
        "n_iter": n_iter,
        "backend": "openfoam",
        "cohort": "expansion",
        "holdout": "artifacts/of_test.json",
        "analog_constants": analog_constant_record(),
        "ai_diagnostics": campaign_diagnostics(ai.evaluations),
        "baseline_diagnostics": campaign_diagnostics(lhs.evaluations),
        "hypotheses": ai.hypotheses,
        "discoveries": ai.discoveries,
        "ai": _score_record(ai, test_rows),
        "baseline": _score_record(lhs, test_rows),
        **run_metadata(openfoam_executed=True, extra={"random_seed": seed}),
    }
    if rand is not None:
        payload["random_diagnostics"] = campaign_diagnostics(rand.evaluations)
        payload["random"] = _score_record(rand, test_rows)
    cmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(
        f"=== expansion seed {seed} done  "
        f"AI R={payload['ai']['final'].get('unstable_recall')}  "
        f"LHS R={payload['baseline']['final'].get('unstable_recall')}  "
        f"RND R={(payload.get('random') or {}).get('final', {}).get('unstable_recall')}"
    )
    return payload


def load_original() -> list[dict]:
    rows = []
    for seed in ORIGINAL_SEEDS:
        path = ROOT / "artifacts" / f"cfd_study_s{seed}" / "cfd_comparison.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def load_expansion() -> list[dict]:
    rows = []
    for seed in EXPANSION_SEEDS:
        path = expansion_dir(seed) / "cfd_comparison.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda d: d["seed"])
    return rows


def _cohort_summary(payloads: list[dict], label: str) -> dict:
    rows = []
    for p in payloads:
        ai = p["ai"]["final"]
        lhs = p["baseline"]["final"]
        rnd = (p.get("random") or {}).get("final") or {}
        rows.append(
            {
                "seed": p["seed"],
                "ai_recall": ai.get("unstable_recall"),
                "lhs_recall": lhs.get("unstable_recall"),
                "random_recall": rnd.get("unstable_recall"),
                "ai_precision": ai.get("unstable_precision"),
                "lhs_precision": lhs.get("unstable_precision"),
                "random_precision": rnd.get("unstable_precision"),
                "ai_f1": ai.get("unstable_f1"),
                "lhs_f1": lhs.get("unstable_f1"),
                "random_f1": rnd.get("unstable_f1"),
                "ai_n_u": ai.get("n_unstable_found"),
                "lhs_n_u": lhs.get("n_unstable_found"),
                "random_n_u": rnd.get("n_unstable_found"),
            }
        )
    n = len(rows)
    n_recall = sum((r["ai_recall"] or -1) > (r["lhs_recall"] or -1) for r in rows)
    n_nu = sum((r["ai_n_u"] or -1) > (r["lhs_n_u"] or -1) for r in rows)
    n_f1 = sum((r["ai_f1"] or -1) > (r["lhs_f1"] or -1) for r in rows)
    return {
        "label": label,
        "n_seeds": n,
        "seeds": [r["seed"] for r in rows],
        "recall": {
            "ai": _stats([r["ai_recall"] for r in rows]),
            "lhs": _stats([r["lhs_recall"] for r in rows]),
            "random": _stats([r["random_recall"] for r in rows]),
        },
        "precision": {
            "ai": _stats([r["ai_precision"] for r in rows]),
            "lhs": _stats([r["lhs_precision"] for r in rows]),
            "random": _stats([r["random_precision"] for r in rows]),
        },
        "f1": {
            "ai": _stats([r["ai_f1"] for r in rows]),
            "lhs": _stats([r["lhs_f1"] for r in rows]),
            "random": _stats([r["random_f1"] for r in rows]),
        },
        "n_unstable_found": {
            "ai": _stats([r["ai_n_u"] for r in rows]),
            "lhs": _stats([r["lhs_n_u"] for r in rows]),
            "random": _stats([r["random_n_u"] for r in rows]),
        },
        "paired_ai_vs_lhs": {
            "ai_higher_unstable_recall": f"{n_recall}/{n}" if n else "0/0",
            "ai_higher_unstable_f1": f"{n_f1}/{n}" if n else "0/0",
            "ai_more_unstable_evaluations": f"{n_nu}/{n}" if n else "0/0",
        },
        "per_seed": rows,
    }


def write_summaries() -> tuple[dict, dict]:
    original = _cohort_summary(load_original(), "frozen_original_n8")
    expansion_payloads = load_expansion()
    expansion = _cohort_summary(expansion_payloads, "post_development_expansion")
    pooled = _cohort_summary(load_original() + expansion_payloads, "pooled_original_plus_expansion")
    exp_doc = {
        "status": "COMPLETE" if len(expansion_payloads) == len(EXPANSION_SEEDS) else "PARTIAL",
        "protocol": {
            "budget": DEFAULT_BUDGET,
            "n_init": DEFAULT_N_INIT,
            "n_iter": DEFAULT_N_ITER,
            "holdout": "artifacts/of_test.json",
            "methods": ["LevelSetAgent(policy=full)", "LatinHypercubeBaseline", "RandomBaseline"],
            "analog_constants": analog_constant_record(),
            "note": (
                "Expansion campaigns were generated after the original n=8 live study. "
                "They were not used to tune constants or acquisition. "
                "Original artifacts/cfd_study_s* files were not rewritten."
            ),
        },
        "planned_seeds": EXPANSION_SEEDS,
        "summary": expansion,
        **run_metadata(openfoam_executed=True),
    }
    pooled_doc = {
        "status": exp_doc["status"],
        "original": original,
        "expansion": expansion,
        "pooled": pooled,
        "interpretation": (
            "Report the frozen n=8 result as the original live study. "
            "The expansion and pooled numbers are additional post-development evidence. "
            "They were not used to retune analog constants or the acquisition."
        ),
        **run_metadata(openfoam_executed=True),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(exp_doc, indent=2), encoding="utf-8")
    POOLED_PATH.write_text(json.dumps(pooled_doc, indent=2), encoding="utf-8")
    return exp_doc, pooled_doc


def run_expansion(
    seeds: list[int] | None = None,
    *,
    budget: int = DEFAULT_BUDGET,
    n_init: int = DEFAULT_N_INIT,
    n_iter: int = DEFAULT_N_ITER,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    seeds = list(seeds or EXPANSION_SEEDS)
    if not openfoam_available():
        payload = {
            "status": "PENDING",
            "reason": "OpenFOAM 14 is not available.",
            "planned_seeds": seeds,
            **run_metadata(openfoam_executed=False),
        }
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    errors: list[dict] = []
    workers = max(1, int(workers))
    if workers == 1:
        for seed in seeds:
            try:
                run_one_seed(seed, budget=budget, n_init=n_init, n_iter=n_iter)
            except Exception as exc:  # pragma: no cover - live solver
                _log(f"FAILED expansion seed {seed}: {exc}")
                errors.append({"seed": seed, "error": str(exc)})
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(run_one_seed, seed, budget=budget, n_init=n_init, n_iter=n_iter): seed
                for seed in seeds
            }
            for fut in as_completed(futs):
                seed = futs[fut]
                try:
                    fut.result()
                except Exception as exc:  # pragma: no cover - live solver
                    _log(f"FAILED expansion seed {seed}: {exc}")
                    errors.append({"seed": seed, "error": str(exc)})
    exp_doc, _pooled = write_summaries()
    if errors:
        exp_doc["seed_errors"] = errors
        SUMMARY_PATH.write_text(json.dumps(exp_doc, indent=2), encoding="utf-8")
    return exp_doc


def run_g_sweep_mesh_verification(
    *,
    s: float = 0.10,
    n: int = 9,
    n_iter: int = 90,
    mesh_scales: tuple[float, ...] = (0.70, 1.00, 1.40),
) -> dict:
    """Re-run the committed g-slice geometry at several meshes. Does not overwrite g_sweep.json."""
    if not openfoam_available():
        payload = {
            "status": "PENDING",
            "reason": "OpenFOAM 14 is not available.",
            "original_g_sweep_not_rewritten": True,
            **run_metadata(openfoam_executed=False),
        }
        GSWEEP_VERIFY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    g_values = [float(v) for v in np.linspace(0.05, 1.95, n)]
    cases = []
    for mesh_scale in mesh_scales:
        rows = []
        for gv in g_values:
            x = {"g": gv, "d": 0.10, "a": 0.55, "s": s, "o": 0.50}
            report = run_mixer(x, n_iter=n_iter, mesh_scale=mesh_scale)
            if report.Cvalid:
                *_, sigma, S, phase = _acoustics(
                    report.tau, report.Um, x["o"], report.R_spatial, report.compactness
                )
            else:
                sigma = float("nan")
                S = float("nan")
                phase = float("nan")
            rows.append(
                {
                    **x,
                    "mesh_scale": mesh_scale,
                    "n_iter": n_iter,
                    "tau": report.tau,
                    "R_spatial": report.R_spatial,
                    "compactness": report.compactness,
                    "Um": report.Um,
                    "sigma_analog": sigma,
                    "S": S,
                    "phase": phase,
                    "stable": int(S < 1.0) if S == S else None,
                    "Cvalid": bool(report.Cvalid),
                    "notes": report.notes,
                }
            )
            _log(f"g-sweep mesh={mesh_scale} g={gv:.2f} Cvalid={report.Cvalid} sigma={sigma}")
        valid = [r for r in rows if r["Cvalid"]]
        intervals = unstable_runs_1d([r["g"] for r in valid], [r["sigma_analog"] for r in valid])
        cases.append(
            {
                "mesh_scale": mesh_scale,
                "n_iter": n_iter,
                "n_unstable_intervals": len(intervals),
                "intervals": [[float(a), float(b)] for a, b in intervals],
                "rows": rows,
            }
        )
    n_int = [c["n_unstable_intervals"] for c in cases]
    payload = {
        "status": "COMPLETE",
        "s": s,
        "n_iter": n_iter,
        "mesh_scales": list(mesh_scales),
        "original_g_sweep_not_rewritten": True,
        "two_intervals_on_every_tested_mesh": all(v == 2 for v in n_int),
        "analog_constants": analog_constant_record(),
        "cases": [
            {
                "mesh_scale": c["mesh_scale"],
                "n_unstable_intervals": c["n_unstable_intervals"],
                "intervals": c["intervals"],
            }
            for c in cases
        ],
        "full_cases": cases,
        "note": (
            "Same 1-D slice as artifacts/g_sweep.json (fixed d,a,s,o). "
            "This asks whether the two analog-unstable intervals survive mesh change. "
            "It is not a 5-D topology proof."
        ),
        **run_metadata(openfoam_executed=True, extra={"random_seed": None}),
    }
    GSWEEP_VERIFY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_expansion_figures() -> list[Path]:
    import matplotlib.pyplot as plt

    written: list[Path] = []
    if not POOLED_PATH.exists():
        return written
    pooled = json.loads(POOLED_PATH.read_text(encoding="utf-8"))
    figdir = ROOT / "artifacts" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    def _bar(ax, cohort, title):
        methods = ["ai", "lhs"]
        labels = ["AI", "LHS"]
        if cohort["recall"]["random"]["n"]:
            methods.append("random")
            labels.append("Random")
        means = [cohort["recall"][m]["mean"] or 0 for m in methods]
        sds = [cohort["recall"][m]["sd"] or 0 for m in methods]
        ax.bar(labels, means, yerr=sds, color=["#1f618d", "#7f8c8d", "#b2babb"][: len(labels)], capsize=4)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("hold-out unstable recall")

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    _bar(axes[0], pooled["original"], f"Frozen original n={pooled['original']['n_seeds']}")
    _bar(axes[1], pooled["expansion"], f"Expansion n={pooled['expansion']['n_seeds']}")
    _bar(axes[2], pooled["pooled"], f"Pooled n={pooled['pooled']['n_seeds']}")
    fig.suptitle("Live CFD unstable recall. Original n=8 remains the primary study.")
    fig.tight_layout()
    path = figdir / "live_cfd_expansion_recall.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    if GSWEEP_VERIFY_PATH.exists():
        gs = json.loads(GSWEEP_VERIFY_PATH.read_text(encoding="utf-8"))
        if gs.get("status") == "COMPLETE":
            fig, ax = plt.subplots(figsize=(7.4, 4.2))
            for case in gs.get("full_cases") or []:
                rows = [r for r in case["rows"] if r.get("Cvalid")]
                ax.plot(
                    [r["g"] for r in rows],
                    [r["sigma_analog"] for r in rows],
                    "o-",
                    label=f"mesh_scale={case['mesh_scale']}  intervals={case['n_unstable_intervals']}",
                )
            ax.axhline(0.0, color="#444", ls="--", lw=1)
            ax.set_xlabel("pattern class g")
            ax.set_ylabel(r"$\sigma_{\mathrm{analog}}$")
            ax.set_title("g-slice at live n_iter=90 vs relative mesh (original g_sweep.json not rewritten)")
            ax.legend(fontsize=8)
            path = figdir / "g_sweep_mesh_verification.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    return written


def write_derived_artifacts() -> list[Path]:
    """Sample-efficiency + expansion figures from already-written campaign files."""
    from cswe.sample_efficiency import analyze_from_payloads, write_figure as write_se_figure

    written: list[Path] = []
    exp = load_expansion()
    if exp:
        se = analyze_from_payloads(
            exp,
            source="artifacts/cfd_expansion_s*/cfd_comparison.json",
            out_path=ROOT / "artifacts" / "sample_efficiency_expansion.json",
            claim="Post-development expansion only. Original n=8 files were not rewritten.",
        )
        path = write_se_figure(se, ROOT / "artifacts" / "figures" / "sample_efficiency_expansion.png")
        if path is not None:
            written.append(path)
        pooled = load_original() + exp
        se_p = analyze_from_payloads(
            pooled,
            source="frozen cfd_study_s* plus cfd_expansion_s*",
            out_path=ROOT / "artifacts" / "sample_efficiency_pooled.json",
            claim="Pooled original+expansion. Report n=8 as the primary live study.",
        )
        path = write_se_figure(se_p, ROOT / "artifacts" / "figures" / "sample_efficiency_pooled.png")
        if path is not None:
            written.append(path)
        write_summaries()
    written.extend(write_expansion_figures())
    return written
