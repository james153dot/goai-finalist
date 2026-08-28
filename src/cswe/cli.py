"""Typer CLI for reproducible campaigns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from cswe.agent import LevelSetAgent
from cswe.baseline import LatinHypercubeBaseline, RandomBaseline
from cswe.environment import ExplorationEnv
from cswe.logging_utils import compare_campaigns, save_campaign
from cswe.metrics import learning_curve, load_test_set, score_against_test

app = typer.Typer(help="Combustion-stability window explorer (GOAI Track 3 Type II).")


@app.command()
def run(
    budget: int = typer.Option(48, help="Matched simulation budget for AI and baseline."),
    seed: int = typer.Option(7, help="Random seed written into logs."),
    n_init: int = typer.Option(8, help="Space-filling start size for the AI agent."),
    out: Path = typer.Option(Path("artifacts/run"), help="Output directory."),
    baseline: str = typer.Option("lhs", help="Baseline: lhs or random."),
) -> None:
    """Run a budget-matched AI campaign and a non-adaptive baseline."""
    env_ai = ExplorationEnv(seed=seed)
    agent = LevelSetAgent(env_ai, n_init=n_init)
    ai_record = agent.run(budget=budget)

    env_base = ExplorationEnv(seed=seed + 10_000)
    if baseline == "random":
        base_record = RandomBaseline(env_base).run(budget=budget)
    else:
        base_record = LatinHypercubeBaseline(env_base).run(budget=budget)

    save_campaign(ai_record, out / "ai")
    save_campaign(base_record, out / "baseline")
    comparison = compare_campaigns(ai_record, base_record)
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {out}")
    typer.echo(json.dumps(comparison, indent=2))


@app.command()
def demo() -> None:
    """Print a single simulation to confirm the environment is alive."""
    env = ExplorationEnv(seed=0)
    x = {"g": 1.0, "d": 0.2, "a": 0.5, "s": 0.6, "o": 0.4}
    result = env.evaluate(x)
    typer.echo(json.dumps(result.as_dict(), indent=2))


@app.command()
def atlas(
    n: int = typer.Option(36, help="Latin-hypercube OpenFOAM mixing cases."),
    seed: int = typer.Option(7),
    n_iter: int = typer.Option(120, help="SIMPLE iterations per case."),
    workers: int = typer.Option(4),
) -> None:
    """Build the OpenFOAM mixing atlas used by the acoustic layer."""
    from cswe.mixing import ATLAS_PATH, build_atlas

    built = build_atlas(n=n, seed=seed, n_iter=n_iter, workers=workers)
    n_ok = sum(1 for r in built.rows if r["Cconv"])
    taus = [r["tau"] for r in built.rows if r["Cconv"]]
    typer.echo(f"Wrote {ATLAS_PATH}  converged={n_ok}/{len(built.rows)}  tau=[{min(taus):.4f},{max(taus):.4f}]")


@app.command("test-set")
def test_set(
    n: int = typer.Option(20),
    seed: int = typer.Option(123),
    n_iter: int = typer.Option(100),
    workers: int = typer.Option(4),
) -> None:
    """Build an independent OpenFOAM hold-out set for scoring campaigns."""
    from cswe.mixing import TEST_PATH, build_test_set

    rows = build_test_set(n=n, seed=seed, n_iter=n_iter, workers=workers)
    n_u = sum(r.get("stable", 1) == 0 for r in rows)
    typer.echo(f"Wrote {TEST_PATH}  n={len(rows)} unstable={n_u}")


@app.command("cfd-study")
def cfd_study(
    budget: int = typer.Option(16, help="OpenFOAM evaluations per method."),
    seed: int = typer.Option(11),
    n_init: int = typer.Option(5),
    n_iter: int = typer.Option(90),
    out: Path = typer.Option(Path("artifacts/cfd_study")),
) -> None:
    """AI vs LHS where each evaluation is a live OpenFOAM run. Score on of_test.json."""
    from cswe.metrics import campaign_diagnostics
    from cswe.mixing import TEST_PATH

    env_ai = ExplorationEnv(seed=seed, backend="openfoam", n_iter=n_iter)
    ai_record = LevelSetAgent(env_ai, n_init=n_init).run(budget=budget)
    env_base = ExplorationEnv(seed=seed + 10_000, backend="openfoam", n_iter=n_iter)
    base_record = LatinHypercubeBaseline(env_base).run(budget=budget)
    save_campaign(ai_record, out / "ai")
    save_campaign(base_record, out / "baseline")
    payload: dict = {
        "budget": budget,
        "seed": seed,
        "backend": "openfoam",
        "ai_diagnostics": campaign_diagnostics(ai_record.evaluations),
        "baseline_diagnostics": campaign_diagnostics(base_record.evaluations),
        "hypotheses": ai_record.hypotheses,
        "discoveries": ai_record.discoveries,
    }
    if TEST_PATH.exists():
        test_rows = load_test_set(TEST_PATH)
        payload["ai"] = {
            "final": score_against_test(ai_record.evaluations, test_rows),
            "curve": learning_curve(ai_record.evaluations, test_rows),
            "hypotheses": ai_record.hypotheses,
        }
        payload["baseline"] = {
            "final": score_against_test(base_record.evaluations, test_rows),
            "curve": learning_curve(base_record.evaluations, test_rows),
            "hypotheses": base_record.hypotheses,
        }
    (out / "cfd_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(json.dumps(payload, indent=2))


@app.command("swirl-sweep")
def swirl_sweep(
    g: float = typer.Option(0.15, help="Pattern class held fixed (0.15 = like-on-like family)."),
    n: int = typer.Option(8, help="Number of swirl stations."),
    n_iter: int = typer.Option(90),
    out: Path = typer.Option(Path("artifacts/swirl_sweep.json")),
) -> None:
    """OpenFOAM sweep in swirl analog s. The figure a scientist can argue with."""
    from cswe.openfoam import run_mixer
    from cswe.physics import _acoustics

    rows = []
    for i, s in enumerate(np.linspace(0.05, 0.95, n)):
        x = {"g": g, "d": 0.10, "a": 0.55, "s": float(s), "o": 0.50}
        report = run_mixer(x, n_iter=n_iter)
        n_index, omega, rayleigh, sigma, S, phase = _acoustics(
            report.tau, report.Um, x["o"], report.R_spatial, report.compactness
        )
        row = {
            **x,
            "tau": report.tau,
            "Um": report.Um,
            "R_spatial": report.R_spatial,
            "compactness": report.compactness,
            "x_q": report.x_q,
            "sigma": sigma,
            "S": S,
            "phase": phase,
            "stable": int(S < 1.0),
            "Cconv": report.Cconv,
            "q_profile": report.q_profile,
            "x_profile": report.x_profile,
            "p_profile": report.p_profile,
        }
        rows.append(row)
        typer.echo(
            f"[{i+1}/{n}] s={s:.2f} tau={report.tau:.4f} R={report.R_spatial:.3f} "
            f"xq={report.x_q:.4f} sigma={sigma:.3f} stable={row['stable']}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"g": g, "rows": rows}, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {out}")


@app.command("g-sweep")
def g_sweep(
    s: float = typer.Option(0.10, help="Swirl analog held fixed."),
    n: int = typer.Option(9),
    n_iter: int = typer.Option(90),
    out: Path = typer.Option(Path("artifacts/g_sweep.json")),
) -> None:
    """OpenFOAM sweep in pattern class g. Primary organizing variable on this analog."""
    from cswe.openfoam import run_mixer
    from cswe.physics import _acoustics

    rows = []
    for i, gv in enumerate(np.linspace(0.05, 1.95, n)):
        x = {"g": float(gv), "d": 0.10, "a": 0.55, "s": s, "o": 0.50}
        report = run_mixer(x, n_iter=n_iter)
        n_index, omega, rayleigh, sigma, S, phase = _acoustics(
            report.tau, report.Um, x["o"], report.R_spatial, report.compactness
        )
        row = {
            **x,
            "tau": report.tau,
            "Um": report.Um,
            "R_spatial": report.R_spatial,
            "compactness": report.compactness,
            "x_q": report.x_q,
            "sigma": sigma,
            "S": S,
            "phase": phase,
            "stable": int(S < 1.0),
            "Cconv": report.Cconv,
            "q_profile": report.q_profile,
            "x_profile": report.x_profile,
            "p_profile": report.p_profile,
        }
        rows.append(row)
        typer.echo(
            f"[{i+1}/{n}] g={gv:.2f} tau={report.tau:.4f} R={report.R_spatial:.3f} "
            f"sigma={sigma:.3f} stable={row['stable']}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"s": s, "rows": rows}, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {out}")


@app.command("seed-study")
def seed_study(
    n_seeds: int = typer.Option(24),
    budget: int = typer.Option(16),
    n_init: int = typer.Option(5),
    out: Path = typer.Option(Path("artifacts/seed_study.json")),
) -> None:
    """Many atlas seeds: where adaptive search beats space-filling, with failures kept."""
    from cswe.metrics import campaign_diagnostics, load_test_set, score_against_test
    from cswe.mixing import TEST_PATH

    test_rows = load_test_set(TEST_PATH) if TEST_PATH.exists() else []
    records = []
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
        records.append(rec)
        typer.echo(
            f"seed {seed}: AI first_u={rec['ai']['time_to_first_unstable']} "
            f"n_u={rec['ai']['n_unstable_found']} edge={rec['ai']['frac_evals_near_edge']:.2f} | "
            f"LHS first_u={rec['baseline']['time_to_first_unstable']} "
            f"n_u={rec['baseline']['n_unstable_found']} edge={rec['baseline']['frac_evals_near_edge']:.2f}"
        )

    def _mean(key, field):
        vals = [r[key][field] for r in records if r[key].get(field) is not None]
        return float(np.mean(vals)) if vals else None

    def _miss(key):
        return float(np.mean([r[key]["n_unstable_found"] == 0 for r in records]))

    summary = {
        "n_seeds": n_seeds,
        "budget": budget,
        "ai_mean_time_to_first_unstable": float(
            np.mean([r["ai"]["time_to_first_unstable"] if r["ai"]["time_to_first_unstable"] is not None else budget for r in records])
        ),
        "lhs_mean_time_to_first_unstable": float(
            np.mean([r["baseline"]["time_to_first_unstable"] if r["baseline"]["time_to_first_unstable"] is not None else budget for r in records])
        ),
        "ai_mean_n_unstable": _mean("ai", "n_unstable_found"),
        "lhs_mean_n_unstable": _mean("baseline", "n_unstable_found"),
        "ai_mean_frac_near_edge": _mean("ai", "frac_evals_near_edge"),
        "lhs_mean_frac_near_edge": _mean("baseline", "frac_evals_near_edge"),
        "ai_frac_missed_unstable": _miss("ai"),
        "lhs_frac_missed_unstable": _miss("baseline"),
    }
    if test_rows:
        def tmean(which, field):
            vals = [r[which][field] for r in records if r[which].get(field) is not None]
            return float(np.mean(vals)) if vals else None
        summary["ai_mean_unstable_recall"] = tmean("ai_test", "unstable_recall")
        summary["lhs_mean_unstable_recall"] = tmean("baseline_test", "unstable_recall")
        summary["ai_mean_boundary_mae"] = tmean("ai_test", "boundary_mae")
        summary["lhs_mean_boundary_mae"] = tmean("baseline_test", "boundary_mae")
        summary["ai_mean_near_boundary_sigma_mae"] = tmean("ai_test", "boundary_mae")
        summary["lhs_mean_near_boundary_sigma_mae"] = tmean("baseline_test", "boundary_mae")
        summary["ai_mean_volume_accuracy"] = tmean("ai_test", "volume_accuracy")
        summary["lhs_mean_volume_accuracy"] = tmean("baseline_test", "volume_accuracy")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"Wrote {out}")


@app.command()
def foam(
    g: float = 1.0,
    d: float = 0.2,
    a: float = 0.5,
    s: float = 0.4,
    o: float = 0.5,
    n_iter: int = 120,
    out: Path = Path("artifacts/foam_case"),
) -> None:
    """Run one live OpenFOAM mixer evaluation (not the interpolator)."""
    from cswe.openfoam import run_mixer

    report = run_mixer({"g": g, "d": d, "a": a, "s": s, "o": o}, work=out, n_iter=n_iter)
    typer.echo(json.dumps(report.__dict__, indent=2))


@app.command("atlas-report")
def atlas_report() -> None:
    """Print classical injectors and the atlas unstable fraction under current acoustics."""
    from cswe.mixing import ATLAS_PATH, get_atlas
    from cswe.physics import _acoustics, simulate
    from cswe.geometry import CLASSICAL_INJECTORS

    atlas = get_atlas()
    sigmas = []
    for r in atlas.rows:
        if not r.get("Cconv"):
            continue
        *_, sigma, S, _ = _acoustics(r["tau"], r["Um"], r["o"], r.get("R_spatial"), r.get("compactness"))
        sigmas.append(sigma)
    n_u = sum(s > 0 for s in sigmas)
    typer.echo(f"atlas {ATLAS_PATH}  n={len(sigmas)}  unstable={n_u} ({n_u/len(sigmas):.0%})")
    typer.echo(f"sigma  [{min(sigmas):.3f}, {max(sigmas):.3f}]")
    rng = np.random.default_rng(0)
    for name, x in CLASSICAL_INJECTORS.items():
        r = simulate(x, rng=rng, backend="atlas")
        typer.echo(
            f"  {name:20s} sigma={r.sigma:.3f}  S={r.S:.3f}  stable={r.stable}  "
            f"tau={r.tau:.4f}  R_spatial={r.R_spatial:.3f}"
        )


@app.command()
def classical() -> None:
    """Evaluate textbook injector analogs through the CFD-informed environment."""
    from cswe.geometry import CLASSICAL_INJECTORS
    from cswe.physics import simulate

    rng = np.random.default_rng(0)
    for name, x in CLASSICAL_INJECTORS.items():
        r = simulate(x, rng=rng)
        typer.echo(
            f"{name:20s}  S={r.S:.3f}  sigma={r.sigma:.3f}  stable={r.stable}  "
            f"tau={r.tau:.4f}  R_spatial={r.R_spatial:.3f}  compactness={r.compactness:.2f}  "
            f"backend={r.backend}"
        )


@app.command()
def figures() -> None:
    """Write static PNG figures from committed artifacts."""
    from cswe.figures import write_all

    for path in write_all():
        typer.echo(str(path))


@app.command()
def sensitivity(
    n_seeds: int = typer.Option(16),
    budget: int = typer.Option(16),
    n_init: int = typer.Option(5),
) -> None:
    """Perturb analog D and ω0 with OpenFOAM mixing fields frozen."""
    from cswe.sensitivity import OUT_PATH, run_sensitivity

    payload = run_sensitivity(n_seeds=n_seeds, budget=budget, n_init=n_init)
    typer.echo(f"Wrote {OUT_PATH}")
    typer.echo(f"second interval persists: {payload['second_unstable_interval_persists_on_g_slice']}")
    typer.echo(f"AI finds more unstables in every setting: {payload['ai_finds_more_unstables_in_every_setting']}")
    for c in payload["cases"]:
        gs = c.get("g_slice", {})
        ss = c["seed_study"]["summary"]
        typer.echo(
            f"  {c['name']:12s} intervals={gs.get('n_unstable_intervals')}  "
            f"atlas_u={c['atlas']['unstable_fraction']:.0%}  "
            f"AI recall={ss.get('ai_mean_unstable_recall')}  "
            f"LHS recall={ss.get('lhs_mean_unstable_recall')}"
        )


@app.command()
def reproduce() -> None:
    """Minimum end-to-end path: one foam case (if present), short matched campaigns, figures."""
    from cswe.figures import write_all
    from cswe.geometry import CLASSICAL_INJECTORS
    from cswe.metrics import campaign_diagnostics, load_test_set, score_against_test
    from cswe.mixing import TEST_PATH
    from cswe.openfoam import openfoam_available, run_mixer

    out = Path("artifacts/reproduce")
    out.mkdir(parents=True, exist_ok=True)
    if openfoam_available():
        report = run_mixer(CLASSICAL_INJECTORS["like_on_like"], work=out / "foam", n_iter=60)
        typer.echo(
            f"foam like-on-like  Cconv={report.Cconv}  tau={report.tau:.4f}  "
            f"R_spatial={report.R_spatial:.3f}  q_proxy compactness={report.compactness:.2f}"
        )
    else:
        typer.echo("OpenFOAM 14 not found; skipping live foam case.")

    budget = 12
    ai = LevelSetAgent(ExplorationEnv(seed=11), n_init=4).run(budget=budget)
    base = LatinHypercubeBaseline(ExplorationEnv(seed=11 + 10_000)).run(budget=budget)
    save_campaign(ai, out / "ai")
    save_campaign(base, out / "baseline")
    summary = {"ai": campaign_diagnostics(ai.evaluations), "lhs": campaign_diagnostics(base.evaluations)}
    if TEST_PATH.exists():
        test = load_test_set(TEST_PATH)
        summary["ai_test"] = score_against_test(ai.evaluations, test)
        summary["lhs_test"] = score_against_test(base.evaluations, test)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))
    for path in write_all():
        typer.echo(str(path))
    typer.echo("reproduce done")


if __name__ == "__main__":
    app()
