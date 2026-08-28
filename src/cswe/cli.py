"""Typer CLI for reproducible campaigns."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from cswe.agent import LevelSetAgent
from cswe.baseline import LatinHypercubeBaseline, RandomBaseline
from cswe.environment import ExplorationEnv
from cswe.logging_utils import compare_campaigns, save_campaign

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


@app.command()
def classical() -> None:
    """Evaluate textbook injector analogs through the CFD-informed environment."""
    from cswe.geometry import CLASSICAL_INJECTORS
    from cswe.physics import simulate

    rng = __import__("numpy").random.default_rng(0)
    for name, x in CLASSICAL_INJECTORS.items():
        r = simulate(x, rng=rng)
        typer.echo(f"{name:20s}  S={r.S:.3f}  stable={r.stable}  tau={r.tau:.4f}  Um={r.Um:.3f}  backend={r.backend}")


if __name__ == "__main__":
    app()
