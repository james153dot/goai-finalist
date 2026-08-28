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


if __name__ == "__main__":
    app()
