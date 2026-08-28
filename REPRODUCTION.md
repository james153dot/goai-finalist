# Reproduction

## Exact second-round command

```bash
uv sync
uv run cswe run --budget 48 --seed 7 --n-init 8 --out artifacts/demo
```

Expected artifacts:

| Path | Role |
| --- | --- |
| `artifacts/demo/ai/exploration.jsonl` | Step-by-step AI evaluations |
| `artifacts/demo/ai/campaign.json` | Seeds, discoveries, hypotheses |
| `artifacts/demo/baseline/exploration.jsonl` | Latin-hypercube baseline |
| `artifacts/demo/comparison.json` | Hold-out reconstruction comparison |

The comparison is stochastic in the Gaussian-process fit. Hold-out accuracy should remain above the baseline for seed 7; if it does not, report the negative result. That is a valid Type II outcome.

Random seeds are the CLI `--seed` (AI environment and GP) and `--seed + 10000` (baseline environment). All parameters live in `src/cswe/physics.py`.

## Dependencies

Pinned by `uv.lock` after `uv sync`. No commercial APIs. No closed-source models. No external scientific datasets.

## What not to ship

Dimensional orifice diameters, chamber drawings, propellant mass-flow schedules, or any package that could be read as a flight-injector specification. Keep public discussion at the level of qualitative window structure.
