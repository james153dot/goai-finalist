# Reproduction

## Mixing atlas (OpenFOAM 14)

```bash
source /opt/openfoam14/etc/bashrc
uv sync
uv run cswe atlas --n 32 --seed 7 --n-iter 100 --workers 4
```

Writes `artifacts/mixing_atlas.json` (35 converged cases in the committed atlas: 32 LHS + 3 classical injectors). τ range 0.086–0.268 s.

A single live case:

```bash
uv run cswe foam --g 1.0 --d 0.2 --a 0.5 --s 0.4 --o 0.5 --n-iter 120 --out artifacts/foam_case
```

## Exploration campaigns (atlas, no solver)

```bash
uv run cswe run --budget 48 --seed 11 --out artifacts/demo
uv run cswe run --budget 48 --seed 7 --out artifacts/seed7
uv run cswe run --budget 48 --seed 19 --out artifacts/seed19
uv run cswe classical
```

Committed demo logs are seed 11. Seed 7 is a known AI Gaussian-process collapse and must be kept in the writeup.

## Tests

```bash
uv run pytest
```

Tests skip if `artifacts/mixing_atlas.json` is missing.

No commercial APIs. No closed-source models. No dimensional injector drawings.
