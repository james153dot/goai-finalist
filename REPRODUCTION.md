# Reproduction

OpenFOAM 14 at `/opt/openfoam14`. `FOAM_SIGFPE=0` is set by the wrapper.

## 1. Mixing atlas and hold-out (OpenFOAM)

```bash
source /opt/openfoam14/etc/bashrc
uv sync
uv run cswe atlas --n 32 --seed 7 --n-iter 100 --workers 4
uv run cswe test-set --n 24 --seed 123 --n-iter 90 --workers 4
uv run cswe atlas-report
```

Atlas: 32 Latin-hypercube cases + 3 classical injector analogs →
`artifacts/mixing_atlas.json` (35/35 converged in the committed atlas).
Independent hold-out: `artifacts/of_test.json` (24 cases, never used to train
the atlas or the campaigns).

## 2. Physics figures (live OpenFOAM)

```bash
uv run cswe swirl-sweep --g 0.15 --n 8 --n-iter 90 --out artifacts/swirl_sweep.json
uv run cswe g-sweep --s 0.10 --n 9 --n-iter 90 --out artifacts/g_sweep.json
uv run cswe classical
```

## 3. Where AI is useful (atlas, many seeds)

```bash
uv run cswe seed-study --n-seeds 24 --budget 16 --n-init 5 --out artifacts/seed_study.json
```

The agent’s first `n_init` points are themselves a Latin hypercube so the
comparison is not “random start vs designed start.” Remaining budget is
regime-seeking then `σ = 0` straddle. Scoring uses the OpenFOAM hold-out.

## 4. Live OpenFOAM AI vs LHS

```bash
uv run cswe cfd-study --budget 16 --seed 11 --n-init 5 --n-iter 90 --out artifacts/cfd_study_s11
uv run cswe cfd-study --budget 16 --seed 14 --n-init 5 --n-iter 90 --out artifacts/cfd_study_s14
uv run cswe cfd-study --budget 16 --seed 19 --n-init 5 --n-iter 90 --out artifacts/cfd_study_s19
```

Each study is 16 sequential `foamRun` evaluations per method. Keep every seed.

## 5. Dashboard demo campaigns (atlas, budget 48)

```bash
uv run cswe run --budget 48 --seed 11 --out artifacts/demo
uv run cswe run --budget 48 --seed 7 --out artifacts/seed7
uv run cswe run --budget 48 --seed 19 --out artifacts/seed19
```

## 6. Tests and dashboard

```bash
uv run pytest
uv run streamlit run app/dashboard.py --server.port 48217 --server.address 0.0.0.0
```

`tests/test_acoustics.py` and `tests/test_metrics.py` do not need OpenFOAM.
Tests that call `simulate` skip if `artifacts/mixing_atlas.json` is missing.

No commercial APIs. No closed-source models. No dimensional injector drawings.
