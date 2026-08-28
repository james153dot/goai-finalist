# Reproduction

OpenFOAM 14 at `/opt/openfoam14`. `FOAM_SIGFPE=0` is set by the wrapper.

## Minimum path (judges)

```bash
uv sync
uv run cswe reproduce
uv run pytest
uv run streamlit run app/dashboard.py --server.port 48217 --server.address 0.0.0.0
```

`reproduce` runs one like-on-like OpenFOAM case if the solver is present, a
budget-12 atlas AI vs LHS pair, scores against `artifacts/of_test.json` when
that file exists, and writes figures. It does not rebuild the full atlas.

## Full CFD rebuild

```bash
source /opt/openfoam14/etc/bashrc
uv sync
uv run cswe atlas --n 32 --seed 7 --n-iter 100 --workers 4
uv run cswe test-set --n 24 --seed 123 --n-iter 90 --workers 4
uv run cswe atlas-report
uv run cswe swirl-sweep --g 0.15 --n 8 --n-iter 90 --out artifacts/swirl_sweep.json
uv run cswe g-sweep --s 0.10 --n 9 --n-iter 90 --out artifacts/g_sweep.json
uv run cswe seed-study --n-seeds 24 --budget 16 --n-init 5 --out artifacts/seed_study.json
uv run cswe sensitivity --n-seeds 16 --budget 16
uv run cswe cfd-study --budget 16 --seed 11 --n-init 5 --n-iter 90 --out artifacts/cfd_study_s11
uv run cswe figures
uv run pytest
```

Hold-out `of_test.json` is never used to train the atlas or to choose the next
live OpenFOAM query. Analog-constant sensitivity freezes mixing fields and only
changes D and ω0.

Index: `artifacts/manifest.json`.

No commercial APIs. No closed-source models. No dimensional injector drawings.
