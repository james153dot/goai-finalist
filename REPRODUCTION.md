# Reproduction

This file is the judge-facing reproduction entry point for the GOAI Open
Exploration package.

## Environment and key versions

- Python: **>= 3.11** (`pyproject.toml`)
- Exact Python package resolution: **`uv.lock`**
- Package/environment manager: **uv**
- Live CFD solver: **OpenFOAM 14**
- Expected OpenFOAM installation for the wrapper: `/opt/openfoam14`
- `FOAM_SIGFPE=0` is set by the wrapper.

The committed atlas, independent hold-out, and campaign logs are sufficient
for the minimum reproduction path and dashboard. **OpenFOAM is not required
to inspect the committed results or run atlas-backed campaigns.** It is
required for a fresh live-CFD evaluation or full CFD rebuild.

Third-party software and licenses are disclosed in `THIRD_PARTY.md`.
Artifact/schema provenance is documented in `ARTIFACT_PROVENANCE.md`.

## Minimum path (judges)

From the repository root:

```bash
uv sync
uv run cswe reproduce
uv run python tools/check_open_exploration.py
uv run pytest
uv run streamlit run app/dashboard.py --server.port 48217 --server.address 0.0.0.0
```

`cswe reproduce` runs one like-on-like OpenFOAM case **if OpenFOAM 14 is
present**, a budget-12 atlas AI-vs-LHS pair, scores against
`artifacts/of_test.json` when present, and writes figures. It does not rebuild
the full atlas.

### Expected outputs

A successful minimum run should produce all of the following:

1. `uv sync` resolves the environment from `pyproject.toml` / `uv.lock`.
2. `uv run cswe reproduce` exits successfully and writes
   `artifacts/reproduce/`, including AI and LHS campaign logs and
   `summary.json`.
3. If OpenFOAM 14 is unavailable, `reproduce` prints that the live foam case
   is skipped; this is expected and does not prevent the atlas-backed
   reproduction path.
4. If `artifacts/of_test.json` exists, the short campaigns are scored on that
   independent hold-out. The hold-out is not used for training or query
   selection.
5. `uv run python tools/check_open_exploration.py` prints `PASS` for the
   required submission-package checks.
6. `uv run pytest` completes without test failures.
7. The Streamlit process starts and serves the dashboard on
   `http://localhost:48217`.

On Windows PowerShell, the same commands work when `uv` is on `PATH`. For a
future fresh OpenFOAM run, use WSL/Linux because the committed wrapper expects
the OpenFOAM 14 Linux environment.

## Full CFD rebuild

The following path regenerates the principal CFD-derived artifacts. It is
substantially more expensive than the judge minimum path.

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
uv run cswe tau-sensitivity
uv run cswe cfd-study --budget 16 --seed 11 --n-init 5 --n-iter 90 --out artifacts/cfd_study_s11
uv run cswe figures
uv run python tools/check_open_exploration.py
uv run pytest
```

The committed primary live study contains eight seeds
`[8, 11, 14, 19, 23, 26, 32, 35]`. Rebuilding all eight requires running
`cswe cfd-study` once per seed with the same budget, initialization count,
and iteration cap. `artifacts/cfd_live_summary.json` records the aggregate
results.

## Configuration and result definitions

The independent hold-out `artifacts/of_test.json` is never used to train the
atlas or to choose the next live OpenFOAM query. Analog-constant sensitivity
holds the OpenFOAM mixing fields fixed and changes only `D` and `ω0`.

Notation: design vector **z** = `[g, d, a, s, o]`; axial coordinate **x**;
mixture fraction **Z** (OpenFOAM field name `T`). Mixing availability
`q_mix(x) = Var_y[Z](x)` is unmixedness, not heat release. Mixing delay `τ`
is the first of 24 axial bins with `Var_y[Z] < 0.045`, then
`τ = x_m / U_b`.

Near-boundary growth-rate MAE `E_{σ,boundary}` is hold-out
`|σ_hat(z) - σ|` among points with `|σ| < 0.20`; it is not geometric contour
distance. Unstable precision and F1 are reported alongside recall.

`cswe tau-sensitivity` re-reads stored `q_mix` profiles without a CFD rerun.

## Artifact and disclosure index

- Machine-readable artifact index: `artifacts/manifest.json`
- Artifact/schema history: `ARTIFACT_PROVENANCE.md`
- Dependency/data/model/API/license disclosure: `THIRD_PARTY.md`
- GOAI Open Exploration self-check: `GOAI_OPEN_EXPLORATION_CHECKLIST.md`

No commercial APIs, closed-source models, external datasets, proprietary
engine measurements, or dimensional flight-injector drawings are used.
