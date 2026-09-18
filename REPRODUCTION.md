# Reproduction

This file is the judge-facing reproduction entry point for the GOAI Open
Exploration **finalist** package.

## Environment and key versions

- Python: **>= 3.11** (`pyproject.toml`)
- Exact Python package resolution: **`uv.lock`** (optional lockfile)
- Package/environment manager: **pip + venv** (supported) or **uv** (optional)
- Live CFD solver: **OpenFOAM 14**
- Expected OpenFOAM installation for the wrapper: `/opt/openfoam14`
- `FOAM_SIGFPE=0` is set by the wrapper.

The committed atlas, independent hold-out, and campaign logs are sufficient
for the minimum reproduction path and dashboard. **OpenFOAM is not required
to inspect the committed results or run atlas-backed campaigns.** It is
required for a fresh live-CFD evaluation, numerical verification, or the
frozen final-validation hold-out. `cswe reproduce` skips only the live foam
evaluation when OpenFOAM is missing.

Third-party software and licenses are disclosed in `THIRD_PARTY.md`.
Artifact/schema provenance is documented in `ARTIFACT_PROVENANCE.md`.

## Minimum path (judges) — standard pip

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
python3 -m pytest
cswe reproduce
python3 tools/check_open_exploration.py
streamlit run app/dashboard.py
```

Dashboard port used in this project: `48217`.

```bash
streamlit run app/dashboard.py --server.port 48217 --server.address 0.0.0.0
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
cswe reproduce
python tools/check_open_exploration.py
streamlit run app/dashboard.py --server.port 48217 --server.address 0.0.0.0
```

If `python` is not on `PATH`, use `python3` in the same commands. Fresh
OpenFOAM runs need WSL/Linux because the wrapper expects the OpenFOAM 14
Linux environment.

### Optional uv path

```bash
uv sync
uv run cswe reproduce
uv run python tools/check_open_exploration.py
uv run pytest
uv run streamlit run app/dashboard.py --server.port 48217 --server.address 0.0.0.0
```

`cswe reproduce` runs one like-on-like OpenFOAM case **if OpenFOAM 14 is
present**, a budget-12 atlas AI-vs-LHS pair, scores against
`artifacts/of_test.json` when present, and writes figures from committed
artifacts. It does not rebuild the full atlas and does not require OpenFOAM
to finish.

Optional analysis (no new live CFD):

```bash
cswe sample-efficiency
cswe ablation-study
cswe verify
cswe final-validation
```

`verify` and `final-validation` write **PENDING** JSON if OpenFOAM is absent.
Committed COMPLETE live runs are already in `artifacts/verification.json`
and `artifacts/final_validation.json` (`openfoam_executed: true`).
They do not invent CFD numbers.

### Expected outputs

A successful minimum run should produce all of the following:

1. `python3 -m pip install -e ".[dev]"` (or `uv sync`) resolves the environment from `pyproject.toml`.
2. `cswe reproduce` exits successfully and writes
   `artifacts/reproduce/`, including AI and LHS campaign logs and
   `summary.json`.
3. If OpenFOAM 14 is unavailable, `reproduce` prints that the live foam case
   is skipped; this is expected and does not prevent the atlas-backed
   reproduction path.
4. If `artifacts/of_test.json` exists, the short campaigns are scored on that
   independent hold-out. The hold-out is not used for training or query
   selection.
5. `python3 tools/check_open_exploration.py` prints `PASS` for mandatory
   source/document checks. Missing OpenFOAM products are `PENDING`/`WARNING`,
   not fabricated passes.
6. `python3 -m pytest` completes without test failures.
7. The Streamlit process starts and serves the dashboard, including the
   **Final Demo — 90 seconds** tab, from committed artifacts.

## Full CFD rebuild

The following path regenerates the principal CFD-derived artifacts. It is
substantially more expensive than the judge minimum path.

```bash
source /opt/openfoam14/etc/bashrc
python3 -m pip install -e ".[dev]"
cswe atlas --n 32 --seed 7 --n-iter 100 --workers 4
cswe test-set --n 24 --seed 123 --n-iter 90 --workers 4
cswe atlas-report
cswe swirl-sweep --g 0.15 --n 8 --n-iter 90 --out artifacts/swirl_sweep.json
cswe g-sweep --s 0.10 --n 9 --n-iter 90 --out artifacts/g_sweep.json
cswe seed-study --n-seeds 24 --budget 16 --n-init 5 --out artifacts/seed_study.json
cswe sensitivity --n-seeds 16 --budget 16
cswe tau-sensitivity
cswe cfd-study --budget 16 --seed 11 --n-init 5 --n-iter 90 --out artifacts/cfd_study_s11
cswe verify
cswe final-validation
cswe figures
python3 tools/check_open_exploration.py
python3 -m pytest
```

The committed primary live study contains eight seeds
`[8, 11, 14, 19, 23, 26, 32, 35]`. Rebuilding all eight requires running
`cswe cfd-study` once per seed with the same budget, initialization count,
and iteration cap. `artifacts/cfd_live_summary.json` records the aggregate
results. Do not rewrite those historical logs merely to refresh wording.

## Configuration and result definitions

The independent hold-out `artifacts/of_test.json` is never used to train the
atlas or to choose the next live OpenFOAM query. Analog-constant sensitivity
holds the OpenFOAM mixing fields fixed and changes only `D` and `ω0`.

The frozen final-validation hold-out, when generated, is a **new** file
(`artifacts/final_validation_holdout.json`) with seed 101. Atlas predictions
are never substituted for that check.

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
- Defense / one-pager: `FINAL_DEFENSE.md`, `FINAL_ONE_PAGER.md`
- V&V / frozen validation / seed 35: `VERIFICATION.md`, `FINAL_VALIDATION.md`, `FAILURE_ANALYSIS.md`

No commercial APIs, closed-source models, external datasets, proprietary
engine measurements, or dimensional flight-injector drawings are used.
