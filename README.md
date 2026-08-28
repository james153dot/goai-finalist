# AI-Guided Discovery of Combustion-Stability Windows

GOAI Track 3 · Type II open exploration · second-round package.
Author: James "Dave" Lu.

Injector **mixing** is computed with **OpenFOAM 14** (2-D laminar dual-jet mixer).
Chamber **acoustics** are a frozen closed-closed first-longitudinal Rayleigh analog
driven by that field. The agent does not search a planted algebraic island.

This is an analog of liquid-rocket injector *stability windows*, not a rocket engine
and not a thrust model. Public outputs stay at abstract design principles.

## What changed after the preliminary paper

The first-round trap was scoring an adaptive agent against Latin hypercube on
**volume accuracy of a Gaussian process fit to a Gaussian process interpolator**.
Space-filling designs win that game. It is the wrong question.

The second-round package does three things a scientist can argue with:

1. **Rayleigh from the field.** Heat-release analog `q(x)` is the cross-stream
   variance of mixture fraction (where mixing-limited reaction would still be
   active). The pressure mode is `p(x) = cos(πx/L)` (injector-face antinode).
   Growth rate `σ = n · R_spatial · cos(ωτ) − damping`. Unstable iff `σ > 0`.
2. **The right efficiency claim.** Adaptive search should spend expensive solver
   calls on the minority unstable class and on the `σ = 0` contour. Metrics:
   unstable recall and boundary MAE on a held-out OpenFOAM test set, plus how
   many unstable evaluations a matched budget bought. Volume accuracy is reported
   and is *not* the headline.
3. **Live `foamRun` campaigns**, not only atlas interpolator campaigns.

## Result where AI is actually useful

Unstable conditions are **34%** of the 35-case OpenFOAM atlas (12/35). That is
the dangerous class. The question is whether adaptive search spends a **small
OpenFOAM budget** on that class better than Latin hypercube.

**Three live OpenFOAM campaigns, 16 `foamRun` evaluations per method, scored
on 24 independent OpenFOAM tests:**

| | AI (σ-straddle + regime-seeking) | Latin hypercube |
| --- | --- | --- |
| Unstable recall | **0.70** | 0.15 |
| Unstable evaluations found | **7.3** | 4.3 |
| Volume accuracy | **0.86** | 0.68 |
| Hold-out boundary MAE | **0.144** | 0.151 |

Per seed (AI / LHS): 11 → recall 0.67 / 0.22; 14 → 0.67 / **0.00**; 19 → 0.78 / 0.22.
Seed 14 is the exhibit: space-filling found three unstable points and still
reconstructed a GP that predicted the hold-out unstable class as empty. The
agent found nine unstable points and recovered two-thirds of the hold-out
unstable set. Learning curves (`artifacts/figures/live_cfd_recall_curves.png`)
show LHS seed 19 briefly at recall 1.0 with five points, then collapsing as
more space-filling samples flood the stable majority — the opposite of
spending budget on the dangerous class.

The same pattern holds on a larger atlas study (24 seeds, budget 16, same
hold-out): recall 0.62 vs 0.46, 7.5 vs 4.5 unstable evaluations, boundary MAE
0.123 vs 0.167. Volume accuracy is the weak metric and is reported anyway.

The agent starts with the same Latin-hypercube initial design as a fair baseline,
then spends the remaining budget hunting the missing class and straddling `σ = 0`.
It does **not** leak “low `g` is unstable” into the acquisition.

At a larger atlas budget of 48, the agent still collects more unstable evaluations
(seed 11: 22 vs 14; seed 7: 16 vs 11; seed 19: 20 vs 12) but hold-out *recall*
is mixed (seed 19: AI 0.78 vs LHS 0.89). That is expected: a GP fit on a
regime-seeking sample can over-represent the minority class. The efficiency
claim is the **low-budget live CFD** setting, where each OpenFOAM call is expensive.

## What the CFD showed, before any agent

| Injector analog | τ (s) | R_spatial | σ | Regime |
| --- | --- | --- | --- | --- |
| like-on-like | 0.089 | 0.71 | +0.38 | **unstable** |
| unlike-impinging | 0.144 | 0.69 | −0.09 | stable (near the edge) |
| swirl-coaxial | 0.228 | 0.56 | −0.60 | stable |

A live OpenFOAM sweep in pattern class `g` at fixed low swirl analog is **not
monotone**. The like-on-like end (`g ≲ 0.3`) is unstable; the map then crosses
into stable near `g ≈ 0.5`, and a **second unstable band** appears around
`g ≈ 1.5` when the coaxial-like layout is run **without** swirl. Swirl-coaxial
stability on this analog needs both the coaxial pattern *and* the swirl analog.
A swirl sweep **inside the like-on-like family** (`g = 0.15`) lowers `σ` but
never stabilizes. You cannot swirl your way out of like-on-like, and you cannot
drop swirl from a coaxial layout and keep the stable well. That falsifies
“more swirl always helps” as a blanket rule and is the disconnected-pocket
signal the agent was asked to look for.

## Environment contract

**Fixed:** chamber `L = 80 mm`, `H = 20 mm`, laminar viscosity, closed-closed 1L
mode, Rayleigh-from-mixing criterion, `σ_crit = 0`.

**Explorable:** `x = [g, d, a, s, o]` — pattern class, orifice-size spread,
impingement, swirl analog, operating analog.

**CFD:** two inlet slots; T = 0 / 1; steady laminar `incompressibleFluid`.
Mixing delay is the station where cross-stream mixture variance drops, divided
by bulk speed. Spatial overlap uses the same variance profile.

**Discovery signals (declared before search):**

1. A reconstructed stable/unstable window under Rayleigh-from-mixing.
2. Whether “more swirl always helps” survives the CFD mixing map.
3. Whether unstable evaluations form one connected region.
4. How classical injector analogs sit on that window.
5. Whether adaptive search recovers the unstable class with fewer solver calls
   than a space-filling design of the same budget.

**Out of scope:** thrust, Isp, dimensional flight injectors, 3-D reacting LES.

## Analog limitations (read these)

- 2-D, laminar, non-reacting. Heat release is a mixing-variance analog, not a
  finite-rate flame.
- `ω`, `n`-index prefactors, and acoustic damping are **analog constants**. They
  set the scale of σ so that both regimes exist in the box. They are not
  measured chamber data. The *ordering* of classical injectors is not coming
  from those constants: like-on-like, unlike-impinging, and swirl-coaxial all
  have n-index ≈ 0.79. Discrimination is from OpenFOAM τ (Rayleigh phase) and
  R_spatial (overlap of mixing variance with the frozen 1L mode).
- The 35-case atlas interpolator is smoother than a new OpenFOAM case. That is
  why live `foamRun` campaigns exist.
- Closed-closed 1L is a duct analog of an injector-face / nozzle-entrance pair,
  not a full acoustic eigenproblem.
- The second unstable band on the g-sweep is a 9-point live slice at fixed
  low swirl, not a fully mapped island.

## Figures

Written by `uv run cswe figures` into `artifacts/figures/`:

- `live_cfd_recall_curves.png` — hold-out unstable recall vs live OpenFOAM budget
- `g_sweep.png` — pattern-class sweep (not monotone)
- `swirl_sweep.png` — swirl analog inside the like-on-like family
- `seed_study_unstable_counts.png` — 24 atlas seeds, AI vs LHS unstable counts

## Run

```bash
source /opt/openfoam14/etc/bashrc   # or your OpenFOAM 14 install
uv sync
uv run cswe atlas --n 32 --seed 7 --n-iter 100 --workers 4
uv run cswe test-set --n 24 --seed 123 --n-iter 90 --workers 4
uv run cswe atlas-report
uv run cswe swirl-sweep --g 0.15 --n 8
uv run cswe g-sweep --s 0.10 --n 9
uv run cswe seed-study --n-seeds 24 --budget 16
uv run cswe cfd-study --budget 16 --seed 11 --out artifacts/cfd_study_s11
uv run cswe run --budget 48 --seed 11 --out artifacts/demo
uv run cswe figures
uv run pytest
uv run streamlit run app/dashboard.py --server.port 48217 --server.address 0.0.0.0
```

Live single CFD check:

```bash
uv run cswe foam --g 0.15 --d 0.08 --a 0.55 --s 0.08 --o 0.45 --out artifacts/foam_like_on_like
```

## Scoring object for GOAI Type II

Problem / environment **45%** — frozen chamber, explorable injector, OpenFOAM
mixing, Rayleigh-from-field, classical injector analogs, declared limitations.

Exploration signal **35%** — adaptive search on `σ = 0`, pre-registered
hypotheses, the minority-class efficiency result.

Verifiability **15%** — JSONL logs, seeds, hold-out OpenFOAM test set, `pytest`.

Open-source **5%** — MIT, no APIs, no closed models.

## License

MIT. Mixing fields are synthetic 2-D CFD, not engine data.
