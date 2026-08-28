# AI-Guided Discovery of Combustion-Stability Windows

GOAI Track 3 · Type II open exploration · second-round package.
Author: James "Dave" Lu.

OpenFOAM determines spatial and temporal mixing features from a fixed-geometry
2-D laminar dual-jet mixing environment. An acoustic model converts those
features into a hypothesis-level Rayleigh stability indicator. The project
therefore evaluates an **autonomous exploration methodology for
combustion-stability analogs**, not predictive stability of a real rocket
combustor.

OpenFOAM supplies physically generated mixing fields from which the stability
analog is constructed. It does not predict combustion instability.

Public outputs stay at abstract design principles.

## Claim levels

| Level | Quantity | Source |
| --- | --- | --- |
| Observed from OpenFOAM | mixing delay τ, spatial overlap R_spatial, proxy field q_proxy(x) | dual-jet `foamRun` |
| Derived from the analog | σ_analog(z), stable / unstable label | declared closed-closed 1L Rayleigh map |
| Inferred by the agent | reconstructed σ_analog = 0 contour, predicted unstable set | GP + straddle / regime-seeking |

```
design vector z = [g, d, a, s, o]
        ↓
OpenFOAM mixing field  (Z stored as field T, U)
        ↓
τ, R_spatial, q_proxy(x) = Var_y[Z](x)
        ↓
σ_analog(z) = n R_spatial cos(ωτ) − D
        ↓
observation  {σ_analog, stable/unstable}
        ↓
GP active explorer
```

The five-dimensional exploration vector is **z**. Axial chamber position is
**x**. The pipeline is then

    z → OpenFOAM → q_proxy(x) → σ_analog(z).

Python campaign logs still use the dict key `x` for the design point; that is
an implementation detail, not the paper notation.

## Heat-release proxy

The passive scalar is denoted **Z** (mixture fraction) here for clarity. It is
stored as OpenFOAM field `T` in the current implementation. The field name
was not renamed so that the committed atlas and live CFD logs remain valid.

q_proxy(x) is the **cross-stream variance of mixture fraction** at station x:

    q_proxy(x) = Var_y[Z](x).

Large Var_y[Z] means the two inlet streams are still unmixed there, so
mixing-limited reaction could still occur. Fully mixed stations (Var → 0)
contribute no further proxy heat release. This is not a finite-rate flame
and is not 4Z(1−Z), which would peak after the gases are already uniform.

    R_spatial = ∫ q_proxy p dx / ∫ q_proxy dx,    p(x) = cos(πx/L)

(injector-face pressure antinode of a closed-closed 1L analog).

## Mixing delay τ

τ is defined operationally in `src/cswe/openfoam.py` (`_metrics_from_fields`),
not as a free analog constant.

Cell centres are partitioned into **24 equal-width axial bins**. In bin b,
V(x_b) is the sample variance of Z among cells whose centres fall in that
bin. If the bin has fewer than three cells, V(x_b) is set to 1.0.

Let x_m be the bin-centre of the **first** bin satisfying the absolute
mixing criterion

    V(x_m) < 0.045.

This threshold is **not** normalized by V(0). There is **no interpolation**
between bins. If no bin meets the threshold, x_m is the last bin centre.

U_b is the mean of the two inlet axial speeds,

    U_b = ½ (|u0x| + |u1x|).

Then

    τ = max(10⁻⁴, x_m / U_b).

## Analog constants

ω, n-index prefactors, and damping D set the *scale* of σ_analog so that both
regimes exist in the box. They are not measured chamber data.

The *ordering* of classical injector analogs is not coming from those
constants: like-on-like, unlike-impinging, and swirl-coaxial all have
n-index ≈ 0.79. Discrimination is from OpenFOAM τ (Rayleigh phase) and
R_spatial.

Sensitivity (`cswe sensitivity`) holds the OpenFOAM mixing fields fixed and
perturbs

D ∈ {0.8 D0, D0, 1.2 D0},   ω ∈ {0.9 ω0, ω0, 1.1 ω0}.

The exact σ_analog = 0 contour moves. On the live g-slice:

- D ± 20%: two unstable intervals persist. Classical like-on-like remains
  analog-unstable; unlike-impinging and swirl-coaxial remain analog-stable.
- ω − 10%: two intervals persist and widen. Unlike-impinging crosses into
  analog-unstable (it was already near threshold).
- ω + 10%: the high-g interval **disappears**. Like-on-like remains analog-unstable.
  Atlas unstable fraction falls from 34% to 17%.

**The existence of the high-g unstable interval is therefore not universal
within the analog; it depends on the assumed acoustic frequency.** That is a
conditional / negative result, not a footnote.

In every setting the agent still finds more unstable evaluations than Latin
hypercube, and mean hold-out unstable recall remains higher (widest gap at
ω + 10%, where the unstable class is rarer). Qualitative topology on this
slice is damping-robust and frequency-sensitive.

Artifacts: `artifacts/sensitivity.json`, `artifacts/figures/sensitivity.png`.

## Environment contract

**Fixed:** chamber L = 80 mm, H = 20 mm, laminar viscosity, closed-closed 1L
mode, Rayleigh-from-mixing analog, threshold σ_analog = 0.

**Explorable:** z = [g, d, a, s, o] — pattern class, orifice-size spread,
impingement, swirl analog, operating analog.

**Discovery signals (declared before search):**

1. A reconstructed stable/unstable window under the Rayleigh analog.
2. Whether increasing the swirl analog within a family always improves stability.
3. Whether a 1-D characterization slice is monotone in pattern class g.
4. How classical injector analogs sit on the analog window.
5. Whether adaptive search recovers the minority unstable class with fewer
   solver calls than a space-filling design of the same budget.

**Out of scope:** thrust, Isp, dimensional flight injectors, 3-D reacting LES.

## Baseline protocol

B_AI = B_LHS (16 live OpenFOAM evaluations in the primary study). Same parameter
bounds, same OpenFOAM model, same SIMPLE iteration cap, same hold-out test set,
same GP kernel family for scoring. The agent's first n_init points are themselves
a Latin hypercube, so the comparison is not random start vs designed start.
Only experiment *selection after initialization* differs.

Atlas campaigns add small observational noise; live `foamRun` campaigns do not.

## Metrics

Unstable recall on a hold-out OpenFOAM set with labels y_i = 1{σ_analog(z_i) > 0}:

    Recall_U = TP_U / (TP_U + FN_U).

Overall classification accuracy can remain high by predicting the dominant
stable regime. Recall_U measures whether an exploration strategy reconstructs
the scientifically important minority regime.

Near-boundary growth-rate MAE. Let B = { i in hold-out : |σ_i| < 0.20 }. Then

    E_{σ,boundary} = (1/|B|) Σ_{i ∈ B} |σ̂(z_i) − σ_i|

where σ̂ is a GP fit on the campaign's σ_analog values and z is the
five-dimensional design vector. This is prediction error in σ_analog among
hold-out points already near the analog threshold. It is **not** Hausdorff
distance, nearest-contour distance, or geometric MAE on the σ_analog = 0
isosurface.

JSON logs keep the key `boundary_mae` as an alias of
`near_boundary_sigma_mae`.

Volume accuracy is overall hold-out classification accuracy of the same GP.
It is reported because space-filling designs can remain competitive on global
metrics while losing on rare-regime recovery.

## Results

### Mixing characterization (OpenFOAM, before any agent)

| Injector analog | τ (s) | R_spatial | σ_analog | Analog regime |
| --- | --- | --- | --- | --- |
| like-on-like | 0.089 | 0.71 | +0.38 | unstable |
| unlike-impinging | 0.144 | 0.69 | −0.09 | stable (near threshold) |
| swirl-coaxial | 0.228 | 0.56 | −0.60 | stable |

A live OpenFOAM sweep in pattern class g at fixed low swirl analog is **not
monotone**. The like-on-like end is unstable; the slice crosses into stable
near g ≈ 0.5; a **second unstable interval** appears around g ≈ 1.5 when a
coaxial-like layout is run without swirl.

That is a 1-D slice observation. It is not a mapped disconnected pocket in
five dimensions. It demonstrates that the analog response is nonmonotonic and
motivates looking for separated unstable regions in the full space.

Increasing the swirl analog within the like-on-like family (g = 0.15) reduced
σ_analog over the tested range but did not cross the stability boundary.

### Primary exhibit: seed 14 sampled both intervals

The adaptive campaign itself encountered evidence of separated instability
behavior; the later one-dimensional sweep was used to characterize that
observation rather than manufacture it.

Live OpenFOAM campaign **seed 14**, independently of the later g-sweep:

| t | g | s | σ_analog | analog class |
| --- | --- | --- | --- | --- |
| 2 | 0.10 | 0.45 | +0.32 | unstable, low g |
| 3 | 1.80 | 0.21 | +0.06 | unstable, high g |
| 6 | 0.52 | 0.27 | +0.21 | unstable, low g |
| 14 | 1.92 | 0.04 | +0.40 | unstable, high g |

The agent independently evaluated unstable conditions at both ends
(g = 0.10, σ = +0.32 and g = 1.80, σ = +0.06), then later again at
g = 1.92, σ = +0.40.

Primary figure: `artifacts/figures/seed14_both_g_intervals.png`.
Seed 11 did not place an unstable point in the high-g interval; that seed is
kept.

### Adaptive search vs Latin hypercube

A 24-seed atlas study (budget 16, scored on 24 independent OpenFOAM tests)
establishes direction. Eight independent live `foamRun` campaigns (budget 16)
check that the advantage survives when every evaluation is a fresh solver run.
These are replicates, not a population census. No p-value is claimed on n = 8.

Atlas (n = 24 seeds), mean:

| | AI | LHS |
| --- | --- | --- |
| Unstable recall | 0.62 | 0.46 |
| Unstable evaluations found | 7.5 | 4.5 |
| Near-boundary σ MAE E_{σ,boundary} | 0.123 | 0.167 |
| Volume accuracy | 0.83 | 0.79 |

Live OpenFOAM, n = 8 seeds, budget 16. Mean ± sample sd:

| | AI | LHS |
| --- | --- | --- |
| Unstable recall | 0.68 ± 0.13 | 0.33 ± 0.21 |
| Unstable evaluations found | 7.5 ± 1.2 | 5.9 ± 1.6 |
| Volume accuracy | 0.84 ± 0.07 | 0.75 ± 0.08 |
| Near-boundary σ MAE E_{σ,boundary} | 0.134 ± 0.031 | 0.134 ± 0.046 |

Across eight matched live-CFD campaigns, adaptive exploration achieved higher
unstable recall than LHS in **7/8** seeds and sampled more unstable conditions
in **7/8** seeds. Volume accuracy follows the same 7/8 split. The one
reversal (seed 35) is retained. Near-boundary growth-rate MAE is a **tie in
the mean** (0.134 vs 0.134); AI is lower (better) in only 3/8 seeds. That is
the intended argument: adaptive sampling is advantageous when the scientific
objective is recovering a rare instability regime under an expensive
evaluation budget, not that AI dominates every metric.

| seed | AI recall | LHS recall | AI n_u | LHS n_u | AI vol | LHS vol | AI E_{σ,b} | LHS E_{σ,b} |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.89 | 0.67 | 9 | 8 | 0.96 | 0.88 | 0.079 | 0.103 |
| 11 | 0.67 | 0.22 | 7 | 5 | 0.88 | 0.71 | 0.147 | 0.099 |
| 14 | 0.67 | 0.00 | 9 | 3 | 0.83 | 0.62 | 0.125 | 0.218 |
| 19 | 0.78 | 0.22 | 6 | 5 | 0.88 | 0.71 | 0.159 | 0.137 |
| 23 | 0.67 | 0.22 | 7 | 6 | 0.79 | 0.71 | 0.162 | 0.083 |
| 26 | 0.67 | 0.33 | 8 | 6 | 0.83 | 0.75 | 0.099 | 0.175 |
| 32 | 0.67 | 0.44 | 8 | 7 | 0.83 | 0.79 | 0.134 | 0.107 |
| 35 | 0.44 | 0.56 | 6 | 7 | 0.71 | 0.79 | 0.165 | 0.148 |

Seed 35 reverses hold-out recall and unstable count; it is kept. Seed 14
remains the exhibit for sampling both g-intervals. Markers:
`artifacts/figures/live_cfd_strip.png`.

The preliminary implementation evaluated adaptive exploration primarily by
global volume reconstruction. Subsequent experiments showed that this metric
favors space-filling designs and does not directly measure recovery of rare
unstable regimes.

### Why LHS hold-out recall can fall as the budget grows

The scoring GP is refit on the campaign so far. It is not a monotone
classifier. Later Latin-hypercube samples overwhelmingly represent the
majority stable class. Hyperparameter refitting can shrink the reconstructed
unstable region, so hold-out Recall_U can decrease even while global
interpolation of σ_analog improves. Live seed 19 shows that pattern: early
recall of 1.0 at five points, then collapse toward 0.22. Space-filling can
improve global interpolation and still worsen minority-regime identification
under a small budget.

## Analog limitations

- 2-D, laminar, non-reacting. q_proxy is mixing variance, not a flame.
- ω, n, and D are analog constants. Sensitivity to D and ω is reported,
  including the ω + 10% disappearance of the high-g interval.
- The 35-case atlas interpolator is smoother than a new OpenFOAM case.
- Closed-closed 1L is a duct analog, not a full acoustic eigenproblem.
- The second unstable interval is a 9-point live g-slice at fixed low swirl.

## Run

Committed atlas, hold-out, and campaign logs are enough for the dashboard
and atlas campaigns. OpenFOAM 14 is required only to rebuild CFD or to run
live `foamRun` studies.

```bash
uv sync
uv run cswe reproduce          # one foam case if present, short campaigns, figures
uv run pytest
uv run streamlit run app/dashboard.py --server.port 48217 --server.address 0.0.0.0
```

Rebuild CFD (OpenFOAM 14):

```bash
source /opt/openfoam14/etc/bashrc
uv run cswe atlas --n 32 --seed 7 --n-iter 100 --workers 4
uv run cswe test-set --n 24 --seed 123 --n-iter 90 --workers 4
uv run cswe swirl-sweep --g 0.15 --n 8
uv run cswe g-sweep --s 0.10 --n 9
uv run cswe seed-study --n-seeds 24 --budget 16
uv run cswe sensitivity
uv run cswe cfd-study --budget 16 --seed 11 --out artifacts/cfd_study_s11
uv run cswe figures
```

Machine-readable index: `artifacts/manifest.json`.

## Scoring object for GOAI Type II

Problem / environment 45% — fixed-geometry chamber, explorable injector,
OpenFOAM mixing fields, Rayleigh analog with declared constants, classical
injector analogs, limitations.

Exploration signal 35% — adaptive search on σ_analog = 0, pre-registered
signals, minority-class efficiency, both g-intervals sampled in live seed 14.

Verifiability 15% — JSONL logs, seeds, hold-out OpenFOAM test set, sensitivity,
`pytest`, `cswe reproduce`.

Open-source 5% — MIT, no APIs, no closed models.

## License

MIT. Mixing fields are synthetic 2-D CFD, not engine data.
