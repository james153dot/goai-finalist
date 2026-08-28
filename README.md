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
| Observed from OpenFOAM | mixing delay τ, spatial overlap R_spatial, mixing-availability field q_mix(x) | dual-jet `foamRun` |
| Derived from the analog | σ_analog(z), stable / unstable label | declared closed-closed 1L Rayleigh map |
| Inferred by the agent | reconstructed σ_analog = 0 contour, predicted unstable set | GP + straddle / regime-seeking |

```
design vector z = [g, d, a, s, o]
        ↓
OpenFOAM mixing field  (Z stored as field T, U)
        ↓
τ, R_spatial, q_mix(x) = Var_y[Z](x)
        ↓
σ_analog(z) = α n R_spatial cos(ωτ) − D
        ↓
observation  {σ_analog, stable/unstable}
        ↓
GP active explorer
```

The five-dimensional exploration vector is **z**. Axial chamber position is
**x**. The pipeline is then

    z → OpenFOAM → q_mix(x) → σ_analog(z).

Python campaign logs still use the dict key `x` for the design point; that is
an implementation detail, not the paper notation.

## Mixing-availability proxy

The passive scalar is denoted **Z** (mixture fraction) here for clarity. It is
stored as OpenFOAM field `T` in the current implementation. The field name
was not renamed so that the committed atlas and live CFD logs remain valid.

q_mix(x) is the **cross-stream variance of mixture fraction** at station x:

    q_mix(x) = Var_y[Z](x).

q_mix is **not a heat-release prediction**. Rayleigh's criterion concerns
heat-release fluctuations coupled to pressure; scalar variance is unmixedness.
q_mix identifies axial locations where scalar segregation remains and is the
declared mixing-side weighting used by the Rayleigh analog. Fully mixed
stations (Var → 0) contribute no further weight. This is not a finite-rate
flame and is not 4Z(1−Z), which would peak after the gases are already uniform.
JSON logs store the axial profile under the key `q_profile`.

    R_spatial = ∫ q_mix p dx / ∫ q_mix dx,    p(x) = cos(πx/L)

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

## Analog constants and n-index

ω, n-index prefactors, the scale α, and damping D set the *scale* of
σ_analog so that both regimes exist in the box. They are not measured
chamber data.

    n = 0.50 + 0.28 tanh(C − 1) + 0.16 (1 − Um) + 0.10 (o − 0.5)²

C is compactness of q_mix (max/mean of the axial profile). Um is mixedness
at the 45% axial station. g, d, a, and s enter n only through those
OpenFOAM-derived features. o also appears directly, and weakly scales the
analog frequency:

    ω = ω0 (0.92 + 0.16 o),    ω0 = 11,
    σ_analog = α n R_spatial cos(ω τ) − D,    α = 1.45,    D = 0.08.

The *ordering* of classical injector analogs is not coming from n:
like-on-like, unlike-impinging, and swirl-coaxial all have n ≈ 0.79.
Discrimination is from OpenFOAM τ (Rayleigh phase) and R_spatial.

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

The mixing-delay definition itself is a modeling choice. `cswe tau-sensitivity`
re-reads stored q_mix profiles (no CFD rerun) and perturbs

    V_crit ∈ {0.035, 0.045, 0.055},    N_bins ∈ {16, 24, 32}

one at a time. N_bins resamples the stored 24-bin profile; it is not a
re-extraction from cell centres.

Classical injector ordering (like-on-like σ > unlike-impinging σ >
swirl-coaxial σ, with like-on-like analog-unstable and swirl-coaxial
analog-stable) survives every setting. Two g-slice unstable intervals
survive N_bins ∈ {16, 24, 32} and V_crit = 0.055.

**V_crit = 0.035 collapses the high-g interval to one.** A stricter mixing
criterion moves x_m downstream, increases τ, and rotates the Rayleigh phase.
That is disclosed, not hidden: the second interval is real on the declared
definition (V_crit = 0.045, 24 bins) and is not an artifact of N_bins, but
it is not invariant to a 22% tighter variance threshold.

Artifacts: `artifacts/tau_sensitivity.json`, `artifacts/figures/tau_sensitivity.png`.

## Environment contract

**Fixed:** chamber L = 80 mm, H = 20 mm, laminar viscosity, closed-closed 1L
mode, Rayleigh-from-mixing analog, threshold σ_analog = 0.

**Explorable:** z = [g, d, a, s, o]. Implementation: `src/cswe/geometry.py`
(`DESIGN_VARIABLES`, `jet_layout`).

| Variable | Meaning | Range | What changes in OpenFOAM |
| --- | --- | --- | --- |
| g | pattern-class analog | [0, 2] | Inlet-slot centre-lines: close like-on-like pair (g≈0) → unlike-separated pair (g≈1) → coaxial-like stacked arrangement (g≈2). |
| d | orifice-size-spread analog | [0, 1] | Relative slot heights h0 = 0.13 H (1+0.55 d), h1 = 0.13 H (1−0.55 d). |
| a | impingement analog | [0, 1] | Inlet-vector polar angle θ = (0.12+0.70 a)×0.70 rad (~7°–47°); jets aimed toward each other. |
| s | swirl analog | [0, 1] | Additional slot offset and opposing cross-stream velocity (spin = 0.45 s). 2-D stand-in for swirl, not azimuthal velocity. |
| o | operating analog | [0, 1] | Bulk axial speeds u0x = U_ref (0.70+0.30 o), u1x = U_ref (1.30−0.30 o). Also weakly scales analog frequency ω. |

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
the scientifically important minority regime. A high-recall map could still
be useless if it simply paints a huge fraction of the box as unstable, so
the same GP is also scored by

    Precision_U = TP_U / (TP_U + FP_U),
    F1_U = 2 Precision_U Recall_U / (Precision_U + Recall_U).

If the GP predicts no unstable hold-out points, Precision_U = 0.

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

### Primary exhibit: seed 14 found unstable regimes at both low and high g

The adaptive campaign independently encountered unstable conditions at both
low and high pattern-class values. This motivated the subsequent
fixed-condition g-sweep, which established that a low-swirl one-dimensional
slice contains two separated unstable intervals.

Those seed-14 evaluations are full five-dimensional samples, not points on
the later slice (d, a, s, o were free):

| t | g | d | a | s | o | σ_analog | analog class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 0.10 | 0.99 | 0.07 | 0.45 | 0.12 | +0.32 | unstable, low g |
| 3 | 1.80 | 0.35 | 0.87 | 0.21 | 0.74 | +0.06 | unstable, high g |
| 6 | 0.52 | 0.51 | 0.92 | 0.27 | 0.90 | +0.21 | unstable, low g |
| 14 | 1.92 | 0.93 | 0.73 | 0.04 | 0.38 | +0.40 | unstable, high g |

The agent did not literally sample both intervals of the later 1-D slice.
It independently found unstable cases at both low and high g, which is why
that slice was run.

Primary figure: `artifacts/figures/seed14_both_g_intervals.png`.
Seed 11 did not place an unstable point at high g; that seed is kept.

### Adaptive search vs Latin hypercube

A 24-seed atlas study (budget 16, scored on 24 independent OpenFOAM tests)
establishes direction. Eight independent live `foamRun` campaigns (budget 16)
check that the advantage survives when every evaluation is a fresh solver run.
These are replicates, not a population census. No p-value is claimed on n = 8.

Atlas (n = 24 seeds), mean:

| | AI | LHS |
| --- | --- | --- |
| Unstable recall | 0.62 | 0.46 |
| Unstable precision | 0.92 | 0.91 |
| Unstable F1 | 0.72 | 0.60 |
| Unstable evaluations found | 7.5 | 4.5 |
| Near-boundary σ MAE E_{σ,boundary} | 0.123 | 0.167 |
| Volume accuracy | 0.83 | 0.79 |

Live OpenFOAM, n = 8 seeds, budget 16. Mean ± sample sd:

| | AI | LHS |
| --- | --- | --- |
| Unstable recall | 0.68 ± 0.13 | 0.33 ± 0.21 |
| Unstable precision | 0.86 ± 0.11 | 0.85 ± 0.35 |
| Unstable F1 | 0.76 ± 0.12 | 0.46 ± 0.25 |
| Unstable evaluations found | 7.5 ± 1.2 | 5.9 ± 1.6 |
| Volume accuracy | 0.84 ± 0.07 | 0.75 ± 0.08 |
| Near-boundary σ MAE E_{σ,boundary} | 0.134 ± 0.031 | 0.134 ± 0.046 |

Across eight matched live-CFD campaigns, adaptive exploration achieved higher
unstable recall than LHS in **7/8** seeds and sampled more unstable conditions
in **7/8** seeds. Unstable F1 follows the same **7/8** split. Volume accuracy
is also 7/8. The one reversal (seed 35) is retained.

Mean unstable precision is a **near-tie** (0.86 vs 0.85); AI is higher in only
1/8 seeds. LHS precision looks high because it under-predicts the unstable
class (high precision, low recall), except seed 14 where the LHS GP predicted
no unstable hold-out points (precision 0). Adaptive search therefore roughly
**doubles recall without a precision collapse** — it is not painting a huge
fraction of the box as unstable.

Near-boundary growth-rate MAE is a **tie in the mean** (0.134 vs 0.134); AI
is lower (better) in only 3/8 seeds. That is the intended argument: adaptive
sampling is advantageous when the scientific objective is recovering a rare
instability regime under an expensive evaluation budget, not that AI
dominates every metric.

| seed | AI R | LHS R | AI P | LHS P | AI F1 | LHS F1 | AI n_u | LHS n_u |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.89 | 0.67 | 1.00 | 1.00 | 0.94 | 0.80 | 9 | 8 |
| 11 | 0.67 | 0.22 | 1.00 | 1.00 | 0.80 | 0.36 | 7 | 5 |
| 14 | 0.67 | 0.00 | 0.86 | 0.00 | 0.75 | 0.00 | 9 | 3 |
| 19 | 0.78 | 0.22 | 0.88 | 1.00 | 0.82 | 0.36 | 6 | 5 |
| 23 | 0.67 | 0.22 | 0.75 | 1.00 | 0.71 | 0.36 | 7 | 6 |
| 26 | 0.67 | 0.33 | 0.86 | 1.00 | 0.75 | 0.50 | 8 | 6 |
| 32 | 0.67 | 0.44 | 0.86 | 1.00 | 0.75 | 0.62 | 8 | 7 |
| 35 | 0.44 | 0.56 | 0.67 | 0.83 | 0.53 | 0.67 | 6 | 7 |

Seed 35 reverses hold-out recall, F1, and unstable count; it is kept. Seed 14
is the exhibit for independently finding unstable conditions at both low and
high g. Markers: `artifacts/figures/live_cfd_strip.png`.

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

- 2-D, laminar, non-reacting. q_mix is scalar variance (unmixedness), not heat release.
- ω, n, α, and D are analog constants. Sensitivity to D, ω, and the τ
  definition (V_crit, N_bins) is reported, including the ω + 10% and
  V_crit = 0.035 disappearances of the high-g interval.
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
uv run cswe tau-sensitivity
uv run cswe cfd-study --budget 16 --seed 11 --out artifacts/cfd_study_s11
uv run cswe figures
```

Machine-readable index: `artifacts/manifest.json`.

## Scoring object for GOAI Type II

Problem / environment 45% — fixed-geometry chamber, explorable injector,
OpenFOAM mixing fields, Rayleigh analog with declared constants, classical
injector analogs, limitations.

Exploration signal 35% — adaptive search on σ_analog = 0, pre-registered
signals, minority-class efficiency, unstable conditions at both low and high g in live seed 14.

Verifiability 15% — JSONL logs, seeds, hold-out OpenFOAM test set, analog and
τ-definition sensitivity, `pytest`, `cswe reproduce`.

Open-source 5% — MIT, no APIs, no closed models.

## License

MIT. Mixing fields are synthetic 2-D CFD, not engine data.
