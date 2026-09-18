# AI-Guided Discovery of Combustion-Stability Windows

[![CI](https://github.com/james153dot/goai-semi/actions/workflows/ci.yml/badge.svg)](https://github.com/james153dot/goai-semi/actions/workflows/ci.yml)

GOAI Track 3 · Type II Open Exploration · **2026 finalist package**.
Author: James "Dave" Lu.

Historical semifinal / second-round artifacts remain in the tree as provenance.

**Problem.** Expensive scientific simulations make exhaustive exploration impractical.

**Question.** Can an autonomous agent allocate a fixed CFD budget better than non-adaptive sampling?

**Testbed.** A declared OpenFOAM-informed combustion-stability analog.

**Finding.** Across eight live OpenFOAM campaigns, AI unstable-regime recall was approximately **0.68 vs 0.33** for Latin hypercube, with similar mean precision (0.86 vs 0.85). AI had higher recall in **7 of 8** seeds. Seed 35 is the reversal and is kept. No p-value is claimed on n = 8.

**Scientific discovery.** The adaptive campaign exposed nonmonotonic structure that motivated a controlled follow-up sweep.

**Boundary.** This is a methodological demonstration of autonomous exploration, not a predictive rocket-engine model.

OpenFOAM supplies physically generated mixing fields from which the stability
analog is constructed. It does not predict combustion instability.

Public outputs stay at abstract design principles.

Judge entry: `FINAL_ONE_PAGER.md`, `FINAL_DEFENSE.md`, Streamlit tab
**Final Demo — 90 seconds**.

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

**V_crit = 0.035 removes the high-\(g\) unstable interval, leaving only the low-\(g\) interval.** A stricter mixing
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
3. Whether the explored response contains nonmonotonic or separated stability
   structure that warrants controlled follow-up characterization.
4. How classical injector analogs sit on the analog window.
5. Whether adaptive search recovers the minority unstable class with fewer
   solver calls than a space-filling design of the same budget.

The fixed-condition g sweep was a follow-up characterization motivated by
seed 14. The general signal — nonmonotonic or separated structure — was
predeclared; the exact slice was not.

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

`Cvalid` means the OpenFOAM run completed successfully, required fields were
available, and derived metrics were finite. It is not an independently parsed
residual-convergence certificate. `Cconv` is retained as a legacy alias for
backward compatibility, and JSON row readers fall back with
`row.get("Cvalid", row.get("Cconv", False))`.

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

| | AI | LHS | Random |
| --- | --- | --- | --- |
| Unstable recall | 0.59 | 0.46 | 0.38 |
| Unstable precision | 0.88 | 0.91 | 0.79 |
| Unstable F1 | 0.69 | 0.60 | 0.47 |
| Unstable evaluations found | 7.5 | 4.5 | 3.5 |
| Near-boundary σ MAE E_{σ,boundary} | 0.123 | 0.167 | 0.195 |
| Volume accuracy | 0.82 | 0.79 | 0.75 |

All three columns are read directly from `artifacts/seed_study.json`
(24 seeds, budget 16), regenerated with
`uv run cswe seed-study --n-seeds 24 --budget 16 --n-init 5 --out artifacts/seed_study.json`.
That command runs `LevelSetAgent`, `LatinHypercubeBaseline`, and the existing
`RandomBaseline` per seed and writes `ai` / `baseline` / `random` (and their
`_test` blocks and `_mean_*` summary fields). The expensive live-CFD study
(AI vs LHS only) is unchanged.

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

## Sample efficiency

The central competition claim is that **AI recovers the scientifically
important minority regime using fewer expensive solver evaluations**.

The eight committed live campaigns were **not rerun**. Their existing
hold-out recall curves are aggregated in
`artifacts/figures/sample_efficiency_live.png` and
`artifacts/sample_efficiency_live.json`.

- Mean and median Recall_U versus live CFD evaluations, with sample sd
- n = 8 is labeled on the figure
- Area under the mean recall-vs-budget curve on the stored range [5, 16]
- Evaluations required to reach specified recall levels **only when a seed
  actually reached that level** inside budget 16

No extrapolation past budget 16. The language is solver calls saved on this
protocol, not “AI is better at everything.” Precision remains a mean near-tie;
near-boundary σ MAE is a mean tie; seed 35 loses.

## Post-development live expansion

After the frozen n = 8 study and the one-shot seed-101 validation, sixteen
additional live OpenFOAM campaigns were run **without rewriting**
`artifacts/cfd_study_s*` and **without retuning** analog constants or the
acquisition. New logs live under `artifacts/cfd_expansion_s*`. Each expansion
seed is budget-16 AI vs LHS vs Random, scored on the original
`artifacts/of_test.json`.

Report **n = 8 as the primary live study**. Expansion and pooled numbers are
additional evidence. No p-value is claimed.

Live OpenFOAM expansion, n = 16 seeds, budget 16. Mean ± sample sd:

| | AI | LHS | Random |
| --- | --- | --- | --- |
| Unstable recall | 0.65 ± 0.24 | 0.50 ± 0.25 | 0.49 ± 0.25 |
| Unstable precision | 0.85 ± 0.12 | 0.94 ± 0.08 | 0.88 ± 0.12 |
| Unstable F1 | 0.70 ± 0.20 | 0.61 ± 0.23 | 0.59 ± 0.21 |
| Unstable evaluations found | 8.1 ± 1.4 | 5.8 ± 1.5 | 5.6 ± 2.2 |

AI had higher hold-out unstable recall than LHS in **11/16** seeds, higher F1
in **10/16**, and more analog-unstable evaluations in **15/16**. The original
7/8 split does **not** reappear at the same magnitude. Seeds 50 and 56 reverse
on recall; seed 41 has AI recall 0.22 while Random reaches 0.78. Those seeds
are kept. Mean precision is higher for LHS: adaptive search still spends more
of the budget on the minority class (8.1 vs 5.8 unstable evaluations) without
being a hold-out recall lock.

Pooled original + expansion (n = 24, Random only on the 16 new seeds):

| | AI | LHS |
| --- | --- | --- |
| Unstable recall | 0.66 ± 0.20 | 0.44 ± 0.25 |
| Unstable F1 | 0.72 ± 0.17 | 0.56 ± 0.24 |
| AI higher recall | 18/24 |  |

Sample-efficiency AUC of mean recall vs budget on [5, 16]: expansion AI 4.33
vs LHS 3.42 vs Random 3.98; pooled AI 4.54 vs LHS 3.03. The frozen n = 8 AUC
(AI 4.96 vs LHS 2.25) is unchanged.

Per-seed expansion recall:

| seed | AI R | LHS R | Random R | AI n_u | LHS n_u |
| --- | ---: | ---: | ---: | ---: | ---: |
| 38 | 0.67 | 0.67 | 0.56 | 8 | 7 |
| 41 | 0.22 | 0.11 | 0.78 | 7 | 5 |
| 44 | 0.67 | 0.33 | 0.33 | 7 | 3 |
| 47 | 0.67 | 0.33 | 0.22 | 6 | 5 |
| 50 | 0.11 | 0.33 | 0.78 | 9 | 4 |
| 53 | 0.56 | 0.22 | 0.22 | 9 | 5 |
| 56 | 0.44 | 0.78 | 0.44 | 6 | 8 |
| 59 | 0.78 | 0.11 | 0.44 | 8 | 5 |
| 62 | 0.89 | 0.78 | 0.78 | 9 | 8 |
| 65 | 0.89 | 0.78 | 0.11 | 7 | 6 |
| 68 | 0.67 | 0.33 | 0.78 | 9 | 6 |
| 71 | 0.56 | 0.56 | 0.56 | 7 | 6 |
| 74 | 0.89 | 0.56 | 0.78 | 11 | 4 |
| 77 | 0.67 | 0.56 | 0.22 | 9 | 8 |
| 80 | 0.67 | 0.67 | 0.22 | 8 | 5 |
| 83 | 1.00 | 0.89 | 0.67 | 10 | 7 |

`cswe g-sweep-verify` re-ran the committed g-slice at relative meshes
0.70 / 1.00 / 1.40. It did **not** overwrite `artifacts/g_sweep.json`.

| mesh_scale | analog-unstable intervals on the 1-D slice |
| ---: | --- |
| 0.70 | 1 (like-on-like family only; high-g interval absent) |
| 1.00 | 2 (same qualitative structure as the committed sweep) |
| 1.40 | 2 (intervals wider than at mesh 1.00) |

The high-g interval is therefore mesh-sensitive, consistent with it also
disappearing under ω + 10% and V_crit = 0.035. That is a 1-D slice statement,
not a 5-D topology proof.

Commands:

```bash
cswe live-expansion --budget 16 --n-init 5 --n-iter 90 --workers 2
cswe g-sweep-verify
```

Artifacts: `artifacts/cfd_live_expansion.json`,
`artifacts/cfd_live_pooled.json`,
`artifacts/g_sweep_mesh_verification.json`,
`artifacts/figures/live_cfd_expansion_recall.png`,
`artifacts/figures/g_sweep_mesh_verification.png`.

## Frozen one-shot final validation

After the method was frozen, `cswe final-validation` generated a **new**
48-case OpenFOAM hold-out (seed 101; 17 analog-unstable) and scored the
already-frozen AI, LHS, and Random campaigns (budget 16). Atlas predictions
were not substituted.

On that single draw, hold-out unstable recall was **0.59 (AI) vs 0.76 (LHS)
vs 0.53 (Random)**. AI found more unstable evaluations (7 vs 4 vs 4). No
constant or policy was changed after seeing this. The eight-seed live study
remains the multi-seed CFD evidence.

Artifacts: `artifacts/final_validation.json`,
`artifacts/final_validation_holdout.json`.

## Numerical verification

`cswe verify` reran the three committed classical analogs at three relative
meshes and three iteration caps (27 live `foamRun` cases).

- Like-on-like stayed analog-unstable in all 9 settings.
- Swirl-coaxial stayed analog-stable in all 9 settings.
- Unlike-impinging (near σ_analog = 0) **changed label** at 40 iterations
  on the default and finer meshes. At the live setting (default mesh, 90
  iterations) it remains analog-stable.

Near-threshold analog classifications are therefore sensitive to
under-iteration. This is not combustor validation.

Artifacts: `artifacts/verification.json`,
`artifacts/figures/numerical_verification.png`.

## Which part of the AI policy actually creates the advantage?

`cswe ablation-study` compares, on the inexpensive committed atlas and the
same hold-out / budget / seed set / GP scoring:

- full current `LevelSetAgent`
- straddle-only acquisition
- uncertainty-only acquisition
- no missing-regime-hunt
- Latin hypercube
- uniform random

This is a **policy-component ablation**. It can show which ingredient is
associated with the atlas-level advantage. It does not identify a unique
causal mechanism and it is not a new live-CFD claim.

On 16 atlas seeds (budget 16, committed hold-out):

| Policy | Recall_U | F1_U | n_unstable | Time to first unstable |
| --- | ---: | ---: | ---: | ---: |
| full LevelSetAgent | 0.64 | 0.74 | 7.4 | 1.4 |
| straddle only | 0.64 | 0.74 | 7.4 | 1.4 |
| no missing-regime hunt | 0.64 | 0.74 | 7.4 | 1.4 |
| uncertainty only | 0.63 | 0.73 | 6.3 | 1.4 |
| Latin hypercube | 0.49 | 0.63 | 4.4 | 2.3 |
| uniform random | 0.44 | 0.54 | 4.1 | 2.4 |

The adaptive policies share the same LHS initialization, so time-to-first
unstable is identical. On these seeds the initialization already observed
both classes, so disabling the missing-regime hunt did not change the
16-seed means. Uncertainty-only recovered a similar hold-out map but found
fewer unstable evaluations. The contrast with LHS/Random is therefore
associated with **adaptive selection after the shared start**, not with a
demonstrated unique contribution of the hunt term on this particular seed
set.

Artifacts: `artifacts/ablation_study.json`, `artifacts/figures/ablation_study.png`.

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
- Near-threshold unlike-impinging analog labels are not robust to a 40-iteration
  solver cap (see Numerical verification).
- One frozen post-development hold-out (seed 101) had higher LHS recall than
  AI; that draw was not used for tuning.

## Run

Committed atlas, hold-out, and campaign logs are enough for the dashboard
and atlas campaigns. OpenFOAM 14 is required only to rebuild CFD or to run
live `foamRun` studies. `cswe reproduce` skips the live foam case when
OpenFOAM is missing.

Standard pip path (uv is optional):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
python3 -m pytest
cswe reproduce
python3 tools/check_open_exploration.py
streamlit run app/dashboard.py --server.port 48217 --server.address 0.0.0.0
```

uv path, if you have it:

```bash
uv sync
uv run cswe reproduce
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

## Artifact provenance and schema

The committed numerical CFD evaluations are preserved rather than rewritten to
look newer than they are. Some historical JSONL rows use the legacy validity
field `Cconv`; current code writes `Cvalid` and accepts `Cconv` as a backward-
compatibility alias. The numerical values, solver outputs, matched budgets,
and hold-out scores are not changed by this compliance pass.

The source code now treats sparse multivariate evidence conservatively: a
neighbor-based observation can **challenge** a global monotonicity or
connectedness hypothesis, but it is not presented as a controlled causal
falsification or as proof of 5-D topology. The controlled one-factor sweeps
remain the follow-up evidence. See `ARTIFACT_PROVENANCE.md` for the exact
scope of the interpretation/schema revision.

## Follow-up research paths

1. Replace the non-reacting mixing proxy with finite-rate reacting CFD and
   test whether the observed window topology survives.
2. Replace the frozen closed-closed 1L assumption with independently solved
   acoustic modes and quantify movement of `σ_analog = 0`.
3. Run mesh, iteration, and solver-convergence studies concentrated on
   candidate boundary and high-`g` cases.
4. Extend the injector analog to three dimensions and test whether adaptive
   exploration retains its rare-regime advantage.
5. Compare alternative autonomous exploration policies under the same
   OpenFOAM budget and independent hold-out protocol. A 16-seed live
   expansion (`cswe live-expansion`) already adds Random beside AI and LHS
   without rewriting the frozen n = 8 study.
6. Treat sensitivity-driven disappearances of the high-`g` interval as
   targets for controlled mechanistic follow-up rather than universal design
   conclusions.

## Mapping to the GOAI Open Exploration judging dimensions

The Open Exploration judging guide lists four dimensions. This
repository does **not** assign unofficial percentage weights to them.
This tree is the **finalist** package; older semifinal wording in
historical artifacts is left in place as provenance.

**Problem Definition & Environment Design Quality.** The problem boundary,
fixed components, explorable coordinates, feedback, and claim levels are
declared in this README and implemented in `src/cswe/`.

**Exploration Process & Scientific/Research Signals.** Predeclared discovery
signals, adaptive search, the seed-14 follow-up, the retained reversal seed,
nonmonotonic slice, and sensitivity-driven negative results are committed as
inspectable artifacts.

**Inspectability & Continuability.** `REPRODUCTION.md`, JSON/JSONL logs, the
independent hold-out set, `artifacts/manifest.json`,
`ARTIFACT_PROVENANCE.md`, tests, the dashboard, and the follow-up paths form
an extendable problem/environment package.

**Open-source Contributions.** Code is MIT licensed. The exploration
environment, baseline, artifact pipeline, dashboard, and data-processing code
are reusable. Third-party software, external data/model/API use, and licenses
are disclosed in `THIRD_PARTY.md`.

A requirement-by-requirement self-check is in
`GOAI_OPEN_EXPLORATION_CHECKLIST.md`. Judges can also run:

```bash
python3 tools/check_open_exploration.py
```

## External resources and licenses

- External datasets: **none**.
- External trained models: **none**.
- Commercial APIs: **none**.
- Proprietary engine geometry or measurements: **none**.
- CFD artifacts: synthetic 2-D OpenFOAM mixing cases generated for this
  project.
- Exact resolved Python dependency versions: `uv.lock`.
- Direct dependencies and upstream licenses: `THIRD_PARTY.md`.
- Project code license: **MIT**.

## License

MIT. Mixing fields are synthetic 2-D CFD, not engine data.
