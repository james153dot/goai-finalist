# AI-Guided Discovery of Combustion-Stability Windows

GOAI Track 3 · Type II open exploration · second-round package.
Author: James "Dave" Lu.

OpenFOAM determines spatial and temporal mixing features from a frozen 2-D
laminar dual-jet chamber. An acoustic model converts those features into a
hypothesis-level Rayleigh stability indicator. The project therefore evaluates
an **autonomous exploration methodology for combustion-stability analogs**, not
predictive stability of a real rocket combustor.

OpenFOAM supplies physically generated mixing fields from which the stability
analog is constructed. It does not predict combustion instability.

Public outputs stay at abstract design principles.

## Claim levels

| Level | Quantity | Source |
| --- | --- | --- |
| Observed from OpenFOAM | mixing delay τ, spatial overlap R_spatial, proxy field q_proxy(x) | dual-jet `foamRun` |
| Derived from the analog | σ_analog, stable / unstable label | frozen closed-closed 1L Rayleigh map |
| Inferred by the agent | reconstructed σ_analog = 0 contour, predicted unstable set | GP + straddle / regime-seeking |

```
geometry x = [g, d, a, s, o]
        ↓
OpenFOAM mixing field  (T, U)
        ↓
τ, R_spatial, q_proxy(x) = Var_y[T](x)
        ↓
σ_analog = n R_spatial cos(ωτ) − D
        ↓
observation  {σ_analog, stable/unstable}
        ↓
GP active explorer
```

## Heat-release proxy

q_proxy(x) is the **cross-stream variance of mixture fraction** at station x.

Large Var_y[T] means the two inlet streams are still unmixed there, so
mixing-limited reaction could still occur. Fully mixed stations (Var → 0)
contribute no further proxy heat release. This is not a finite-rate flame
and is not 4T(1−T), which would peak after the gases are already uniform.

R_spatial = ∫ q_proxy p dx / ∫ q_proxy dx with p(x) = cos(πx/L) (injector-face
pressure antinode of a closed-closed 1L analog).

## Analog constants

ω, n-index prefactors, and damping D set the *scale* of σ_analog so that both
regimes exist in the box. They are not measured chamber data.

The *ordering* of classical injector analogs is not coming from those
constants: like-on-like, unlike-impinging, and swirl-coaxial all have
n-index ≈ 0.79. Discrimination is from OpenFOAM τ (Rayleigh phase) and
R_spatial.

Sensitivity (`cswe sensitivity`) freezes the OpenFOAM fields and perturbs

D ∈ {0.8 D0, D0, 1.2 D0},   ω0 ∈ {0.9 ω0, ω0, 1.1 ω0}.

The exact σ_analog = 0 contour moves. On the live g-slice:

- D ± 20%: two unstable intervals persist. Classical like-on-like remains
  analog-unstable; unlike-impinging and swirl-coaxial remain analog-stable.
- ω0 − 10%: two intervals persist and widen. Unlike-impinging crosses into
  analog-unstable (it was already near threshold).
- ω0 + 10%: the high-g interval disappears. Like-on-like remains analog-unstable.
  Atlas unstable fraction falls from 34% to 17%.

In every setting the agent still finds more unstable evaluations than Latin
hypercube, and mean hold-out unstable recall remains higher (widest gap at
ω0 + 10%, where the unstable class is rarer). Qualitative topology on this
slice is therefore damping-robust and frequency-sensitive. That limitation
is part of the result, not a footnote.

Artifacts: `artifacts/sensitivity.json`, `artifacts/figures/sensitivity.png`.

## Environment contract

**Fixed:** chamber L = 80 mm, H = 20 mm, laminar viscosity, closed-closed 1L
mode, Rayleigh-from-mixing analog, threshold σ_analog = 0.

**Explorable:** x = [g, d, a, s, o] — pattern class, orifice-size spread,
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

Unstable recall on a hold-out OpenFOAM set with labels y_i = 1{σ_analog(x_i) > 0}:

    Recall_U = TP_U / (TP_U + FN_U).

Overall classification accuracy can remain high by predicting the dominant
stable regime. Recall_U measures whether an exploration strategy reconstructs
the scientifically important minority regime.

Boundary MAE. Let B = { i in hold-out : |σ_i| < 0.20 }. Then

    E_B = (1/|B|) Σ_{i ∈ B} |σ̂(x_i) − σ_i|

where σ̂ is a GP fit on the campaign's σ_analog values. This is hold-out error
in predicted growth rate near the analog threshold, not Euclidean distance to
a contour in parameter space.

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

### Did the agent sample both intervals?

Yes, in live OpenFOAM campaign **seed 14**, independently of the later g-sweep:

| t | g | s | σ_analog | analog class |
| --- | --- | --- | --- | --- |
| 2 | 0.10 | 0.45 | +0.32 | unstable, low g |
| 3 | 1.80 | 0.21 | +0.06 | unstable, high g |
| 6 | 0.52 | 0.27 | +0.21 | unstable, low g |
| 14 | 1.92 | 0.04 | +0.40 | unstable, high g |

The g-sweep then characterized the topology suggested by those samples.
Figure: `artifacts/figures/seed14_both_g_intervals.png`. Seed 11 did not place
an unstable point in the high-g interval; that seed is kept.

### Adaptive search vs Latin hypercube

A 24-seed atlas study (budget 16, scored on 24 independent OpenFOAM tests)
establishes direction. Live `foamRun` campaigns verify that the advantage
survives when every evaluation is a fresh solver run. Live campaigns are
few; treat them as independent replicates, not a population estimate.

Atlas (n = 24 seeds), mean:

| | AI | LHS |
| --- | --- | --- |
| Unstable recall | 0.62 | 0.46 |
| Unstable evaluations found | 7.5 | 4.5 |
| Boundary MAE E_B | 0.123 | 0.167 |
| Volume accuracy | 0.83 | 0.79 |

Live OpenFOAM (budget 16). Per-seed recall is plotted as individual markers
in `artifacts/figures/live_cfd_strip.png`. With the original three seeds
11 / 14 / 19:

| seed | AI recall | LHS recall | AI n_unstable | LHS n_unstable |
| --- | --- | --- | --- | --- |
| 11 | 0.67 | 0.22 | 7 | 5 |
| 14 | 0.67 | 0.00 | 9 | 3 |
| 19 | 0.78 | 0.22 | 6 | 5 |

Additional live seeds, if present, live in `artifacts/cfd_study_s*/`.

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
- ω, n, and D are analog constants. Sensitivity to D and ω0 is reported.
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

Problem / environment 45% — frozen chamber, explorable injector, OpenFOAM
mixing fields, Rayleigh analog with declared constants, classical injector
analogs, limitations.

Exploration signal 35% — adaptive search on σ_analog = 0, pre-registered
signals, minority-class efficiency, both g-intervals sampled in live seed 14.

Verifiability 15% — JSONL logs, seeds, hold-out OpenFOAM test set, sensitivity,
`pytest`, `cswe reproduce`.

Open-source 5% — MIT, no APIs, no closed models.

## License

MIT. Mixing fields are synthetic 2-D CFD, not engine data.
