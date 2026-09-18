# Final defense notes

This is a GOAI 2026 Open Exploration **finalist** package. Historical
semifinal / second-round wording in old artifacts is provenance, not the
current claim set.

## 30-second project explanation

Expensive CFD makes exhaustive search of a design space impractical. We built
a declared OpenFOAM-informed combustion-stability **analog** and asked whether
an autonomous agent can spend a fixed budget of 16 solver calls to recover
rare analog-unstable regimes faster than Latin hypercube. Across eight live
campaigns, AI hold-out unstable recall was about 0.68 versus 0.33 for LHS,
with similar mean precision. This is a method test, not a rocket-engine
predictor.

## 2-minute technical explanation

Design vector z = [g, d, a, s, o] places two inlet slots in a **fixed**
2-D laminar chamber. OpenFOAM solves incompressible mixing and a passive
mixture fraction Z (field name `T`). From that field we extract mixing delay
τ and spatial overlap R_spatial of a mixing-availability proxy
q_mix(x) = Var_y[Z](x). A declared closed-closed 1L Rayleigh analog maps
those features to σ_analog. Unstable iff σ_analog > 0.

The agent fits a GP to σ_analog. After a Latin-hypercube start it hunts a
missing class, then straddles σ = 0. LHS and Random use the same budget,
bounds, solver, and hold-out. Scoring is hold-out Recall_U, Precision_U,
F1_U, near-boundary σ MAE, and volume accuracy of that GP.

Live confirmation: eight independent `foamRun` campaigns, seeds
[8, 11, 14, 19, 23, 26, 32, 35], budget 16. Seed 35 reverses and is kept.

## Central scientific claim

AI recovers the scientifically important minority regime using fewer
expensive solver evaluations than non-adaptive space-filling on this analog.

## Strongest numerical evidence

Committed live n = 8, `artifacts/cfd_live_summary.json`:

- Unstable recall 0.68 ± 0.13 (AI) vs 0.33 ± 0.21 (LHS); AI higher in 7/8
- F1 0.76 vs 0.46; AI higher in 7/8
- Mean precision 0.86 vs 0.85 (near-tie)
- Sample-efficiency curves: `artifacts/figures/sample_efficiency_live.png`
- Seed 14 independently found unstable 5-D samples at low and high g

No p-value is claimed on n = 8.

A later frozen one-shot OpenFOAM hold-out (seed 101, 48 new cases) is in
`FINAL_VALIDATION.md`. On that single draw LHS hold-out recall was higher
(0.76 vs 0.59) while AI found more unstables (7 vs 4). It was not used to
retune. Mesh/iteration V&V is in `VERIFICATION.md`.

## Why AI is necessary

Space-filling wastes calls on the majority stable class. The scientific
object is the rare unstable window. Adaptive selection is the only part of
the pipeline that can concentrate a 16-call budget there. Ablations on the
atlas ask which policy ingredient is associated with that advantage; they
do not prove a unique cause.

## Why OpenFOAM is used

The agent must query an expensive, physically generated mixing field, not a
planted algebraic island. OpenFOAM supplies τ, R_spatial, compactness, and
U_m. It does **not** predict combustion instability.

## Exact meaning of σ_analog

σ_analog = α n R_spatial cos(ω τ) − D, with α = 1.45, D = 0.08,
ω = ω0 (0.92 + 0.16 o), ω0 = 11.
n = 0.50 + 0.28 tanh(C − 1) + 0.16 (1 − U_m) + 0.10 (o − 0.5)².
Unstable iff σ_analog > 0. This is a declared analog indicator, not a
measured growth rate of a combustor mode.

## Why this is not a real-engine predictor

2-D, laminar, non-reacting. q_mix is unmixedness, not heat release. Chamber
geometry is a fixed analog box. ω, n, α, D are analog constants. Classical
injector names are geometry analogs, not flight hardware.

## Seed 35 explanation

LHS beat AI on recall (0.56 vs 0.44), F1, and unstable count. AI started
unstable at t = 0, then hold-out recall collapsed to 0 at budgets 11–13
after a run of stables. The seed is retained. See `FAILURE_ANALYSIS.md`.

## Sensitivity-result explanation

Holding OpenFOAM mixing fields fixed: D ± 20% keeps two g-slice intervals;
ω − 10% keeps/widens them; **ω + 10% removes the high-g interval**.
Re-reading stored q_mix profiles: **V_crit = 0.035** also removes it.
Those are conditional / negative results, not footnotes.

## Why Random / LHS are appropriate baselines

The question is allocation of a fixed expensive budget versus non-adaptive
sampling. LHS is the standard space-filling design; the agent itself starts
with LHS, so only later selection differs. Random is the weaker unstructured
control, retained on the atlas study. They do not use future information.

## Likely judge questions

**Is this a rocket-engine stability model?**
No. It is a combustion-stability analog and an exploration-method testbed.

**Did you cherry-pick seeds?**
No. The live set is eight pre-listed seeds. Seed 35 loses and is kept.

**Did seed 14 map two disconnected 5-D regions?**
No. It found unstable 5-D samples at low and high g. A later 1-D g-sweep
showed two intervals on that slice.

**Is n = 8 significant?**
Not claimed. Report 7/8 paired comparisons and means ± sample sd.

**What did the frozen final CFD validation show?**
Seed 101, 48 new OpenFOAM tests, budget 16: AI recall 0.59, LHS 0.76,
Random 0.53. AI found 7 unstables vs 4 for each baseline. Not used to retune.
The n = 8 live study remains the multi-seed CFD evidence.

**Cconv vs Cvalid?**
Historical rows use Cconv. Current code writes Cvalid (solver-valid, not a
residual certificate) and reads `row.get("Cvalid", row.get("Cconv", False))`.

## Limitations

- Analog constants set the scale of σ_analog
- High-g interval is assumption-dependent
- Atlas interpolator is smoother than a new foamRun
- n = 8 live seeds; no population inference
- Near-threshold unlike-impinging analog labels flip at a 40-iteration cap
- One frozen hold-out (seed 101) had higher LHS recall than AI

## Next scientific experiment

Keep the frozen policy. Test a minority-class refresh after a reconstructed
recall collapse on atlas seeds first (`FAILURE_ANALYSIS.md`), then a new
live OpenFOAM seed that is not 35 or 101. Separately, require iteration
caps ≥ 90 for near-threshold analog claims, as the V&V study showed.
