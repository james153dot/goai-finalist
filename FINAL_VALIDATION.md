# Frozen final validation

The algorithm, constants, discovery criteria, and evaluation metrics were frozen before the final validation set was generated.

That sentence is the contract. `cswe final-validation` is a **one-shot,
post-development** check. It records the frozen state, then — only if
OpenFOAM 14 is available — generates a **new** independent hold-out and
scores the already-frozen AI method, Latin hypercube, and uniform random
against it.

## What is recorded before any new CFD

- current git commit SHA
- whether the working tree is dirty
- analog constants (`α`, `D`, `ω0`, n-index prefactors, τ definition)
- acquisition configuration (full `LevelSetAgent`, n_init, ε, straddle)
- parameter bounds
- final-validation seed (default **101**, not in the live n=8 list)
- hold-out size (default 48)
- timestamp and `openfoam_executed`

The existing hold-out `artifacts/of_test.json` is **not** reused as the
final-validation set. A new file
`artifacts/final_validation_holdout.json` is written only when OpenFOAM runs.

## What is not allowed

- Tuning constants, thresholds, acquisition, or metrics after seeing the result
- Substituting atlas interpolator predictions for a claimed independent
  final CFD validation
- Fabricating hold-out rows when OpenFOAM is missing

## If OpenFOAM is unavailable

`artifacts/final_validation.json` is written with `"status": "PENDING"`.
That is the honest result. Inspect committed live n=8 campaigns for the
historical CFD evidence; do not treat PENDING as a pass or a fail of the
method.

## Executed result (not used for tuning)

A COMPLETE live run is committed: seed **101**, hold-out **48** new
`foamRun` cases (17 analog-unstable), budget **16**, n_iter **90**.
Hold-out file: `artifacts/final_validation_holdout.json`.
`artifacts/of_test.json` was not reused.

| Method | Recall_U | Precision_U | F1_U | Unstable evals found | E_σ,boundary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen AI | 0.59 | 0.83 | 0.69 | 7 | 0.102 |
| LHS | 0.76 | 0.87 | 0.81 | 4 | 0.123 |
| Random | 0.53 | 1.00 | 0.69 | 4 | 0.131 |

On this **one** frozen draw, Latin hypercube has higher hold-out unstable
recall than the frozen agent, while the agent sampled more analog-unstable
conditions. That split is retained. It does not replace the eight-seed live
study, and it was **not** used to change constants, acquisition, or metrics.

```bash
cswe final-validation
```
