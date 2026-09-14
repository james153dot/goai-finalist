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

```bash
cswe final-validation
```
