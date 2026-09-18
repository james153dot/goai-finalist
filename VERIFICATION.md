# Numerical verification (V&V)

## What is actually being verified

`cswe verify` tests whether **reported analog classifications** for three
already-used injector analogs depend strongly on numerical resolution and
solver iteration cap.

The three conditions are the committed classical analogs in
`src/cswe/geometry.py` / the README table. They are not invented for this
study:

| Condition | Role on the declared analog | Committed σ_analog (default mesh / atlas table) |
| --- | --- | ---: |
| like-on-like | clearly analog-unstable | +0.38 |
| unlike-impinging | near σ_analog = 0 | −0.09 |
| swirl-coaxial | clearly analog-stable | −0.60 |

For each condition the existing OpenFOAM mixer is run at:

- coarse relative mesh (`mesh_scale = 0.70`)
- current / default mesh (`mesh_scale = 1.00`, `nxs = 48`, same `_ny` as committed cases)
- finer relative mesh (`mesh_scale = 1.40`)

and at iteration caps `{40, 90, 160}`. Cap 90 is the live-campaign setting.

Tracked quantities: τ, R_spatial, compactness, U_m, σ_analog, stable/unstable
classification, and Cvalid.

Outputs:

- `artifacts/verification.json` — always written; includes git SHA, dirty flag,
  timestamp, random seed, configuration, and `openfoam_executed`
- `artifacts/figures/numerical_verification.png` — written **only** when
  OpenFOAM actually executed

## This does not validate the analog against a real combustor

A mesh / iteration study asks whether the *declared analog labels* are
numerically fragile. It does **not** ask whether σ_analog predicts combustion
instability in a rocket engine. OpenFOAM here is a 2-D laminar mixer. The
acoustic map is a hypothesis-level Rayleigh analog.

## Robustness (executed OpenFOAM 14)

Classification robustness is defined narrowly: for each condition, do all
solver-valid mesh / iteration settings keep the same analog-stable /
analog-unstable label?

A COMPLETE live run is in `artifacts/verification.json`
(`openfoam_executed: true`, 27 solver-valid cases) and
`artifacts/figures/numerical_verification.png`.

| Condition | Role | Labels across 9 settings | Robust? |
| --- | --- | --- | --- |
| like-on-like | clearly analog-unstable | unstable in all 9 | yes |
| swirl-coaxial | clearly analog-stable | stable in all 9 | yes |
| unlike-impinging | near σ_analog = 0 | stable *and* unstable | **no** |

Unlike-impinging flips to analog-unstable only at the **40-iteration** cap
on the default and finer meshes (`mesh_scale` 1.0 and 1.4). At the live
campaign setting (`mesh_scale` 1.0, 90 iterations) it remains analog-stable
(σ_analog ≈ −0.06), consistent with the committed classical table. Coarse
mesh at 40 iterations stays analog-stable.

σ_analog still moves with mesh and iteration even when the label is constant
(like-on-like ranges about +0.37 to +0.66). Under-iterated solves are not a
reliable classifier for a near-threshold analog.

This is numerical robustness of the declared analog, **not** validation
against a real combustor.

```bash
cswe verify
```
