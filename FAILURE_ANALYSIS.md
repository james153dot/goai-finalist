# Failure analysis: live seed 35

Seed 35 is the one live OpenFOAM campaign in which Latin hypercube beat the
adaptive agent on hold-out unstable recall, F1, and unstable-evaluation count.
The algorithm was not changed to erase this seed.

Historical comparison JSON for this seed still contains the legacy hypothesis
status `"falsified"`. Current source interprets sparse 5-D neighbor evidence
as a **challenge**, not a controlled causal falsification. The numbers below
are taken from the committed logs; those logs were not rewritten.

## Directly observed facts

Protocol: budget 16, n_init = 5, backend `openfoam`, hold-out
`artifacts/of_test.json` (24 rows). Files:
`artifacts/cfd_study_s35/ai/exploration.jsonl`,
`artifacts/cfd_study_s35/baseline/exploration.jsonl`,
`artifacts/cfd_study_s35/cfd_comparison.json`.

### Final scores

| | AI | LHS |
| --- | ---: | ---: |
| Hold-out unstable recall | 0.44 | 0.56 |
| Hold-out unstable precision | 0.67 | 0.83 |
| Hold-out F1 | 0.53 | 0.67 |
| Unstable evaluations found | 6 | 7 |
| Volume accuracy | 0.71 | 0.79 |
| Time to first unstable | 0 | 3 |

### AI observations (committed log)

The first five points are the Latin-hypercube initialization (acquisition `null`).

| t | g | d | a | s | o | σ_analog | class | acquisition |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 1.46 | 0.31 | 0.61 | 0.41 | 0.18 | +0.003 | unstable | init |
| 1 | 1.20 | 0.11 | 0.82 | 0.04 | 0.62 | +0.075 | unstable | init |
| 2 | 0.47 | 0.57 | 0.20 | 0.93 | 1.00 | −0.15 | stable | init |
| 3 | 1.98 | 0.99 | 0.48 | 0.68 | 0.58 | −0.23 | stable | init |
| 4 | 0.13 | 0.77 | 0.12 | 0.33 | 0.23 | +0.15 | unstable | init |
| 5 | 0.16 | 0.53 | 0.77 | 0.52 | 0.60 | +0.39 | unstable | 0.15 |
| 6 | 1.88 | 0.98 | 0.41 | 0.96 | 0.25 | −0.41 | stable | 0.39 |
| 7 | 1.60 | 0.26 | 0.02 | 0.00 | 0.90 | −0.59 | stable | 0.38 |
| 8 | 1.98 | 0.65 | 0.96 | 0.61 | 0.73 | +0.057 | unstable | 0.43 |
| 9 | 0.20 | 0.21 | 0.02 | 0.78 | 0.11 | −0.27 | stable | 0.27 |
| 10 | 1.23 | 0.05 | 0.95 | 0.96 | 0.35 | −0.46 | stable | 0.37 |
| 11 | 1.19 | 0.68 | 0.94 | 0.41 | 0.94 | −0.15 | stable | 0.55 |
| 12 | 1.49 | 0.77 | 0.49 | 0.29 | 0.62 | −0.24 | stable | 0.50 |
| 13 | 0.46 | 0.58 | 0.71 | 0.02 | 0.87 | +0.21 | unstable | 0.49 |
| 14 | 1.75 | 0.96 | 0.71 | 0.69 | 0.57 | −0.074 | stable | 0.47 |
| 15 | 0.63 | 0.29 | 0.64 | 0.97 | 0.69 | −0.50 | stable | 0.40 |

AI unstable times: t ∈ {0, 1, 4, 5, 8, 13}. After initialization the agent
spent most remaining queries at mid-to-high g with stable returns.

LHS first unstable is at t = 3 (g = 0.22, σ = +0.46). LHS finishes with
7 unstable evaluations.

### Hold-out recall evolution (committed curves)

AI: 0.11 at budget 5 → 0.67 at 7–9 → **0.00 at 11–13** → 0.44 at 15–16.

LHS: 0.22 at 5 → 0.44 at 7–11 → 0.56 at 13 → 0.67 at 15 → 0.56 at 16.

The scoring GP is refit on the campaign so far. It is not a monotone
classifier. AI's reconstructed unstable set collapsed to empty on the
hold-out at budgets 11 and 13, then partially recovered.

### GP state

The live campaign did not serialize GP hyperparameters. Recoverable state is
the committed (z, σ) pairs above and the hold-out scores of the refit GP at
each even budget in `cfd_comparison.json`. No additional GP snapshot exists.

### Where AI allocated its budget

- Initialization already contained three analog-unstable points, including a
  near-threshold point at g ≈ 1.46, σ ≈ +0.003.
- Adaptive queries after t = 5 are concentrated at high g / high s or mid g,
  with five consecutive stables (t = 9–12) immediately before the recall
  collapse.
- Only one late unstable (t = 13) was added after that collapse.

## Interpretation / hypothesis

These are hypotheses, not additional measurements.

1. **Early luck, then boundary over-concentration.** Finding an unstable on
   the first evaluation removed any missing-class hunt. The remaining policy
   is ε-greedy straddle of σ = 0. Several later queries sit near threshold
   but on the stable side and may have pulled the GP toward a smaller
   unstable region.
2. **Majority-class flooding of the GP.** The same mechanism documented for
   LHS seed 19 (recall falling as stables accumulate) appears here on the
   *adaptive* side: after budget 9 the campaign added mostly stables and
   hold-out Recall_U went to zero.
3. **LHS happened to keep a more diverse unstable set.** Space-filling does
   not chase the current GP boundary, so it continued to land unstables after
   t = 3. That is a property of this seed, not a general LHS superiority.

None of these hypotheses is proven by n = 1. They are reasons the adaptive
policy *can* underperform under a 16-call budget.

## What this is worth scientifically

A retained reversal is evidence that the method is not being edited to
guarantee a win. It also shows a concrete failure mode of straddle-on-σ:
once both classes are seen, the agent can spend the budget refining a
locally wrong boundary and temporarily erase the minority class from the
reconstructed map.

## What would we test next?

1. Freeze the scoring GP hyperparameters after the initialization block and
   ask whether the budget-11 collapse still occurs.
2. Add a hard minority-class refresh: if hold-out Recall_U (or a cheap proxy)
   drops, force one missing-class hunt even after both classes have been seen.
   Test that change on atlas seeds first; do not retune on seed 35 live CFD.
3. Repeat the live protocol with a larger budget (32) on the same seed to see
   whether the collapse is transient.
4. Log GP length-scales and predicted unstable volume after every query so
   the collapse is inspectable without refitting from JSONL.

## Expansion reversals (not used to retune)

A later 16-seed live expansion (`artifacts/cfd_expansion_s*`) also contains
hold-out recall reversals. They are kept. Analog constants were not changed.

| seed | AI recall | LHS recall | Random recall | note |
| ---: | ---: | ---: | ---: | --- |
| 50 | 0.11 | 0.33 | 0.78 | AI sampled 9 unstables but the reconstructed map missed the hold-out minority |
| 56 | 0.44 | 0.78 | 0.44 | LHS recall win; AI found fewer unstables (6 vs 8) |
| 41 | 0.22 | 0.11 | 0.78 | AI beats LHS but Random wins this seed |

The frozen n = 8 study, including seed 35, was not rewritten.
