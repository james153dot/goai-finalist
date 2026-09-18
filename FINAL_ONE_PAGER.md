# One-pager: AI-guided exploration of a combustion-stability analog

GOAI 2026 Track 3 · Type II Open Exploration · finalist package.
Author: James "Dave" Lu.

## Problem

High-fidelity scientific simulations are expensive. Exhaustive search of a
five-dimensional injector / operating analog is impractical. The useful
question is how to spend a **fixed** solver budget.

## Method

A 2-D laminar OpenFOAM mixer returns mixing delay and spatial overlap. A
declared closed-closed 1L Rayleigh map converts those features into
σ_analog. An agent fits a Gaussian process to σ_analog, hunts a missing
regime, then straddles σ_analog = 0. Latin hypercube and uniform random
use the same budget, bounds, and hold-out. The environment is a
combustion-stability **analog**, not a flight-engine model.

## Key result

Across eight live OpenFOAM campaigns (budget 16), mean hold-out unstable
recall was approximately **0.68 (AI) vs 0.33 (LHS)**, with similar mean
precision (0.86 vs 0.85) and F1 0.76 vs 0.46. AI had higher recall in
**7 of 8** seeds. No statistical significance is claimed on n = 8.
A later 16-seed live expansion is additional post-development evidence
(AI recall 0.65 vs LHS 0.50 vs Random 0.49; 11/16) and was not used to
retune; n = 8 remains the primary live study. The high-g 1-D interval is
absent on a coarser mesh.

## Discovery

Live seed 14 independently encountered analog-unstable samples at both low
and high pattern-class g. That observation motivated a controlled g-sweep,
which showed two unstable **intervals** on a fixed-low-swirl 1-D slice.
The agent did not map two full five-dimensional disconnected regions.

## Negative result

Seed 35 reverses: LHS recall 0.56 vs AI 0.44. A later frozen one-shot
hold-out (seed 101) also had higher LHS recall (0.76 vs 0.59) while AI
found more unstables. The high-g interval disappears under ω + 10% and
under V_crit = 0.035. Near-threshold unlike-impinging analog labels flip
at a 40-iteration solver cap. All of these are retained.

## Reproducibility

Committed atlas, hold-out, campaign logs, V&V, and the frozen final-validation
set are inspectable from the repository. Install with a standard venv / pip
path or with uv. `cswe reproduce` skips live foam when OpenFOAM is missing
and does not invent CFD. Historical JSONL is not rewritten.

## Broader significance

The reusable object is a protocol: freeze the analog and the policy, spend
a matched expensive-solver budget, score recovery of a rare scientific
regime, and keep the failures. AI does not replace the simulator; it
chooses the next experiment.
