# GOAI Open Exploration Finalist Self-Check

Basis: 2026 GOAI Track 3 `AI for Research` Open Exploration sub-track
(historical semi-final participation guide plus the finalist package
additions in this repository).

The guide's core submission principle is that the work must be inspectable,
verifiable, and reproducible. For Open Exploration it requests a complete
"three-piece set": a minimal runnable exploration environment, at least one
complete exploration log, and a reference-frame design, together with README
and reproduction instructions. It also requires disclosure of external data,
models, commercial APIs, dependencies, and licenses.

## Required deliverables

- [x] **Minimal runnable exploration environment** — `src/cswe/`, committed
  atlas, CLI, and `app/dashboard.py`.
- [x] **At least one complete exploration log** — multiple
  `artifacts/cfd_study_s*/ai/exploration.jsonl` files are committed; matched
  baseline logs are committed beside them.
- [x] **Reference-frame design** — matched-budget Latin hypercube baseline;
  random baseline additionally retained in the atlas seed study.
- [x] **README** — scientific problem, environment contract, discovery
  signals, baseline protocol, metrics, results, limitations, follow-up paths,
  and judge entry points.
- [x] **Reproduction instructions** — `REPRODUCTION.md` specifies entry
  commands, installation, configuration, versions, dependencies, and expected
  outputs.
- [x] **Dependency / license / data / model / API disclosure** —
  `THIRD_PARTY.md` plus `pyproject.toml` and `uv.lock`.

## Judging dimension: Problem Definition & Environment Design Quality

- [x] Research question is bounded to autonomous exploration of a declared
  combustion-stability **analog**, not predictive flight-engine stability.
- [x] Fixed components are explicit.
- [x] Explorable components `z = [g, d, a, s, o]` and bounds are explicit.
- [x] Feedback mechanism and returned quantities are explicit.
- [x] Observed OpenFOAM quantities, analog-derived quantities, and
  agent-inferred quantities are separated.
- [x] Out-of-scope claims are explicit.

## Judging dimension: Exploration Process & Scientific/Research Signals

- [x] Discovery signals were declared before the follow-up slices.
- [x] Complete adaptive-search traces are committed.
- [x] Matched non-adaptive baselines are committed.
- [x] Positive findings are reported.
- [x] Nonmonotonic / anomalous structure is reported conservatively.
- [x] Interpretable negative/conditional results are retained: the high-`g`
  interval disappears for `ω0 + 10%` and for `V_crit = 0.035`.
- [x] An unfavorable/reversal seed (35) is retained rather than removed.
- [x] Sparse 5-D evidence is no longer presented as a one-factor causal
  falsification or proof of disconnected 5-D topology.

## Judging dimension: Inspectability & Continuability

- [x] Machine-readable JSON/JSONL artifacts and manifest are committed.
- [x] Independent hold-out is identified and kept separate from training/query
  selection.
- [x] Minimum reproduction command is documented.
- [x] Tests are documented.
- [x] Dashboard entry point is documented.
- [x] Historical schema/interpretation drift is disclosed in
  `ARTIFACT_PROVENANCE.md` instead of hidden.
- [x] Follow-up research paths are explicit.
- [x] `tools/check_open_exploration.py` checks the repository package itself
  and distinguishes mandatory source/docs from optional OpenFOAM products.
- [x] Finalist extras: `FINAL_DEFENSE.md`, `FINAL_ONE_PAGER.md`,
  `VERIFICATION.md`, `FINAL_VALIDATION.md`, `FAILURE_ANALYSIS.md`,
  `cswe verify`, `cswe final-validation`, `cswe ablation-study`,
  sample-efficiency figure from committed live curves, Final Demo tab,
  pip install path, and `.github/workflows/ci.yml`.

## Judging dimension: Open-source Contributions

- [x] Project code has an MIT license.
- [x] Environment, baseline, logging, metrics, figures, and dashboard source
  are committed.
- [x] Direct dependencies and upstream licenses are disclosed.
- [x] Exact resolved Python dependency versions are retained in `uv.lock`.
- [x] No commercial API, closed model, proprietary engine dataset, or external
  scientific dataset is required.

## Defense readiness (operational, not code-scored)

The guide separately asks teams to prepare the report/PPT/code/reproduction
materials and test network, camera, microphone, screen sharing, and demo
materials. Those operational checks are outside the repository and should be
completed before the scheduled online defense.
