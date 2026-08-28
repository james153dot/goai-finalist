# AI-Guided Discovery of Combustion-Stability Windows

GOAI Track 3 · Type II open exploration · second-round package.

This repository is the **minimal runnable exploration environment** promised in the preliminary problem-definition paper. An agent searches a frozen low-order liquid-rocket combustion–acoustic model for the boundary between stable and unstable regimes. It does not optimize thrust or specific impulse.

The chamber, propellant family, acoustic boundaries, solver, and stability criterion stay fixed. The agent only chooses the next nondimensional injector / operating analog to evaluate.

## How to proceed in the competition

You are in the **second round** (semifinals). Type I (algorithm) and Type II (open exploration) are ranked separately. This project is Type II.

| Stage | Dates (handbook; official notice wins) | What you must deliver |
| --- | --- | --- |
| Second round | 25 Aug – **3 Sep** | Runnable environment + ≥1 full execution log + baseline + README / reproduction notes |
| Second-round review | 4–10 Sep | No new work unless the committee asks. Top 20 go to the finals |
| Finals | **22 Sep**, Hangzhou | Pitch deck, live demo, one-pager, exploration report |
| GOAI Day | 23 Sep | Awards |

Treat **3 September** as the hard deadline unless an email or the track page says otherwise. Check the official track page, participant group, and mail daily. Mentorship, scientist Q&A, and compute/API allocations for shortlisted teams are announced there, not in this repo.

### What judges score (Type II)

1. **Problem definition and environment design — 45%**  
   Real, scoped problem. Clear fixed vs explorable parts and feedback.
2. **Exploration process and scientific signal — 35%**  
   A window, a shift, a counterexample, a stable negative, or a problem revision. Negative results count if they are pre-registered and explained.
3. **Verifiability and extensibility — 15%**  
   Baseline, logs, seeds, follow-up path, reusable environment package.
4. **Open-source contribution — 5%**  
   License, install, reuse.

Do not spend the week on higher-fidelity CFD. The handbook wants a **reviewable environment**, not a flight engine. Public outputs must stay at **abstract design principles** — no dimensional injector drawings or propellant flow packages.

### Second-round checklist (this repo)

- [x] Frozen n-τ chamber with five explorable analogs
- [x] Pre-registered discovery signals and a “more swirl always helps” hypothesis
- [x] AI level-set agent vs budget-matched Latin-hypercube baseline
- [x] JSONL logs, campaign metadata, comparison metrics
- [x] Reproduction entry point (`cswe run`) and dashboard
- [ ] After you run: push the public repo URL and submit through the official portal
- [ ] Disclose APIs, licenses, and seeds in the submission form
- [ ] If invited: apply for mentor hours and any compute quota immediately

### After 3 September, if you reach the finals

Write the exploration report around **what the signals mean**, not around model architecture. Include failed runs, the swirl counterexample, the disconnected unstable pocket, and what a next-fidelity model would freeze vs explore. Prepare a  live demo of the dashboard and a one-page map of the window. Finals are offline in Hangzhou on 22 September.

## Environment contract

**Fixed:** first-longitudinal chamber frequency, acoustic damping, Crocco n-τ reaction delay model, stability threshold `S_crit = 1`, allowable domain.

**Explorable vector** `x = [g, d, a, s, o]`

| Symbol | Meaning (nondimensional analog) | Bounds |
| --- | --- | --- |
| `g` | Injector pattern class (like-on-like → unlike → swirl-coaxial) | [0, 2] |
| `d` | Orifice-size spread | [0, 1] |
| `a` | Impingement / injection interaction | [0, 1] |
| `s` | Swirl intensity | [0, 1] |
| `o` | Operating-condition analog | [0, 1] |

**Feedback:** pressure amplitude `Ap`, dominant frequency `f_dom`, Rayleigh coupling `R`, mixture uniformity `Um`, wall thermal analog `Qw`, convergence flag `Cconv`, stability metric `S`.

**Discovery signals (declared before search):**

1. A reconstructed stable / unstable window under a matched budget.
2. A **counterexample** to “increasing swirl monotonically improves stability.”
3. A **disconnected unstable pocket** near like-on-like pattern + high load.
4. Numerical non-convergence, logged and *not* counted as physics.

**Baseline:** Latin-hypercube sampling with the same budget, bounds, model, and threshold. The only difference is experiment selection.

## Install and run

Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run cswe demo
uv run cswe run --budget 48 --seed 7 --out artifacts/demo
uv run pytest
uv run streamlit run app/dashboard.py --server.port 48217 --server.address 0.0.0.0
```

The CLI writes:

- `artifacts/demo/ai/exploration.jsonl` — one complete AI campaign
- `artifacts/demo/baseline/exploration.jsonl` — matched LHS baseline
- `artifacts/demo/comparison.json` — hold-out map accuracy, first boundary hit, discovery counts

## Why this is a scientific environment, not a demo

Uniform grids waste evaluations far from the interesting set `{x : S(x) = S_crit}`. The agent uses a Gaussian-process surrogate and the **straddle** acquisition rule for level-set estimation, so later samples pile up on the uncertain window edge. That is the claim in the preliminary paper: adaptive exploration recovers the window more efficiently than non-adaptive sampling under the same simulation budget.

The physics is intentionally cheap and canonical. It is the smallest model that still produces:

- non-monotonic injector effects (time lag vs coupling strength)
- an interaction between swirl and orifice spread
- a compact secondary-mode island that one-factor sweeps miss
- occasional solver failures the agent must refuse to over-interpret

Extending the same loop to a higher-fidelity combustion solver later does not change the exploration contract.

## License

MIT. All data in this repository are synthetic and nondimensional.
