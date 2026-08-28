# AI-Guided Discovery of Combustion-Stability Windows

GOAI Track 3 · Type II open exploration · second-round package.

Injector **mixing** is computed with **OpenFOAM 14** (2-D laminar dual-jet mixer). Chamber **acoustics** stay a frozen Crocco n-τ model whose only injector inputs are the CFD mixing delay τ and unmixedness. The agent does not search a planted algebraic map.

## How to place this for judges

Submit this repo plus the dashboard. The scoring object is the environment, not a claim that AI “discovered rocket stability.”

What is actually new relative to the preliminary 4-pager:

- 35 OpenFOAM mixing cases (Latin hypercube + three classical injector analogs)
- τ and Um interpolated from that CFD atlas
- n-τ acoustics with **no hidden island**
- LHS baseline **and** like-on-like / unlike-impinging / swirl-coaxial analogs
- Three random seeds, including a failed AI reconstruction (seed 7)

What the CFD showed, before any agent:

| Injector analog | τ (s) | Regime under frozen n-τ |
| --- | --- | --- |
| like-on-like | 0.089 | **unstable** |
| unlike-impinging | 0.144 | stable |
| swirl-coaxial | 0.228 | stable |

Along a swirl sweep, higher swirl analog **lengthened** mixing delay (recirculation), which moved the Rayleigh phase toward damping. That is the opposite of the first toy model, where swirl was hard-coded to shorten τ.

The efficiency claim is modest and seed-dependent. Seed 7: AI map accuracy 0.76 vs LHS 0.94 (GP collapse). Seeds 11 and 19: AI 0.97 / 0.96 vs LHS 0.96 / 0.94. Report all three.

## Run

OpenFOAM 14 is used to **build** the mixing atlas. Exploration campaigns reuse the atlas and do not launch a solver per step.

```bash
source /opt/openfoam14/etc/bashrc   # or your OpenFOAM 14 install
uv sync
uv run cswe atlas --n 32 --seed 7 --n-iter 100 --workers 4
uv run cswe classical
uv run cswe run --budget 48 --seed 11 --out artifacts/demo
uv run pytest
uv run streamlit run app/dashboard.py --server.port 48217 --server.address 0.0.0.0
```

Live single CFD check (not the interpolator):

```bash
uv run cswe foam --g 0.15 --d 0.08 --a 0.55 --s 0.08 --o 0.45 --out artifacts/foam_like_on_like
```

## Environment contract

**Fixed:** chamber length/height, laminar viscosity, first-longitudinal n-τ criterion, `S_crit = 1`.

**Explorable:** `x = [g, d, a, s, o]` — pattern class, orifice-size spread, impingement, swirl analog, operating analog.

**CFD:** two inlet slots in a 2-D chamber; mixture fraction T=0 / T=1; steady laminar `incompressibleFluid`. Mixing delay is the downstream station where mixture variance drops, divided by bulk speed.

**Discovery signals (declared before search):**

1. A reconstructed stable/unstable window
2. Whether “more swirl always helps” survives the CFD mixing map
3. Whether unstable evaluations form one connected region
4. How classical injector analogs sit on that window

**Out of scope:** thrust, Isp, dimensional flight injectors. Public outputs stay at abstract design principles.

## Competition dates

Second-round package due **3 September**. Finals (if invited) **22 September**, Hangzhou. Type II is ranked separately from the algorithm track.

## License

MIT. Mixing fields are synthetic 2-D CFD, not engine data.
