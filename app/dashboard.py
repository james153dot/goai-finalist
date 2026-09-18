"""Streamlit dashboard for the GOAI 2026 Open Exploration finalist package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cswe.agent import LevelSetAgent
from cswe.baseline import LatinHypercubeBaseline
from cswe.environment import ExplorationEnv
from cswe.geometry import CLASSICAL_INJECTORS, L, DESIGN_VARIABLES
from cswe.logging_utils import compare_campaigns, load_evaluations
from cswe.physics import PARAM_BOUNDS, STABILITY_THRESHOLD, simulate, true_stability

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "artifacts" / "demo"

st.set_page_config(
    page_title="Combustion-stability windows",
    page_icon="🔥",
    layout="wide",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1.1rem; max-width: 1400px; }
    div[data-testid="stMetric"] { background: #141414; border: 1px solid #2a2a2a; padding: 0.6rem 0.8rem; border-radius: 12px; }
    .demo-q { font-size: 1.55rem; line-height: 1.35; font-weight: 600; margin: 0.2rem 0 1.0rem 0; }
    .demo-h { font-size: 1.25rem; font-weight: 650; margin: 1.15rem 0 0.35rem 0; }
    .demo-line { font-size: 1.15rem; line-height: 1.45; margin: 0.15rem 0; }
    .demo-take { font-size: 1.35rem; line-height: 1.4; font-weight: 650; margin: 0.8rem 0 0.2rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_demo() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ai_path = DEMO_DIR / "ai" / "exploration.jsonl"
    base_path = DEMO_DIR / "baseline" / "exploration.jsonl"
    cmp_path = DEMO_DIR / "comparison.json"
    if not ai_path.exists():
        return pd.DataFrame(), pd.DataFrame(), {}
    ai = pd.DataFrame(load_evaluations(ai_path))
    base = pd.DataFrame(load_evaluations(base_path))
    comparison = json.loads(cmp_path.read_text()) if cmp_path.exists() else {}
    return ai, base, comparison


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)
    if "Cvalid" in df.columns:
        v = df["Cvalid"].astype(bool)
        if "Cconv" in df.columns:
            v = v | df["Cconv"].astype(bool)
        return v
    if "Cconv" in df.columns:
        return df["Cconv"].astype(bool)
    return pd.Series([False] * len(df), index=df.index)


def scatter_map(df: pd.DataFrame, title: str) -> go.Figure:
    valid = df[_valid_mask(df)].copy() if not df.empty else df
    if valid.empty:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_dark", height=380)
        return fig
    valid["regime"] = np.where(valid["stable"] == 1, "stable", "unstable")
    fig = px.scatter(
        valid,
        x="g",
        y="s",
        color="regime",
        size=np.clip(np.abs(valid["sigma"]) + 0.15, 0.15, 2.0) if "sigma" in valid else valid["S"],
        hover_data=["d", "a", "o", "S", "sigma", "tau", "t"],
        color_discrete_map={"stable": "#3dd68c", "unstable": "#ff5d5d"},
        title=title,
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="pattern class g",
        yaxis_title="swirl analog s",
        legend_title="",
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig


def progress_curve(ai: pd.DataFrame, base: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for df, name, color in ((ai, "AI", "#6ea8fe"), (base, "LHS", "#adb5bd")):
        valid = df[_valid_mask(df)].copy()
        if valid.empty:
            continue
        valid = valid.sort_values("t")
        found = (valid["sigma"] > 0).astype(int) if "sigma" in valid else (valid["stable"] == 0).astype(int)
        valid["n_unstable"] = found.cumsum()
        fig.add_trace(go.Scatter(x=valid["t"], y=valid["n_unstable"], name=name, line=dict(color=color, width=2)))
    fig.update_layout(
        template="plotly_dark",
        height=320,
        title="Cumulative unstable evaluations (the dangerous class)",
        xaxis_title="CFD / atlas evaluation t",
        yaxis_title="unstable count",
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig


st.title("AI-guided combustion-stability windows")
st.caption(
    "GOAI Track 3 · Type II · 2026 finalist package. "
    "OpenFOAM 14 dual-jet mixing + declared closed-closed 1L Rayleigh analog. "
    "This is not a rocket engine. The scoring object is the environment and whether adaptive search "
    "spends expensive solver calls on the unstable window."
)

tabs = st.tabs(
    [
        "Final Demo — 90 seconds",
        "Where AI is useful",
        "CFD physics",
        "Campaign logs",
        "Run a live campaign",
        "Probe one condition",
        "Atlas slice",
        "For a scientist judge",
    ]
)

ai_df, base_df, comparison = load_demo()
seed_study = _load_json(ROOT / "artifacts" / "seed_study.json")
seeds_summary = _load_json(ROOT / "artifacts" / "seeds_summary.json")
swirl = _load_json(ROOT / "artifacts" / "swirl_sweep.json")
gsweep = _load_json(ROOT / "artifacts" / "g_sweep.json")

cfd_studies = []
for p in sorted((ROOT / "artifacts").glob("cfd_study_s*/cfd_comparison.json")):
    payload = _load_json(p)
    if payload:
        cfd_studies.append(payload)
cfd_studies.sort(key=lambda d: d.get("seed", 0))

with tabs[0]:
    st.markdown('<p class="demo-h">A. Scientific question</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="demo-q">Given only 16 expensive numerical experiments, can an AI scientist '
        "choose which simulations to run so that rare instability regimes are discovered "
        "faster than blind space-filling?</p>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="demo-h">B. Pipeline</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="demo-line">design variables <b>z = [g, d, a, s, o]</b> '
        "→ OpenFOAM mixing simulation → extracted mixing quantities "
        r"($\tau$, $R_{\mathrm{spatial}}$, $q_{\mathrm{mix}}$) "
        r"→ declared stability analog $\sigma_{\mathrm{analog}}$ "
        "→ AI selects next experiment</p>",
        unsafe_allow_html=True,
    )
    st.caption("Offline demo. Committed artifacts only. No OpenFOAM. No internet.")

    st.markdown('<p class="demo-h">C. Main live result · eight OpenFOAM campaigns · budget 16</p>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AI unstable recall", "0.68")
    m2.metric("LHS unstable recall", "0.33")
    m3.metric("AI F1", "0.76")
    m4.metric("LHS F1", "0.46")
    m5.metric("AI higher recall", "7/8")
    st.markdown(
        '<p class="demo-line">Mean precision is a near-tie (0.86 vs 0.85). '
        "Seed 35 is the reversal and is kept. No p-value on n = 8.</p>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="demo-h">D. Sample efficiency</p>', unsafe_allow_html=True)
    se_fig = ROOT / "artifacts" / "figures" / "sample_efficiency_live.png"
    if se_fig.exists():
        st.image(str(se_fig), use_container_width=True)
        st.caption("Mean / median hold-out unstable recall vs live CFD evaluations. Shaded band: sample sd. n = 8. No extrapolation past budget 16.")
    else:
        st.info("Generate with `cswe sample-efficiency` from the committed live campaigns.")

    st.markdown('<p class="demo-h">E. Discovery story · seed 14</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="demo-line">AI independently encountered unstable samples at low and high <b>g</b> '
        "(t=2, g=0.10, σ=+0.32 and t=3, g=1.80, σ=+0.06). That motivated the controlled g-sweep.</p>",
        unsafe_allow_html=True,
    )
    st.caption("Do not read this as the agent mapping two full 5-D disconnected regions.")
    seed14_fig = ROOT / "artifacts" / "figures" / "seed14_both_g_intervals.png"
    if seed14_fig.exists():
        st.image(str(seed14_fig), use_container_width=True)

    st.markdown('<p class="demo-h">F. Scientific skepticism</p>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown('<p class="demo-line"><b>Seed 35 reversal.</b> LHS recall 0.56 vs AI 0.44.</p>', unsafe_allow_html=True)
        st.caption("The adaptive policy can lose. The seed is retained.")
    with s2:
        st.markdown(
            '<p class="demo-line"><b>High-g interval disappears</b> at ω+10% and at V_crit = 0.035.</p>',
            unsafe_allow_html=True,
        )
        st.caption("Topology is conditional on analog / τ assumptions.")
    s3, s4 = st.columns(2)
    with s3:
        st.markdown(
            '<p class="demo-line"><b>Near-threshold V&amp;V.</b> Unlike-impinging flips label at 40 iterations.</p>',
            unsafe_allow_html=True,
        )
        st.caption("Live setting (default mesh, 90 iters) stays analog-stable.")
    with s4:
        st.markdown(
            '<p class="demo-line"><b>Frozen hold-out (seed 101).</b> LHS recall 0.76 vs AI 0.59; AI found more unstables (7 vs 4).</p>',
            unsafe_allow_html=True,
        )
        st.caption("One-shot post-development check. Not used to retune.")
    st.caption("Negative and conditional results are retained.")

    st.markdown('<p class="demo-h">G. Post-development expansion · not used to retune</p>', unsafe_allow_html=True)
    exp = _load_json(ROOT / "artifacts" / "cfd_live_expansion.json")
    pooled = _load_json(ROOT / "artifacts" / "cfd_live_pooled.json")
    gsv = _load_json(ROOT / "artifacts" / "g_sweep_mesh_verification.json")
    if exp and exp.get("status") == "COMPLETE":
        s = exp.get("summary") or {}
        rec = s.get("recall") or {}
        paired = s.get("paired_ai_vs_lhs") or {}
        e1, e2, e3, e4 = st.columns(4)
        ai_r = (rec.get("ai") or {}).get("mean")
        lhs_r = (rec.get("lhs") or {}).get("mean")
        rnd_r = (rec.get("random") or {}).get("mean")
        e1.metric("Expansion AI recall", "—" if ai_r is None else f"{ai_r:.2f}")
        e2.metric("Expansion LHS recall", "—" if lhs_r is None else f"{lhs_r:.2f}")
        e3.metric("Expansion Random recall", "—" if rnd_r is None else f"{rnd_r:.2f}")
        e4.metric("AI higher recall", paired.get("ai_higher_unstable_recall", "—"))
        st.caption(
            f"n = {s.get('n_seeds', 0)} extra live seeds under `artifacts/cfd_expansion_s*`. "
            "The frozen n = 8 study is unchanged. Constants and acquisition were not retuned."
        )
        rows = s.get("per_seed") or []
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        exp_fig = ROOT / "artifacts" / "figures" / "live_cfd_expansion_recall.png"
        if exp_fig.exists():
            st.image(str(exp_fig), use_container_width=True)
        if pooled:
            p = pooled.get("pooled") or {}
            pr = (p.get("recall") or {})
            st.caption(
                "Pooled original+expansion is additional evidence, not a replacement for n = 8. "
                f"Pooled AI recall {(pr.get('ai') or {}).get('mean')} vs LHS {(pr.get('lhs') or {}).get('mean')} "
                f"({(p.get('paired_ai_vs_lhs') or {}).get('ai_higher_unstable_recall')})."
            )
    elif exp and exp.get("status") == "PARTIAL":
        st.info(
            f"Expansion is PARTIAL ({len((exp.get('summary') or {}).get('seeds') or [])} of "
            f"{len(exp.get('planned_seeds') or [])} seeds). Frozen n = 8 is still the primary live study."
        )
    else:
        st.caption(
            "Optional extra live seeds are written to `artifacts/cfd_expansion_s*` by `cswe live-expansion`. "
            "The primary live study remains the frozen n = 8."
        )
    if gsv and gsv.get("status") == "COMPLETE":
        st.caption(
            "g-slice mesh check: two analog-unstable intervals on every tested mesh = "
            f"{gsv.get('two_intervals_on_every_tested_mesh')}. `g_sweep.json` was not rewritten."
        )
        gsv_fig = ROOT / "artifacts" / "figures" / "g_sweep_mesh_verification.png"
        if gsv_fig.exists():
            st.image(str(gsv_fig), use_container_width=True)
    elif gsv and gsv.get("status") == "PENDING":
        st.caption("g-slice mesh verification is PENDING (OpenFOAM was not available when requested).")

    st.markdown(
        '<p class="demo-take">H. Final takeaway — AI does not replace the simulator. '
        "It decides which expensive scientific experiment should be performed next.</p>",
        unsafe_allow_html=True,
    )

with tabs[1]:
    st.markdown(
        """
The useful job is recovering the minority unstable class with a small solver budget.
Unstable injectors are ~34% of the OpenFOAM atlas. Overall accuracy can remain high
by predicting the dominant stable regime. Unstable recall
Recall_U = TP_U / (TP_U + FN_U) measures whether the reconstructed map recovers
the scientifically important minority class.

The agent fits a Gaussian process to **σ_analog** (level set σ_analog = 0) and:

1. hunts for the missing class if every observation so far is stable,
2. then straddles the analog threshold.
        """
    )
    if seed_study:
        s = seed_study["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unstable recall (24 seeds × budget 16)", f"{s['ai_mean_unstable_recall']:.2f}")
        c1.caption(f"LHS {s['lhs_mean_unstable_recall']:.2f}")
        c2.metric("Unstable evals per campaign", f"{s['ai_mean_n_unstable']:.1f}")
        c2.caption(f"LHS {s['lhs_mean_n_unstable']:.1f}")
        c3.metric("Near-boundary growth-rate MAE", f"{s['ai_mean_boundary_mae']:.3f}")
        c3.caption(f"LHS {s['lhs_mean_boundary_mae']:.3f}  ·  E_σ,boundary, not contour distance")
        c4.metric("Volume accuracy", f"{s['ai_mean_volume_accuracy']:.2f}")
        c4.caption(f"LHS {s['lhs_mean_volume_accuracy']:.2f}")
        st.caption(
            "Atlas campaigns scored on an independent 24-case OpenFOAM hold-out (`artifacts/of_test.json`). "
            "Volume accuracy is the weak metric — space-filling already tiles the majority class. "
            "Recall of the dangerous class is the claim. Atlas precision is "
            "0.88 vs 0.91; F1 follows recall. Near-boundary growth-rate MAE "
            "E_σ,boundary is prediction error in σ_analog among hold-out points with |σ| < 0.20, "
            "not Hausdorff distance to a contour."
        )
        recs = pd.DataFrame(
            [
                {
                    "seed": r["seed"],
                    "AI n_unstable": r["ai"]["n_unstable_found"],
                    "LHS n_unstable": r["baseline"]["n_unstable_found"],
                    "AI first unstable": r["ai"]["time_to_first_unstable"],
                    "LHS first unstable": r["baseline"]["time_to_first_unstable"],
                    "AI recall": (r.get("ai_test") or {}).get("unstable_recall"),
                    "LHS recall": (r.get("baseline_test") or {}).get("unstable_recall"),
                }
                for r in seed_study["records"]
            ]
        )
        fig = px.scatter(
            recs,
            x="LHS n_unstable",
            y="AI n_unstable",
            hover_data=["seed", "AI recall", "LHS recall"],
            title="Each point is one matched budget-16 campaign",
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=12, y1=12, line=dict(color="#888", dash="dash"))
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=48, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Points above the diagonal: the agent spent more of the same budget on unstable conditions.")
    else:
        st.info("Run `cswe seed-study` to populate the 24-seed comparison.")

    st.subheader("Confirmation: each evaluation is a live OpenFOAM run")
    live_summary = _load_json(ROOT / "artifacts" / "cfd_live_summary.json")
    seed14_fig = ROOT / "artifacts" / "figures" / "seed14_both_g_intervals.png"
    if seed14_fig.exists():
        st.markdown(
            "**Primary exhibit — seed 14.** The adaptive campaign independently "
            "encountered unstable conditions at both low and high pattern-class *g* "
            "(full 5-D samples: g = 0.10, σ = +0.32 and g = 1.80, σ = +0.06; later "
            "g = 1.92, σ = +0.40). That motivated the subsequent fixed-condition "
            "g-sweep, which established two separated unstable intervals on a "
            "low-swirl 1-D slice. Seed 14 did not literally sample that slice."
        )
        st.image(str(seed14_fig), use_container_width=True)
    if live_summary:
        paired = live_summary.get("paired", {})
        st.markdown(
            f"Across eight matched live-CFD campaigns, adaptive exploration achieved higher "
            f"unstable recall than LHS in **{paired.get('ai_higher_unstable_recall', '7/8')}** seeds "
            f"and sampled more unstable conditions in **{paired.get('ai_more_unstable_evaluations', '7/8')}**. "
            f"Unstable F1 follows the same **{paired.get('ai_higher_unstable_f1', '7/8')}**. "
            "The one reversal (seed 35) is retained. Mean unstable precision is a near-tie "
            "(AI higher in only 1/8): adaptive search roughly doubles recall without a precision collapse. "
            "Near-boundary growth-rate MAE is a mean tie. Adaptive sampling is for rare-regime recovery, not every metric."
        )
    if cfd_studies:
        rows = []
        for p in cfd_studies:
            ai = p.get("ai_diagnostics") or (p.get("ai") or {}).get("final") or {}
            base = p.get("baseline_diagnostics") or (p.get("baseline") or {}).get("final") or {}
            ai_f = (p.get("ai") or {}).get("final") or {}
            base_f = (p.get("baseline") or {}).get("final") or {}
            rows.append(
                {
                    "seed": p["seed"],
                    "AI unstable found": ai.get("n_unstable_found"),
                    "LHS unstable found": base.get("n_unstable_found"),
                    "AI recall": ai_f.get("unstable_recall"),
                    "LHS recall": base_f.get("unstable_recall"),
                    "AI precision": ai_f.get("unstable_precision"),
                    "LHS precision": base_f.get("unstable_precision"),
                    "AI F1": ai_f.get("unstable_f1"),
                    "LHS F1": base_f.get("unstable_f1"),
                    "AI volume acc.": ai_f.get("volume_accuracy"),
                    "LHS volume acc.": base_f.get("volume_accuracy"),
                    "AI E_σ,boundary": ai_f.get("near_boundary_sigma_mae") or ai_f.get("boundary_mae"),
                    "LHS E_σ,boundary": base_f.get("near_boundary_sigma_mae") or base_f.get("boundary_mae"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Live `foamRun` campaigns, 16 evaluations per method, scored on the same OpenFOAM hold-out. "
            "E_σ,boundary is near-boundary growth-rate MAE (|σ| < 0.20), not geometric contour distance. "
            "Seed 35 reverses recall and is kept."
        )
        fig = go.Figure()
        palette = {11: "#6ea8fe", 14: "#3dd68c", 19: "#c4b5fd"}
        for p in cfd_studies:
            seed = p["seed"]
            color = palette.get(seed, "#adb5bd")
            ai_c = (p.get("ai") or {}).get("curve") or []
            base_c = (p.get("baseline") or {}).get("curve") or []
            if ai_c:
                fig.add_trace(
                    go.Scatter(
                        x=[r["budget"] for r in ai_c],
                        y=[r.get("unstable_recall") for r in ai_c],
                        name=f"AI seed {seed}",
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                    )
                )
            if base_c:
                fig.add_trace(
                    go.Scatter(
                        x=[r["budget"] for r in base_c],
                        y=[r.get("unstable_recall") for r in base_c],
                        name=f"LHS seed {seed}",
                        mode="lines+markers",
                        line=dict(color=color, width=1, dash="dash"),
                    )
                )
        fig.update_layout(
            template="plotly_dark",
            height=380,
            title="Hold-out unstable recall vs live OpenFOAM budget",
            xaxis_title="foamRun evaluations",
            yaxis_title="unstable recall on of_test.json",
            yaxis=dict(range=[-0.05, 1.05]),
            margin=dict(l=10, r=10, t=48, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Dashed = Latin hypercube. Seed 14 LHS recall stays at 0: the space-filling GP never reconstructed the dangerous class.")
    else:
        st.info("Live OpenFOAM campaigns are still running (`cswe cfd-study`).")

with tabs[2]:
    st.markdown(
        """
**Fixed-geometry chamber.** Length 80 mm, height 20 mm, laminar `incompressibleFluid`, complementary
mixture fraction Z = 0 / 1 (OpenFOAM field name `T`). **Explorable injector vector** `z = [g, d, a, s, o]`.

**Mixing-availability proxy** q_mix(x) = Var_y[Z](x) is unmixedness, not a heat-release prediction.
It is the declared mixing-side weighting of the Rayleigh analog — not 4Z(1−Z).

**Acoustics.** Declared closed-closed 1L, `p(x) = cos(πx/L)`, injector face a pressure antinode.
n = 0.50 + 0.28 tanh(C−1) + 0.16 (1−Um) + 0.10 (o−0.5)², then
σ_analog = α n R_spatial cos(ωτ) − D with α = 1.45, D = 0.08, ω = ω0 (0.92 + 0.16 o).
Unstable iff σ > 0. No planted island.
Mixing delay τ is the first axial bin with Var_y[Z] < 0.045, then τ = x_m / U_b.
        """
    )
    st.dataframe(pd.DataFrame(DESIGN_VARIABLES), use_container_width=True, hide_index=True)
    st.caption("What changing each coordinate of z does in the OpenFOAM dual-jet case (`jet_layout`).")
    cols = st.columns(3)
    rng = np.random.default_rng(0)
    for col, (name, x) in zip(cols, CLASSICAL_INJECTORS.items()):
        r = simulate(x, rng=rng, backend="atlas")
        col.metric(name.replace("_", " "), "unstable" if not r.stable else "stable")
        col.caption(f"σ={r.sigma:.2f}  τ={r.tau:.3f}s  Rₓ={r.R_spatial:.2f}")
    st.caption("Like-on-like is the dangerous classical analog. Unlike-impinging sits near the window. Swirl-coaxial is deep stable.")

    if gsweep:
        gdf = pd.DataFrame(gsweep["rows"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gdf["g"], y=gdf["sigma"], name="σ", mode="lines+markers", line=dict(color="#ff5d5d")))
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.add_trace(go.Scatter(x=gdf["g"], y=gdf["tau"], name="τ (s)", yaxis="y2", mode="lines+markers", line=dict(color="#6ea8fe")))
        fig.update_layout(
            template="plotly_dark",
            height=380,
            title=f"Pattern-class sweep at s={gsweep.get('s', 0.1):.2f} (live OpenFOAM)",
            xaxis_title="pattern class g  (0 ≈ like-on-like family, 2 ≈ swirl-coaxial family)",
            yaxis_title="growth rate σ",
            yaxis2=dict(title="mixing delay τ (s)", overlaying="y", side="right"),
            margin=dict(l=10, r=10, t=48, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "On this slice the like-on-like end is unstable. The map is not monotone: "
            "a second unstable *interval* appears at high g with low swirl. That is a "
            "1-D slice observation, not a mapped 5-D pocket. Swirl-coaxial stability "
            "on this analog needs both the coaxial pattern and the swirl analog."
        )
        tau_fig = ROOT / "artifacts" / "figures" / "tau_sensitivity.png"
        if tau_fig.exists():
            st.image(str(tau_fig), use_container_width=True)
            st.caption(
                "τ-definition sensitivity from stored q_mix profiles (no CFD rerun). "
                "Classical ordering survives. V_crit = 0.035 collapses the high-g interval; "
                "N_bins ∈ {16, 32} does not."
            )

    if swirl:
        sdf = pd.DataFrame(swirl["rows"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sdf["s"], y=sdf["sigma"], name="σ", mode="lines+markers", line=dict(color="#ff5d5d")))
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.update_layout(
            template="plotly_dark",
            height=340,
            title=f"Swirl analog sweep inside the like-on-like family (g={swirl.get('g', 0.15):.2f})",
            xaxis_title="swirl analog s",
            yaxis_title="growth rate σ",
            margin=dict(l=10, r=10, t=48, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Increasing the swirl analog inside the like-on-like family lowers σ_analog "
            "but does not cross the stability boundary over the tested range."
        )

    if swirl and swirl["rows"][0].get("q_profile"):
        row0 = swirl["rows"][0]
        row1 = swirl["rows"][-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=row0["x_profile"], y=row0["q_profile"], name=f"s={row0['s']:.2f}  q(x) variance", line=dict(color="#ff5d5d")))
        fig.add_trace(go.Scatter(x=row1["x_profile"], y=row1["q_profile"], name=f"s={row1['s']:.2f}  q(x) variance", line=dict(color="#3dd68c")))
        if row0.get("p_profile"):
            pmax = max(abs(v) for v in row0["p_profile"]) or 1.0
            qmax = max(row0["q_profile"]) or 1.0
            fig.add_trace(
                go.Scatter(
                    x=row0["x_profile"],
                    y=[v / pmax * qmax for v in row0["p_profile"]],
                    name="p(x) closed-closed 1L (scaled)",
                    line=dict(color="#adb5bd", dash="dot"),
                )
            )
        fig.update_layout(
            template="plotly_dark",
            height=360,
            title="Mixing-availability q_mix(x) vs closed-closed 1L pressure",
            xaxis_title="axial position x (m)",
            yaxis_title="q_mix(x)  (cross-stream Var_y[Z])",
            margin=dict(l=10, r=10, t=48, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    if ai_df.empty:
        st.warning("No demo logs yet. Run `cswe run --out artifacts/demo`.")
    else:
        ai_d = comparison.get("ai", {}).get("diagnostics", {})
        base_d = comparison.get("baseline", {}).get("diagnostics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AI unstable found (budget 48)", str(ai_d.get("n_unstable_found", "—")))
        c2.metric("LHS unstable found", str(base_d.get("n_unstable_found", "—")))
        c3.metric("AI first unstable step", str(ai_d.get("time_to_first_unstable", "—")))
        c4.metric("LHS first unstable step", str(base_d.get("time_to_first_unstable", "—")))
        left, right = st.columns(2)
        left.plotly_chart(scatter_map(ai_df, "AI level-set campaign (pattern class × swirl)"), use_container_width=True)
        right.plotly_chart(scatter_map(base_df, "Latin-hypercube baseline (same budget)"), use_container_width=True)
        st.plotly_chart(progress_curve(ai_df, base_df), use_container_width=True)

        meta_path = DEMO_DIR / "ai" / "campaign.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Pre-registered hypotheses**")
                for h in meta.get("hypotheses", []):
                    st.write(f"- `{h['id']}` · **{h['status']}** — {h['note']}")
            with cols[1]:
                st.markdown("**Discoveries**")
                for d in meta.get("discoveries", []):
                    st.write(f"- **{d['type']}** — {d['note']}")

        if seeds_summary:
            st.subheader("Three committed atlas seeds (budget 48), scored on OpenFOAM hold-out")
            st.json(seeds_summary)
            st.caption("All seeds are kept, including any in which the surrogate is no better than Latin hypercube.")

        with st.expander("Raw exploration log"):
            st.dataframe(ai_df, use_container_width=True, hide_index=True)

with tabs[4]:
    st.write(
        "Interactive campaigns query the OpenFOAM mixing **atlas** (instant). "
        "The committed efficiency claim uses live `foamRun` logs in `artifacts/cfd_study_s*`."
    )
    budget = st.slider("Matched budget", min_value=12, max_value=36, value=16, step=2)
    seed = st.number_input("Seed", min_value=0, value=11)
    if st.button("Run matched AI vs LHS campaign", type="primary"):
        with st.spinner("Exploring the stability window…"):
            env_ai = ExplorationEnv(seed=int(seed))
            ai_rec = LevelSetAgent(env_ai, n_init=5, n_candidates=400).run(budget=int(budget))
            env_b = ExplorationEnv(seed=int(seed) + 10_000)
            base_rec = LatinHypercubeBaseline(env_b).run(budget=int(budget))
            cmp = compare_campaigns(ai_rec, base_rec)
        st.session_state["live_ai"] = pd.DataFrame(ai_rec.evaluations)
        st.session_state["live_base"] = pd.DataFrame(base_rec.evaluations)
        st.session_state["live_cmp"] = cmp
        st.session_state["live_disc"] = ai_rec.discoveries
        st.session_state["live_hyp"] = ai_rec.hypotheses

    if "live_ai" in st.session_state:
        cmp = st.session_state["live_cmp"]
        m1, m2, m3 = st.columns(3)
        m1.metric("AI unstable found", str(cmp["ai"]["diagnostics"]["n_unstable_found"]))
        m2.metric("LHS unstable found", str(cmp["baseline"]["diagnostics"]["n_unstable_found"]))
        m3.metric("AI discoveries", str(cmp["ai"]["n_discoveries"]))
        a, b = st.columns(2)
        a.plotly_chart(scatter_map(st.session_state["live_ai"], "Live AI"), use_container_width=True)
        b.plotly_chart(scatter_map(st.session_state["live_base"], "Live LHS"), use_container_width=True)
        st.json({"hypotheses": st.session_state["live_hyp"], "discoveries": st.session_state["live_disc"]})

with tabs[5]:
    st.write("Evaluate one nondimensional injector / operating analog. Same fixed-geometry environment the agent queries.")
    cols = st.columns(5)
    g = cols[0].slider("pattern class g", 0.0, 2.0, 0.15, 0.05)
    d = cols[1].slider("orifice spread d", 0.0, 1.0, 0.08, 0.01)
    a = cols[2].slider("impingement a", 0.0, 1.0, 0.55, 0.01)
    s = cols[3].slider("swirl s", 0.0, 1.0, 0.08, 0.01)
    o = cols[4].slider("operating analog o", 0.0, 1.0, 0.45, 0.01)
    result = simulate({"g": g, "d": d, "a": a, "s": s, "o": o}, rng=np.random.default_rng(0))
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Growth rate σ", "—" if not result.Cvalid else f"{result.sigma:.3f}")
    k2.metric("Regime", "solver-invalid" if not result.Cvalid else ("stable" if result.stable else "unstable"))
    k3.metric("Rayleigh R_spatial", "—" if not result.Cvalid else f"{result.R_spatial:.3f}")
    k4.metric("Time lag τ", f"{result.tau:.3f} s")
    k5.metric("Phase cos(ωτ)", "—" if result.phase != result.phase else f"{result.phase:.2f}")
    st.caption(f"Unstable iff σ > 0 (equivalently S > {STABILITY_THRESHOLD}). Chamber length L = {L} m. Bounds: {PARAM_BOUNDS}")
    if not result.Cvalid:
        st.error("OpenFOAM run was not solver-valid (missing fields or non-finite metrics). The agent logs this and does not treat it as a discovery.")

with tabs[6]:
    st.write(
        "Noise-free slice of the **atlas interpolator**, not LES truth. The agent never sees this surface; "
        "it only receives individual simulation returns."
    )
    slice_d = st.slider("Fixed orifice spread d", 0.0, 1.0, 0.10, 0.02)
    slice_a = st.slider("Fixed impingement a", 0.0, 1.0, 0.55, 0.02)
    slice_o = st.slider("Fixed operating analog o", 0.0, 1.0, 0.50, 0.02)
    n = 36
    gg = np.linspace(0, 2, n)
    ss = np.linspace(0, 1, n)
    Z = np.zeros((n, n))
    for i, sv in enumerate(ss):
        for j, gv in enumerate(gg):
            Z[i, j] = 0.0 if true_stability({"g": float(gv), "d": slice_d, "a": slice_a, "s": float(sv), "o": slice_o}) else 1.0
    fig = px.imshow(
        Z,
        origin="lower",
        x=gg,
        y=ss,
        color_continuous_scale=["#3dd68c", "#ff5d5d"],
        labels={"x": "pattern class g", "y": "swirl s", "color": "unstable"},
        title="Atlas interpolator: stable (green) / unstable (red)",
        aspect="auto",
    )
    fig.update_layout(template="plotly_dark", height=480, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with tabs[7]:
    st.markdown(
        """
**What this package is.** A 2-D laminar dual-jet mixing analog whose delay and
mixing-availability overlap drive a declared closed-closed 1L Rayleigh criterion.
Liquid-rocket injector *names* (like-on-like, unlike-impinging, swirl-coaxial)
are geometry analogs, not flight hardware.

**What a scientist can believe.**
- OpenFOAM supplies mixing fields; σ_analog is a Rayleigh *indicator*, not an engine stability prediction.
- Design vector z = [g, d, a, s, o]; axial coordinate x; q_mix(x) = Var_y[Z](x) is unmixedness, not heat release.
- The passive scalar is denoted Z; it is stored as OpenFOAM field T.
- n = 0.50 + 0.28 tanh(C−1) + 0.16 (1−Um) + 0.10 (o−0.5)². Classical cases share n ≈ 0.79.
- Like-on-like analog: σ_analog > 0. Unlike-impinging: near the threshold. Swirl-coaxial: σ_analog < 0.
- A fixed-low-swirl OpenFOAM g-slice has two separated unstable *intervals*, not a mapped 5-D pocket.
- Live seed 14 independently found unstable 5-D samples at both low and high g; that motivated the later slice.
- Across eight matched live CFD campaigns, AI has higher unstable recall in 7/8 and higher F1 in 7/8. Mean precision is a near-tie. Seed 35 is the reversal and is kept.
- A later 16-seed live expansion (AI vs LHS vs Random) is additional post-development evidence. It was not used to retune. The frozen n=8 study remains the primary live CFD claim.
- The high-g 1-D interval is present at mesh_scale 1.00 and 1.40 and absent at 0.70. `g_sweep.json` was not rewritten.
- Near-boundary growth-rate MAE E_σ,boundary is a mean tie. The claim is rare-regime recovery, not dominance on every metric.
- The high-g unstable interval is not universal within the analog; it disappears at ω + 10% and at V_crit = 0.035.

**What a scientist must not believe.**
- This is not 3-D reacting LES, not a stability margin for a real engine, not a dimensional injector.
- ω, n-index prefactors, α, and damping D are analog constants. They set the scale of σ_analog.
  Classical injectors share n-index ≈ 0.79; ordering is from OpenFOAM τ and R_spatial.
- The atlas interpolator is smoother than a new OpenFOAM case; live foamRun campaigns exist for that reason.
- Volume accuracy can stay high by predicting the majority stable class. Unstable recall is the metric of interest; precision and F1 close the flooding argument.
- E_σ,boundary is not Hausdorff or nearest-contour distance.

**Claim levels.** Observed from OpenFOAM: τ, R_spatial, q_mix(x). Derived from the analog: σ_analog(z), stable/unstable. Inferred by the agent: reconstructed σ_analog=0 contour and predicted unstable set.

**Safety / dual-use.** Public outputs stay at abstract design principles. No dimensional
flight-injector packages.

        """
    )
