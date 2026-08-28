"""Streamlit dashboard for the GOAI second-round exploration environment."""

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
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetric"] { background: #141414; border: 1px solid #2a2a2a; padding: 0.6rem 0.8rem; border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


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


def scatter_map(df: pd.DataFrame, title: str) -> go.Figure:
    valid = df[df["Cconv"] == 1].copy() if not df.empty else df
    if valid.empty:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_dark", height=380)
        return fig
    valid["regime"] = np.where(valid["stable"] == 1, "stable", "unstable")
    fig = px.scatter(
        valid,
        x="s",
        y="d",
        color="regime",
        size="S",
        hover_data=["g", "a", "o", "S", "t"],
        color_discrete_map={"stable": "#3dd68c", "unstable": "#ff5d5d"},
        title=title,
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="swirl analog s",
        yaxis_title="orifice-spread analog d",
        legend_title="",
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig


def progress_curve(df: pd.DataFrame, name: str) -> go.Figure:
    valid = df[df["Cconv"] == 1].copy()
    if valid.empty:
        return go.Figure()
    valid = valid.sort_values("t")
    near = (np.abs(valid["S"] - STABILITY_THRESHOLD) < 0.35).astype(int)
    valid["boundary_hits"] = near.cumsum()
    fig = px.line(valid, x="t", y="boundary_hits", title=f"{name}: cumulative near-boundary evaluations")
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=48, b=10))
    return fig


st.title("AI-guided combustion-stability windows")
st.caption(
    "GOAI Track 3 · Type II open exploration · second-round environment. "
    "The chamber, reaction model, acoustics, and stability criterion are frozen. "
    "The agent only chooses the next injector / operating analog to evaluate."
)

tabs = st.tabs(
    ["Campaign results", "Run a live campaign", "Probe one condition", "True map (held-out)", "What judges should see"]
)

ai_df, base_df, comparison = load_demo()

with tabs[0]:
    if ai_df.empty:
        st.warning("No demo logs yet. Run `cswe run --out artifacts/demo` first.")
    else:
        ai_valid = ai_df[ai_df["Cconv"] == 1]
        base_valid = base_df[base_df["Cconv"] == 1]
        c1, c2, c3, c4 = st.columns(4)
        ai_acc = comparison.get("ai", {}).get("reconstruction", {}).get("holdout_accuracy")
        base_acc = comparison.get("baseline", {}).get("reconstruction", {}).get("holdout_accuracy")
        c1.metric("AI hold-out map accuracy", f"{ai_acc:.2f}" if ai_acc else "—")
        c2.metric("LHS baseline accuracy", f"{base_acc:.2f}" if base_acc else "—")
        c3.metric("AI first boundary step", str(comparison.get("ai", {}).get("first_boundary_step")))
        c4.metric(
            "AI vs baseline discoveries",
            f"{comparison.get('ai', {}).get('n_discoveries')} / {comparison.get('baseline', {}).get('n_discoveries')}",
        )
        left, right = st.columns(2)
        left.plotly_chart(scatter_map(ai_df, "AI level-set campaign (swirl × orifice spread)"), use_container_width=True)
        right.plotly_chart(scatter_map(base_df, "Latin-hypercube baseline (same budget)"), use_container_width=True)
        st.plotly_chart(progress_curve(ai_df, "AI"), use_container_width=True)

        st.subheader("Discovery signals (pre-registered)")
        meta_path = DEMO_DIR / "ai" / "campaign.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Hypotheses**")
                for h in meta.get("hypotheses", []):
                    st.write(f"- `{h['id']}` · **{h['status']}** — {h['note']}")
            with cols[1]:
                st.markdown("**Discoveries**")
                for d in meta.get("discoveries", []):
                    st.write(f"- **{d['type']}** — {d['note']}")

        with st.expander("Raw exploration log"):
            st.dataframe(ai_df, use_container_width=True, hide_index=True)

with tabs[1]:
    st.write(
        "Live campaigns use a tiny budget so the dashboard stays interactive. "
        "The committed demo logs were generated with the official CLI budget of 48."
    )
    budget = st.slider("Matched budget", min_value=12, max_value=36, value=20, step=2)
    seed = st.number_input("Seed", min_value=0, value=11)
    if st.button("Run matched AI vs LHS campaign", type="primary"):
        with st.spinner("Exploring the stability window…"):
            env_ai = ExplorationEnv(seed=int(seed))
            ai_rec = LevelSetAgent(env_ai, n_init=6, n_candidates=250).run(budget=int(budget))
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
        m1.metric("AI map accuracy", f"{cmp['ai']['reconstruction']['holdout_accuracy']:.2f}")
        m2.metric("LHS map accuracy", f"{cmp['baseline']['reconstruction']['holdout_accuracy']:.2f}")
        m3.metric("AI discoveries", str(cmp["ai"]["n_discoveries"]))
        a, b = st.columns(2)
        a.plotly_chart(scatter_map(st.session_state["live_ai"], "Live AI"), use_container_width=True)
        b.plotly_chart(scatter_map(st.session_state["live_base"], "Live LHS"), use_container_width=True)
        st.json({"hypotheses": st.session_state["live_hyp"], "discoveries": st.session_state["live_disc"]})

with tabs[2]:
    st.write("Evaluate one nondimensional injector / operating analog. This is the same frozen environment the agent queries.")
    cols = st.columns(5)
    g = cols[0].slider("pattern class g", 0.0, 2.0, 1.0, 0.05)
    d = cols[1].slider("orifice spread d", 0.0, 1.0, 0.25, 0.01)
    a = cols[2].slider("impingement a", 0.0, 1.0, 0.45, 0.01)
    s = cols[3].slider("swirl s", 0.0, 1.0, 0.55, 0.01)
    o = cols[4].slider("operating analog o", 0.0, 1.0, 0.40, 0.01)
    result = simulate({"g": g, "d": d, "a": a, "s": s, "o": o}, rng=np.random.default_rng(0))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stability metric S", "—" if not result.Cconv else f"{result.S:.3f}")
    k2.metric("Regime", "failed" if not result.Cconv else ("stable" if result.stable else "unstable"))
    k3.metric("Rayleigh coupling R", "—" if not result.Cconv else f"{result.R:.3f}")
    k4.metric("Time lag τ", f"{result.tau:.3f}")
    st.caption(f"Threshold S_crit = {STABILITY_THRESHOLD}. S < 1 is stable. Bounds: {PARAM_BOUNDS}")
    if not result.Cconv:
        st.error("Numerical non-convergence. The agent logs this and does not treat it as a discovery.")

with tabs[3]:
    st.write(
        "Noise-free slice of the hidden map for reviewers. The agent never sees this surface; "
        "it only receives individual simulation returns."
    )
    slice_g = st.slider("Fixed pattern class g", 0.0, 2.0, 0.4, 0.1)
    slice_a = st.slider("Fixed impingement a", 0.0, 1.0, 0.32, 0.02)
    slice_o = st.slider("Fixed operating analog o", 0.0, 1.0, 0.72, 0.02)
    n = 40
    ss = np.linspace(0, 1, n)
    dd = np.linspace(0, 1, n)
    Z = np.zeros((n, n))
    for i, dv in enumerate(dd):
        for j, sv in enumerate(ss):
            Z[i, j] = 0.0 if true_stability({"g": slice_g, "d": dv, "a": slice_a, "s": sv, "o": slice_o}) else 1.0
    fig = px.imshow(
        Z,
        origin="lower",
        x=ss,
        y=dd,
        color_continuous_scale=["#3dd68c", "#ff5d5d"],
        labels={"x": "swirl s", "y": "orifice spread d", "color": "unstable"},
        title="True stable (green) / unstable (red) slice",
        aspect="auto",
    )
    fig.update_layout(template="plotly_dark", height=480, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The compact red pocket at low swirl / moderate spread is the disconnected unstable island.")

with tabs[4]:
    st.markdown(
        """
**Second-round package (Type II)**

1. **Runnable environment** — this app plus `cswe run`.
2. **Exploration logs** — `artifacts/demo/ai/exploration.jsonl` is one complete campaign.
3. **Baseline** — budget-matched Latin hypercube in `artifacts/demo/baseline/`.
4. **Reproduction** — `uv sync && cswe run --budget 48 --seed 7 --out artifacts/demo`.

**What is in scope.** Discover how injector analogs move the stable/unstable boundary of a frozen low-order n-τ chamber. Not thrust, not Isp, not a flight injector.

**What counts as a discovery.** A reconstructed window, a window shift, a counterexample to “more swirl always helps,” a disconnected unstable pocket, or a stable negative result. Failed CFD-analog runs are logged and discarded.

**Safety.** Outputs stay at abstract design principles. There are no dimensional orifice diameters, chamber drawings, or propellant flow rates.
        """
    )
