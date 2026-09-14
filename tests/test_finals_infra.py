"""Infrastructure tests for the GOAI finalist package. No live OpenFOAM."""

from pathlib import Path

from cswe.agent import LevelSetAgent
from cswe.environment import ExplorationEnv
from cswe.final_validation import FROZEN_STATEMENT, frozen_record
from cswe.geometry import CLASSICAL_INJECTORS, jet_layout
from cswe.openfoam import _nxs, _ny, mesh_plan, openfoam_available
from cswe.sample_efficiency import analyze_live_sample_efficiency
from cswe.verification import REPRESENTATIVE_CONDITIONS, planned_configuration, run_verification


def test_default_mesh_scale_matches_committed_resolution():
    layout = jet_layout(1.0, 0.2, 0.5, 0.4, 0.5)
    plan = mesh_plan(layout, 1.0)
    assert plan["nxs"] == 48
    assert _nxs(1.0) == 48
    span = 0.004
    assert _ny(span, 1.0) == max(4, int(round(28 * span / 0.020)))


def test_representative_conditions_are_committed_classicals():
    assert set(REPRESENTATIVE_CONDITIONS) == set(CLASSICAL_INJECTORS)
    for name, spec in REPRESENTATIVE_CONDITIONS.items():
        assert spec["x"] == CLASSICAL_INJECTORS[name]
    cfg = planned_configuration()
    assert cfg["default_nxs"] == 48
    assert 1.0 in cfg["mesh_scales"]


def test_verify_is_pending_without_openfoam(tmp_path):
    if openfoam_available():
        return
    out = tmp_path / "verification.json"
    payload = run_verification(out=out)
    assert payload["status"] == "PENDING"
    assert payload["openfoam_executed"] is False
    assert payload["results"] == []
    assert "git_commit_sha" in payload
    assert "timestamp_utc" in payload
    assert out.exists()


def test_agent_default_policy_is_full_and_reproducible():
    a = LevelSetAgent(ExplorationEnv(seed=3), n_init=4, n_candidates=80)
    b = LevelSetAgent(ExplorationEnv(seed=3), n_init=4, n_candidates=80, policy="full")
    assert a.policy == "full"
    ra = a.run(8)
    rb = b.run(8)
    assert [(r["g"], r["d"], r["s"], r["sigma"]) for r in ra.evaluations] == [
        (r["g"], r["d"], r["s"], r["sigma"]) for r in rb.evaluations
    ]


def test_straddle_policy_disables_epsilon_explore():
    agent = LevelSetAgent(ExplorationEnv(seed=3), n_init=4, n_candidates=80, policy="straddle")
    assert agent.policy == "straddle"
    rec = agent.run(8)
    assert rec.method == "ai_straddle"
    assert len(rec.evaluations) == 8


def test_frozen_validation_statement_is_exact():
    rec = frozen_record()
    assert rec["frozen_statement"] == FROZEN_STATEMENT
    assert FROZEN_STATEMENT == (
        "The algorithm, constants, discovery criteria, and evaluation metrics "
        "were frozen before the final validation set was generated."
    )
    assert rec["final_validation_seed"] == 101
    assert rec["existing_holdout_not_reused"]


def test_sample_efficiency_reads_eight_campaigns_without_rewrite(tmp_path):
    before = {}
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "artifacts").glob("cfd_study_s*/cfd_comparison.json")):
        before[str(path)] = path.read_text(encoding="utf-8")
    payload = analyze_live_sample_efficiency()
    assert payload["n_seeds"] == 8
    assert payload["budget_max"] == 16
    assert payload["historical_campaigns_rewritten"] is False
    assert payload["openfoam_executed"] is False
    auc = payload["area_under_mean_recall_curve"]
    assert auc["ai"] > auc["lhs"]
    for path, text in before.items():
        assert Path(path).read_text(encoding="utf-8") == text


def test_dashboard_has_final_demo_tab():
    text = Path(__file__).resolve().parents[1].joinpath("app/dashboard.py").read_text(encoding="utf-8")
    assert "Final Demo — 90 seconds" in text
