"""Expansion isolation tests. No live OpenFOAM required."""

from pathlib import Path

from cswe.live_expansion import (
    EXPANSION_SEEDS,
    GSWEEP_VERIFY_PATH,
    ORIGINAL_SEEDS,
    ROOT,
    expansion_dir,
)
from cswe.sample_efficiency import analyze_live_sample_efficiency


def test_expansion_seeds_disjoint_from_frozen_and_final_validation():
    assert set(ORIGINAL_SEEDS).isdisjoint(EXPANSION_SEEDS)
    assert 101 not in ORIGINAL_SEEDS
    assert 101 not in EXPANSION_SEEDS
    assert ORIGINAL_SEEDS == [8, 11, 14, 19, 23, 26, 32, 35]
    assert len(EXPANSION_SEEDS) == 16
    assert len(set(EXPANSION_SEEDS)) == 16


def test_expansion_paths_do_not_match_frozen_glob():
    frozen = sorted((ROOT / "artifacts").glob("cfd_study_s*/cfd_comparison.json"))
    assert len(frozen) == 8
    assert all("cfd_expansion" not in p.as_posix() for p in frozen)
    assert expansion_dir(38).name == "cfd_expansion_s38"
    assert not expansion_dir(38).match("cfd_study_s*")


def test_g_sweep_verify_does_not_target_historical_file():
    assert GSWEEP_VERIFY_PATH == ROOT / "artifacts" / "g_sweep_mesh_verification.json"
    assert GSWEEP_VERIFY_PATH.name != "g_sweep.json"


def test_sample_efficiency_still_reads_only_frozen_eight():
    before = {}
    for path in sorted((ROOT / "artifacts").glob("cfd_study_s*/cfd_comparison.json")):
        before[str(path)] = path.read_text(encoding="utf-8")
    payload = analyze_live_sample_efficiency()
    assert payload["n_seeds"] == 8
    assert payload["seeds"] == ORIGINAL_SEEDS
    for path, text in before.items():
        assert Path(path).read_text(encoding="utf-8") == text


def test_cli_exposes_expansion_commands():
    text = (ROOT / "src" / "cswe" / "cli.py").read_text(encoding="utf-8")
    assert "live-expansion" in text
    assert "g-sweep-verify" in text


def test_committed_expansion_complete_and_isolated():
    import json

    exp_path = ROOT / "artifacts" / "cfd_live_expansion.json"
    if not exp_path.exists():
        return
    payload = json.loads(exp_path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        return
    assert payload["openfoam_executed"] is True
    assert payload["summary"]["n_seeds"] == 16
    assert payload["summary"]["seeds"] == EXPANSION_SEEDS
    frozen = sorted((ROOT / "artifacts").glob("cfd_study_s*/cfd_comparison.json"))
    assert len(frozen) == 8
    gv_path = ROOT / "artifacts" / "g_sweep_mesh_verification.json"
    if gv_path.exists():
        gv = json.loads(gv_path.read_text(encoding="utf-8"))
        if gv.get("status") == "COMPLETE":
            assert gv["original_g_sweep_not_rewritten"] is True
            assert gv["openfoam_executed"] is True
            assert (ROOT / "artifacts" / "g_sweep.json").exists()
            assert gv_path.name != "g_sweep.json"
