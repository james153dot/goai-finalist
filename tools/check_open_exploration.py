"""Fast repository self-check for the GOAI Open Exploration finalist package."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def warn(warnings: list[str], condition: bool, message: str) -> None:
    if not condition:
        warnings.append(message)


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _json(path: str) -> dict | None:
    p = ROOT / path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _pending_or_missing_of_artifact(warnings: list[str], rel: str) -> None:
    payload = _json(rel)
    if payload is None:
        warnings.append(f"PENDING: {rel} is missing (OpenFOAM-generated; do not fabricate)")
        return
    status = payload.get("status")
    executed = payload.get("openfoam_executed")
    if status == "COMPLETE" and executed is True:
        print(f"PASS: {rel} is COMPLETE with openfoam_executed=true")
        return
    if status == "PENDING" or executed is False:
        warnings.append(
            f"PENDING/WARNING: {rel} status={status!r} openfoam_executed={executed!r}"
        )
        return
    warnings.append(f"WARNING: {rel} has unexpected status={status!r} openfoam_executed={executed!r}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    required_files = [
        "README.md",
        "REPRODUCTION.md",
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        "THIRD_PARTY.md",
        "ARTIFACT_PROVENANCE.md",
        "GOAI_OPEN_EXPLORATION_CHECKLIST.md",
        "FINAL_DEFENSE.md",
        "FINAL_ONE_PAGER.md",
        "FINAL_VALIDATION.md",
        "VERIFICATION.md",
        "FAILURE_ANALYSIS.md",
        "artifacts/manifest.json",
        "artifacts/mixing_atlas.json",
        "artifacts/of_test.json",
        "artifacts/cfd_live_summary.json",
        "app/dashboard.py",
        "src/cswe/environment.py",
        "src/cswe/agent.py",
        "src/cswe/baseline.py",
        "src/cswe/verification.py",
        "src/cswe/final_validation.py",
        "src/cswe/ablation.py",
        "src/cswe/sample_efficiency.py",
        ".github/workflows/ci.yml",
    ]
    for rel in required_files:
        require(errors, (ROOT / rel).exists(), f"missing required file: {rel}")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    readme = text("README.md")
    reproduce = text("REPRODUCTION.md")
    third_party = text("THIRD_PARTY.md")
    provenance = text("ARTIFACT_PROVENANCE.md")
    agent = text("src/cswe/agent.py")
    dashboard = text("app/dashboard.py")
    cli = text("src/cswe/cli.py")
    ci = text(".github/workflows/ci.yml")

    require(
        errors,
        "## Scoring object for GOAI Type II" not in readme,
        "README still contains the unsupported percentage-scoring section",
    )
    require(
        errors,
        "Mapping to the GOAI Open Exploration judging dimensions" in readme,
        "README lacks explicit GOAI judging-dimension mapping",
    )
    require(errors, "Follow-up research paths" in readme, "README lacks follow-up research paths")
    require(errors, "External datasets: **none**" in readme, "README lacks external-data disclosure")
    require(errors, "Sample efficiency" in readme, "README lacks Sample efficiency section")
    require(
        errors,
        "Which part of the AI policy actually creates the advantage?" in readme,
        "README lacks ablation question",
    )
    require(errors, "### Expected outputs" in reproduce, "REPRODUCTION.md lacks expected outputs")
    require(errors, "Python: **>= 3.11**" in reproduce, "REPRODUCTION.md lacks Python version")
    require(errors, "OpenFOAM 14" in reproduce, "REPRODUCTION.md lacks OpenFOAM version")
    require(errors, "python3 -m venv .venv" in reproduce, "REPRODUCTION.md lacks standard venv path")
    require(errors, "python3 -m pip install -e \".[dev]\"" in reproduce, "REPRODUCTION.md lacks pip editable install")
    require(errors, "python3 -m pytest" in reproduce, "REPRODUCTION.md lacks pip pytest command")
    require(errors, "cswe reproduce" in reproduce, "REPRODUCTION.md lacks cswe reproduce")
    require(errors, "python3 tools/check_open_exploration.py" in reproduce, "REPRODUCTION.md lacks checker command")
    require(errors, "streamlit run app/dashboard.py" in reproduce, "REPRODUCTION.md lacks streamlit command")
    require(errors, "External datasets" in third_party, "THIRD_PARTY.md lacks data disclosure")
    require(errors, "Commercial APIs" in third_party, "THIRD_PARTY.md lacks API disclosure")
    require(errors, "Cconv" in provenance and "Cvalid" in provenance, "provenance lacks schema compatibility note")
    require(
        errors,
        '"status": "challenged" if counterexamples else "not_challenged"' in agent,
        "agent still treats sparse multivariate swirl evidence as causal falsification",
    )
    require(errors, "Final Demo — 90 seconds" in dashboard, "dashboard lacks Final Demo tab")
    require(errors, "def verify" in cli, "CLI lacks cswe verify")
    require(errors, "final-validation" in cli, "CLI lacks cswe final-validation")
    require(errors, "ablation-study" in cli, "CLI lacks cswe ablation-study")
    require(errors, "live-expansion" in cli, "CLI lacks cswe live-expansion")
    require(errors, "g-sweep-verify" in cli, "CLI lacks cswe g-sweep-verify")
    require(errors, "3.11" in ci, "CI workflow is not pinned to Python 3.11")
    require(errors, "pip install -e" in ci, "CI does not pip-install the package")
    require(errors, "pytest" in ci, "CI does not run pytest")
    require(errors, "check_open_exploration.py" in ci, "CI does not run the self-check")

    manifest = json.loads(text("artifacts/manifest.json"))
    require(errors, manifest.get("project", {}).get("subtrack") == "Open Exploration / Type II", "manifest sub-track missing")
    require(errors, "submission_contract" in manifest, "manifest lacks three-piece-set contract")
    require(errors, "artifact_provenance" in manifest, "manifest lacks artifact provenance")
    require(errors, "disclosure" in manifest, "manifest lacks disclosure block")
    require(errors, not manifest["disclosure"].get("external_datasets"), "manifest declares external datasets unexpectedly")
    require(errors, not manifest["disclosure"].get("commercial_apis"), "manifest declares commercial APIs unexpectedly")

    live_dirs = sorted((ROOT / "artifacts").glob("cfd_study_s*"))
    frozen_names = {f"cfd_study_s{s}" for s in (8, 11, 14, 19, 23, 26, 32, 35)}
    require(errors, len(live_dirs) == 8, f"expected 8 frozen live CFD study dirs, found {len(live_dirs)}")
    require(
        errors,
        {d.name for d in live_dirs} == frozen_names,
        f"frozen live CFD dirs drifted: {[d.name for d in live_dirs]}",
    )
    for directory in live_dirs:
        require(errors, (directory / "ai" / "exploration.jsonl").exists(), f"missing AI exploration log in {directory.name}")
        require(errors, (directory / "baseline" / "exploration.jsonl").exists(), f"missing baseline exploration log in {directory.name}")
        require(errors, (directory / "cfd_comparison.json").exists(), f"missing comparison in {directory.name}")
    expansion_dirs = sorted((ROOT / "artifacts").glob("cfd_expansion_s*"))
    for directory in expansion_dirs:
        require(
            errors,
            not directory.name.startswith("cfd_study_"),
            f"expansion directory collides with frozen glob: {directory.name}",
        )
    _pending_or_missing_of_artifact(warnings, "artifacts/cfd_live_expansion.json")
    _pending_or_missing_of_artifact(warnings, "artifacts/g_sweep_mesh_verification.json")

    # Atlas ablation + sample-efficiency are generated from committed data (no new CFD).
    require(errors, (ROOT / "src/cswe/ablation.py").exists(), "missing ablation infrastructure")
    if (ROOT / "artifacts" / "figures" / "sample_efficiency_live.png").exists():
        require(errors, (ROOT / "artifacts" / "sample_efficiency_live.json").exists(), "sample-efficiency figure present without JSON")
    else:
        warnings.append("WARNING: artifacts/figures/sample_efficiency_live.png not generated yet")

    if (ROOT / "artifacts" / "ablation_study.json").exists():
        ab = json.loads(text("artifacts/ablation_study.json"))
        require(errors, ab.get("backend") == "atlas", "ablation_study.json is not marked as atlas-backed")
        require(errors, ab.get("openfoam_executed") is False, "ablation_study.json unexpectedly claims live OpenFOAM")
    else:
        warnings.append("WARNING: artifacts/ablation_study.json not generated yet")

    _pending_or_missing_of_artifact(warnings, "artifacts/verification.json")
    _pending_or_missing_of_artifact(warnings, "artifacts/final_validation.json")
    if (ROOT / "artifacts" / "figures" / "numerical_verification.png").exists():
        v = _json("artifacts/verification.json") or {}
        if not v.get("openfoam_executed"):
            errors.append("numerical_verification.png exists without openfoam_executed=true (do not fabricate)")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print("PASS: mandatory source/document infrastructure is present.")
    print(f"PASS: found {len(live_dirs)} live matched CFD study directories.")
    print("PASS: historical numerical artifacts were not required to be rewritten.")
    for message in warnings:
        print(message)
    if warnings:
        print("NOTE: PENDING/WARNING items are optional OpenFOAM numerical products, not a false PASS on fabricated CFD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
