"""Fast repository self-check for the GOAI Open Exploration semifinal package."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> int:
    errors: list[str] = []

    required_files = [
        "README.md",
        "REPRODUCTION.md",
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        "THIRD_PARTY.md",
        "ARTIFACT_PROVENANCE.md",
        "GOAI_OPEN_EXPLORATION_CHECKLIST.md",
        "artifacts/manifest.json",
        "artifacts/mixing_atlas.json",
        "artifacts/of_test.json",
        "artifacts/cfd_live_summary.json",
        "app/dashboard.py",
        "src/cswe/environment.py",
        "src/cswe/agent.py",
        "src/cswe/baseline.py",
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
    require(errors, "### Expected outputs" in reproduce, "REPRODUCTION.md lacks expected outputs")
    require(errors, "Python: **>= 3.11**" in reproduce, "REPRODUCTION.md lacks Python version")
    require(errors, "OpenFOAM 14" in reproduce, "REPRODUCTION.md lacks OpenFOAM version")
    require(errors, "External datasets" in third_party, "THIRD_PARTY.md lacks data disclosure")
    require(errors, "Commercial APIs" in third_party, "THIRD_PARTY.md lacks API disclosure")
    require(errors, "Cconv" in provenance and "Cvalid" in provenance, "provenance lacks schema compatibility note")
    require(
        errors,
        '"status": "challenged" if counterexamples else "not_challenged"' in agent,
        "agent still treats sparse multivariate swirl evidence as causal falsification",
    )

    manifest = json.loads(text("artifacts/manifest.json"))
    require(errors, manifest.get("project", {}).get("subtrack") == "Open Exploration / Type II", "manifest sub-track missing")
    require(errors, "submission_contract" in manifest, "manifest lacks three-piece-set contract")
    require(errors, "artifact_provenance" in manifest, "manifest lacks artifact provenance")
    require(errors, "disclosure" in manifest, "manifest lacks disclosure block")
    require(errors, not manifest["disclosure"].get("external_datasets"), "manifest declares external datasets unexpectedly")
    require(errors, not manifest["disclosure"].get("commercial_apis"), "manifest declares commercial APIs unexpectedly")

    live_dirs = sorted((ROOT / "artifacts").glob("cfd_study_s*"))
    require(errors, len(live_dirs) >= 1, "no live CFD study directory found")
    for directory in live_dirs:
        require(errors, (directory / "ai" / "exploration.jsonl").exists(), f"missing AI exploration log in {directory.name}")
        require(errors, (directory / "baseline" / "exploration.jsonl").exists(), f"missing baseline exploration log in {directory.name}")
        require(errors, (directory / "cfd_comparison.json").exists(), f"missing comparison in {directory.name}")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print("PASS: GOAI Open Exploration repository package checks completed successfully.")
    print(f"PASS: found {len(live_dirs)} live matched CFD study directories.")
    print("PASS: numerical artifacts remain historical; schema/interpretation compatibility is documented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
