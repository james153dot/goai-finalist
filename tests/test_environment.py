from pathlib import Path

import pytest

from cswe.environment import ExplorationEnv
from cswe.geometry import CLASSICAL_INJECTORS
from cswe.physics import STABILITY_THRESHOLD, simulate, true_stability

ATLAS = Path(__file__).resolve().parents[1] / "artifacts" / "mixing_atlas.json"
pytestmark = pytest.mark.skipif(not ATLAS.exists(), reason="mixing atlas not built yet")


def test_simulate_returns_finite_for_interior_point():
    result = simulate({"g": 1.0, "d": 0.2, "a": 0.5, "s": 0.5, "o": 0.4})
    assert result.Cconv
    assert result.S > 0
    assert result.stable == (result.S < STABILITY_THRESHOLD)
    assert result.backend == "openfoam_atlas"
    assert result.sigma == result.sigma


def test_true_map_has_both_regimes():
    env = ExplorationEnv(seed=1)
    stables = unstables = 0
    for _ in range(80):
        if true_stability(env.sample_uniform()):
            stables += 1
        else:
            unstables += 1
    assert stables > 5
    assert unstables > 5


def test_classical_injectors_evaluate():
    for name, x in CLASSICAL_INJECTORS.items():
        r = simulate(x)
        assert r.Cconv, name
        assert r.tau > 0
        assert r.sigma == r.sigma


def test_classical_regimes_match_cfd_story():
    from cswe.geometry import CLASSICAL_INJECTORS

    lol = simulate(CLASSICAL_INJECTORS["like_on_like"])
    unl = simulate(CLASSICAL_INJECTORS["unlike_impinging"])
    sw = simulate(CLASSICAL_INJECTORS["swirl_coaxial"])
    assert not lol.stable
    assert unl.stable
    assert sw.stable
    assert lol.tau < unl.tau < sw.tau
    assert lol.R_spatial > sw.R_spatial


def test_agent_and_baseline_same_budget():
    from cswe.agent import LevelSetAgent
    from cswe.baseline import LatinHypercubeBaseline

    budget = 12
    ai = LevelSetAgent(ExplorationEnv(seed=3), n_init=4, n_candidates=80).run(budget)
    base = LatinHypercubeBaseline(ExplorationEnv(seed=4)).run(budget)
    assert len(ai.evaluations) == budget
    assert len(base.evaluations) == budget
    assert all("sigma" in r for r in ai.evaluations)
