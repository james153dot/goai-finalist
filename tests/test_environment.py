from cswe.environment import ExplorationEnv
from cswe.physics import STABILITY_THRESHOLD, simulate, true_stability


def test_simulate_returns_finite_for_interior_point():
    result = simulate({"g": 1.0, "d": 0.2, "a": 0.5, "s": 0.5, "o": 0.4})
    assert result.Cconv
    assert result.S > 0
    assert result.stable == (result.S < STABILITY_THRESHOLD)


def test_true_map_has_both_regimes():
    stables = 0
    unstables = 0
    env = ExplorationEnv(seed=1)
    for _ in range(80):
        x = env.sample_uniform()
        if true_stability(x):
            stables += 1
        else:
            unstables += 1
    assert stables > 5
    assert unstables > 5


def test_hidden_island_is_unstable():
    island = {"g": 0.35, "d": 0.55, "a": 0.32, "s": 0.25, "o": 0.72}
    calm = {"g": 1.6, "d": 0.1, "a": 0.7, "s": 0.35, "o": 0.3}
    assert true_stability(island) is False
    assert true_stability(calm) is True


def test_swirl_is_not_monotone_at_high_swirl():
    low = {"g": 1.2, "d": 0.10, "a": 0.4, "s": 0.15, "o": 0.3}
    high = {"g": 1.2, "d": 0.10, "a": 0.4, "s": 0.95, "o": 0.3}
    assert true_stability(low) is True
    assert true_stability(high) is False


def test_agent_and_baseline_same_budget():
    from cswe.agent import LevelSetAgent
    from cswe.baseline import LatinHypercubeBaseline

    budget = 12
    ai = LevelSetAgent(ExplorationEnv(seed=3), n_init=4, n_candidates=80).run(budget)
    base = LatinHypercubeBaseline(ExplorationEnv(seed=4)).run(budget)
    assert len(ai.evaluations) == budget
    assert len(base.evaluations) == budget
    assert ai.discoveries
