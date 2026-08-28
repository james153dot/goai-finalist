"""Exploration environment: fixed physics, explorable injector vector, logged feedback."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cswe.physics import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    STABILITY_THRESHOLD,
    SimulationResult,
    params_from_vector,
    simulate,
    vector_from_params,
)


@dataclass
class ExplorationEnv:
    seed: int = 7
    rng: np.random.Generator = field(init=False)
    step: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.step = 0

    def sample_uniform(self) -> dict[str, float]:
        x = {}
        for name in PARAM_NAMES:
            lo, hi = PARAM_BOUNDS[name]
            x[name] = float(self.rng.uniform(lo, hi))
        return x

    def evaluate(self, x: dict[str, float]) -> SimulationResult:
        self.step += 1
        return simulate(x, rng=self.rng)

    @staticmethod
    def bounds_array() -> np.ndarray:
        return np.array([PARAM_BOUNDS[n] for n in PARAM_NAMES], dtype=float)

    @staticmethod
    def to_vector(x: dict[str, float]) -> np.ndarray:
        return vector_from_params(x)

    @staticmethod
    def from_vector(v: np.ndarray) -> dict[str, float]:
        return params_from_vector(v)

    @staticmethod
    def threshold() -> float:
        return STABILITY_THRESHOLD
