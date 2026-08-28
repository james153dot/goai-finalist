"""Budget-matched non-adaptive baselines: uniform random and Latin hypercube."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from cswe.agent import CampaignRecord, _assess_hypotheses, _extract_discoveries
from cswe.environment import ExplorationEnv
from cswe.physics import PARAM_NAMES


class LatinHypercubeBaseline:
    def __init__(self, env: ExplorationEnv) -> None:
        self.env = env

    def run(self, budget: int) -> CampaignRecord:
        record = CampaignRecord(method="lhs_baseline", seed=self.env.seed, budget=budget)
        sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=self.env.seed)
        unit = sampler.random(n=budget)
        bounds = self.env.bounds_array()
        scaled = qmc.scale(unit, bounds[:, 0], bounds[:, 1])
        for row in scaled:
            x = self.env.from_vector(row)
            result = self.env.evaluate(x)
            record.add_eval(result, acquisition=None)
        record.hypotheses = _assess_hypotheses(record)
        record.discoveries = _extract_discoveries(record)
        return record


class RandomBaseline:
    def __init__(self, env: ExplorationEnv) -> None:
        self.env = env

    def run(self, budget: int) -> CampaignRecord:
        record = CampaignRecord(method="random_baseline", seed=self.env.seed, budget=budget)
        for _ in range(budget):
            x = self.env.sample_uniform()
            result = self.env.evaluate(x)
            record.add_eval(result, acquisition=None)
        record.hypotheses = _assess_hypotheses(record)
        record.discoveries = _extract_discoveries(record)
        return record
