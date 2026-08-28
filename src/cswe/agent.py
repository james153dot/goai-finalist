"""Active level-set exploration of the stable/unstable boundary.

The agent fits a Gaussian-process surrogate to the acoustic growth rate σ(x)
— not to the exponential amplitude S = f(σ). The stability boundary is the
level set σ = 0. Straddle concentrates evaluations there; if only one class
has been seen, the next point hunts for the missing regime. That is the job
where a space-filling design wastes CFD budget: the unstable class is the
minority.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
import warnings

from cswe.environment import ExplorationEnv
from cswe.physics import PARAM_NAMES, SimulationResult

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Level set the agent actually searches. σ > 0 is unstable.
SIGMA_THRESHOLD = 0.0


def _kernel() -> ConstantKernel:
    return ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(len(PARAM_NAMES)),
        length_scale_bounds=(0.12, 6.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=0.02, noise_level_bounds=(1e-4, 0.2))


@dataclass
class CampaignRecord:
    method: str
    seed: int
    budget: int
    evaluations: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    discoveries: list[dict] = field(default_factory=list)

    def add_eval(self, result: SimulationResult, acquisition: float | None = None) -> None:
        row = result.as_dict()
        row["t"] = len(self.evaluations)
        row["acquisition"] = acquisition
        self.evaluations.append(row)


class LevelSetAgent:
    """Straddle search on σ=0 with regime-seeking and a space-filling start.

    Initial points are a Latin hypercube so the comparison with an LHS
    baseline is not 'random start vs designed start'. After that the remaining
    budget is spent on the missing class, then on the boundary.
    """

    def __init__(
        self,
        env: ExplorationEnv,
        n_init: int = 6,
        n_candidates: int = 700,
        epsilon: float = 0.10,
    ) -> None:
        self.env = env
        self.n_init = n_init
        self.n_candidates = n_candidates
        self.epsilon = epsilon
        self.gp = GaussianProcessRegressor(
            kernel=_kernel(),
            normalize_y=True,
            n_restarts_optimizer=4,
            random_state=env.seed,
        )
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    def _fit(self) -> None:
        X = np.vstack(self.X)
        y = np.array(self.y, dtype=float)
        self.gp.fit(X, y)

    def _straddle(self, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
        return 1.96 * std - np.abs(mu - SIGMA_THRESHOLD)

    def _sample_candidates(self, rng: np.random.Generator) -> np.ndarray:
        bounds = self.env.bounds_array()
        u = rng.random((self.n_candidates, len(PARAM_NAMES)))
        return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])

    def _init_design(self, n: int) -> list[dict[str, float]]:
        sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=self.env.seed)
        unit = sampler.random(n=n)
        bounds = self.env.bounds_array()
        scaled = qmc.scale(unit, bounds[:, 0], bounds[:, 1])
        return [self.env.from_vector(row) for row in scaled]

    def _choose(self, rng: np.random.Generator) -> tuple[dict[str, float], float | None]:
        if len(self.y) < 3 or rng.random() < self.epsilon:
            return self.env.sample_uniform(), None
        self._fit()
        cand = self._sample_candidates(rng)
        mu, std = self.gp.predict(cand, return_std=True)
        y = np.array(self.y, dtype=float)
        seen_unstable = np.any(y > SIGMA_THRESHOLD)
        seen_stable = np.any(y <= SIGMA_THRESHOLD)
        if not seen_unstable:
            # Only stables so far: hunt the missing unstable class (UCB on σ).
            scores = mu + 1.9 * std
        elif not seen_stable:
            scores = -(mu - 1.9 * std)
        else:
            scores = self._straddle(mu, std)
        idx = int(np.argmax(scores))
        return self.env.from_vector(cand[idx]), float(scores[idx])

    def run(self, budget: int) -> CampaignRecord:
        record = CampaignRecord(method="ai_level_set", seed=self.env.seed, budget=budget)
        rng = self.env.rng

        for x in self._init_design(min(self.n_init, budget)):
            result = self.env.evaluate(x)
            record.add_eval(result, acquisition=None)
            if result.Cconv and result.sigma == result.sigma:
                self.X.append(self.env.to_vector(x))
                self.y.append(result.sigma)

        while len(record.evaluations) < budget:
            x, acq = self._choose(rng)
            result = self.env.evaluate(x)
            record.add_eval(result, acquisition=acq)
            if result.Cconv and result.sigma == result.sigma:
                self.X.append(self.env.to_vector(x))
                self.y.append(result.sigma)

        record.hypotheses = _assess_hypotheses(record)
        record.discoveries = _extract_discoveries(record)
        return record


def _valid_rows(record: CampaignRecord) -> list[dict]:
    return [r for r in record.evaluations if r["Cconv"] == 1]


def _is_unstable(r: dict) -> bool:
    if r.get("sigma") == r.get("sigma") and r.get("sigma") is not None:
        return float(r["sigma"]) > SIGMA_THRESHOLD
    return float(r.get("S", 0.0)) >= 1.0


def _assess_hypotheses(record: CampaignRecord) -> list[dict]:
    rows = _valid_rows(record)
    if len(rows) < 8:
        return []

    # H1: "increasing swirl monotonically improves stability."
    counterexamples = []
    for r in rows:
        near = [
            q
            for q in rows
            if abs(q["d"] - r["d"]) < 0.2
            and abs(q["g"] - r["g"]) < 0.6
            and q["s"] > r["s"] + 0.15
        ]
        if not near:
            continue
        if np.mean([q["S"] for q in near]) > r["S"] + 0.15 and r["S"] < 1.0:
            if any(_is_unstable(q) for q in near):
                counterexamples.append(r)

    h1 = {
        "id": "H_swirl_monotonic",
        "statement": "Increasing swirl intensity analog s consistently improves stability.",
        "status": "falsified" if counterexamples else "not_falsified",
        "evidence_count": len(counterexamples),
        "note": (
            "Higher swirl analog changed mixing delay and Rayleigh overlap enough "
            "to cross σ = 0, so 'more swirl is always more stable' fails on this CFD map."
            if counterexamples
            else "No clear counterexample in this budget."
        ),
    }

    island = []
    for r in rows:
        if not _is_unstable(r):
            continue
        others = [q for q in rows if q is not r]
        if not others:
            continue
        dist = [
            abs(q["g"] - r["g"]) + abs(q["d"] - r["d"]) + abs(q["s"] - r["s"]) + abs(q["o"] - r["o"])
            for q in others
        ]
        near = [others[i] for i in np.argsort(dist)[:6]]
        if sum(not _is_unstable(q) for q in near) >= 4:
            island.append(r)
    h2 = {
        "id": "H_connected_unstable",
        "statement": "Unstable conditions form a single connected region in the explored domain.",
        "status": "challenged" if len(island) >= 2 else "not_challenged",
        "evidence_count": len(island),
        "note": (
            "At least two unstable evaluations sit among mostly stable neighbors, "
            "which is consistent with a disconnected pocket in the CFD mixing map."
            if len(island) >= 2
            else "The campaign did not isolate a disconnected pocket."
        ),
    }
    return [h1, h2]


def _extract_discoveries(record: CampaignRecord) -> list[dict]:
    rows = _valid_rows(record)
    discoveries = []
    if not rows:
        return discoveries

    stables = [r for r in rows if not _is_unstable(r)]
    unstables = [r for r in rows if _is_unstable(r)]
    if stables and unstables:
        discoveries.append(
            {
                "type": "stability_window",
                "n_stable": len(stables),
                "n_unstable": len(unstables),
                "stable_fraction": len(stables) / len(rows),
                "note": "A verified stable/unstable partition exists under the Rayleigh-from-mixing criterion.",
            }
        )

    boundary = [r for r in rows if abs(float(r.get("sigma", r["S"] - 1.0))) < 0.12]
    if boundary:
        discoveries.append(
            {
                "type": "boundary_structure",
                "n_near_threshold": len(boundary),
                "mean_swirl_on_boundary": float(np.mean([r["s"] for r in boundary])),
                "mean_spread_on_boundary": float(np.mean([r["d"] for r in boundary])),
                "note": "Evaluations concentrated near σ ≈ 0 reconstruct the window edge.",
            }
        )

    for hyp in record.hypotheses:
        if hyp["status"] in {"falsified", "challenged"}:
            discoveries.append(
                {
                    "type": "counterexample" if hyp["status"] == "falsified" else "anomaly",
                    "hypothesis": hyp["id"],
                    "evidence_count": hyp["evidence_count"],
                    "note": hyp["note"],
                }
            )
    return discoveries
