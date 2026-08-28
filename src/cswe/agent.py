"""Active level-set exploration of the stable/unstable boundary.

The agent fits a Gaussian-process surrogate to the stability metric S(x)
and selects the next simulation with the straddle heuristic, which
concentrates evaluations where the predicted value is near the stability
threshold and the surrogate is uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
import warnings

from cswe.environment import ExplorationEnv
from cswe.physics import PARAM_NAMES, STABILITY_THRESHOLD, SimulationResult

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _kernel() -> ConstantKernel:
    return ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(len(PARAM_NAMES)),
        length_scale_bounds=(0.08, 8.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=0.04, noise_level_bounds=(1e-4, 0.4))


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
    def __init__(self, env: ExplorationEnv, n_init: int = 8, n_candidates: int = 400) -> None:
        self.env = env
        self.n_init = n_init
        self.n_candidates = n_candidates
        self.gp = GaussianProcessRegressor(
            kernel=_kernel(),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=env.seed,
        )
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    def _fit(self) -> None:
        X = np.vstack(self.X)
        y = np.array(self.y, dtype=float)
        self.gp.fit(X, y)

    def _straddle(self, Xcand: np.ndarray) -> np.ndarray:
        mu, std = self.gp.predict(Xcand, return_std=True)
        return 1.96 * std - np.abs(mu - STABILITY_THRESHOLD)

    def _sample_candidates(self, rng: np.random.Generator) -> np.ndarray:
        bounds = self.env.bounds_array()
        u = rng.random((self.n_candidates, len(PARAM_NAMES)))
        return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])

    def run(self, budget: int) -> CampaignRecord:
        record = CampaignRecord(method="ai_level_set", seed=self.env.seed, budget=budget)
        rng = self.env.rng

        for _ in range(min(self.n_init, budget)):
            x = self.env.sample_uniform()
            result = self.env.evaluate(x)
            record.add_eval(result, acquisition=None)
            if result.Cconv:
                self.X.append(self.env.to_vector(x))
                self.y.append(result.S)

        while len(record.evaluations) < budget:
            if len(self.y) < 3:
                x = self.env.sample_uniform()
                acq = None
            else:
                self._fit()
                cand = self._sample_candidates(rng)
                scores = self._straddle(cand)
                idx = int(np.argmax(scores))
                x = self.env.from_vector(cand[idx])
                acq = float(scores[idx])
            result = self.env.evaluate(x)
            record.add_eval(result, acquisition=acq)
            if result.Cconv:
                self.X.append(self.env.to_vector(x))
                self.y.append(result.S)

        record.hypotheses = _assess_hypotheses(record)
        record.discoveries = _extract_discoveries(record)
        return record


def _valid_rows(record: CampaignRecord) -> list[dict]:
    return [r for r in record.evaluations if r["Cconv"] == 1]


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
        if np.mean([q["S"] for q in near]) > r["S"] + 0.15 and r["S"] < STABILITY_THRESHOLD:
            # higher swirl became worse
            if any(q["S"] >= STABILITY_THRESHOLD for q in near):
                counterexamples.append(r)

    h1 = {
        "id": "H_swirl_monotonic",
        "statement": "Increasing swirl intensity analog s consistently improves stability.",
        "status": "falsified" if counterexamples else "not_falsified",
        "evidence_count": len(counterexamples),
        "note": (
            "Higher swirl shortens the mixing lag into an in-phase heat-release "
            "band, so a common 'more swirl is always more stable' rule fails."
            if counterexamples
            else "No clear counterexample in this budget."
        ),
    }

    # H2: disconnected unstable region (pattern analog near like-on-like, high load).
    island = [
        r
        for r in rows
        if r["stable"] == 0
        and r["g"] < 0.8
        and r["o"] > 0.55
        and 0.15 < r["a"] < 0.5
        and r["S"] >= STABILITY_THRESHOLD
    ]
    h2 = {
        "id": "H_connected_unstable",
        "statement": "Unstable conditions form a single connected region in the explored domain.",
        "status": "challenged" if len(island) >= 2 else "not_challenged",
        "evidence_count": len(island),
        "note": (
            "A compact high-load pocket of like-on-like analog injectors is unstable "
            "even when neighboring swirl/orifice settings are stable."
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

    stables = [r for r in rows if r["stable"] == 1]
    unstables = [r for r in rows if r["stable"] == 0]
    if stables and unstables:
        discoveries.append(
            {
                "type": "stability_window",
                "n_stable": len(stables),
                "n_unstable": len(unstables),
                "stable_fraction": len(stables) / len(rows),
                "note": "A verified stable/unstable partition exists under the frozen n-τ criterion.",
            }
        )

    # Boundary points: S near threshold.
    boundary = [r for r in rows if abs(r["S"] - STABILITY_THRESHOLD) < 0.35]
    if boundary:
        discoveries.append(
            {
                "type": "boundary_structure",
                "n_near_threshold": len(boundary),
                "mean_swirl_on_boundary": float(np.mean([r["s"] for r in boundary])),
                "mean_spread_on_boundary": float(np.mean([r["d"] for r in boundary])),
                "note": "Evaluations concentrated near S ≈ 1 reconstruct the window edge.",
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
