"""Frozen closed-closed 1L acoustics driven by an OpenFOAM mixing field.

Heat-release analog q(x) is the cross-stream mixture variance — the stations
where mixing-limited reaction would still be active. The chamber pressure
mode is the frozen first longitudinal of a closed-closed duct,
p(x) = cos(π x / L), so the injector face is a pressure antinode. The Rayleigh
overlap is ∫ q p dx / ∫ q dx. The same field supplies the convective delay
τ = L_mix / U. There is no planted island.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cswe.mixing import mix as mix_cfd
from cswe.openfoam import MixingReport, run_mixer

PARAM_NAMES = ("g", "d", "a", "s", "o")
PARAM_BOUNDS = {
    "g": (0.0, 2.0),
    "d": (0.0, 1.0),
    "a": (0.0, 1.0),
    "s": (0.0, 1.0),
    "o": (0.0, 1.0),
}

# Frozen chamber: closed-closed 1L analog. ω is not an explorable.
CHAMBER_OMEGA = 11.0
ACOUSTIC_DAMPING = 0.08
STABILITY_THRESHOLD = 1.0  # S(σ=0) = 1; the agent level-set is σ = 0
HEAT_RELEASE_SCALE = 1.45
WALL_HEAT_BASE = 0.45
N_BASE = 0.50
N_COMPACT = 0.28
N_UNMIXED = 0.16
N_LOAD = 0.10


@dataclass(frozen=True)
class SimulationResult:
    x: dict[str, float]
    Ap: float
    f_dom: float
    R: float
    Um: float
    Qw: float
    Cconv: bool
    S: float
    sigma: float
    n_index: float
    tau: float
    stable: bool
    notes: str
    backend: str = "openfoam_atlas"
    R_spatial: float = float("nan")
    compactness: float = float("nan")
    x_q: float = float("nan")
    phase: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "g": self.x["g"],
            "d": self.x["d"],
            "a": self.x["a"],
            "s": self.x["s"],
            "o": self.x["o"],
            "Ap": self.Ap,
            "f_dom": self.f_dom,
            "R": self.R,
            "Um": self.Um,
            "Qw": self.Qw,
            "Cconv": int(self.Cconv),
            "S": self.S,
            "sigma": self.sigma,
            "n_index": self.n_index,
            "tau": self.tau,
            "stable": int(self.stable),
            "notes": self.notes,
            "backend": self.backend,
            "R_spatial": self.R_spatial,
            "compactness": self.compactness,
            "x_q": self.x_q,
            "phase": self.phase,
        }


def clip_params(x: Mapping[str, float]) -> dict[str, float]:
    return {name: float(np.clip(x[name], *PARAM_BOUNDS[name])) for name in PARAM_NAMES}


def vector_from_params(x: Mapping[str, float]) -> np.ndarray:
    x = clip_params(x)
    return np.array([x[n] for n in PARAM_NAMES], dtype=float)


def params_from_vector(v: np.ndarray) -> dict[str, float]:
    return clip_params({n: float(v[i]) for i, n in enumerate(PARAM_NAMES)})


def _acoustics(
    tau: float,
    um: float,
    o: float,
    R_spatial: float | None = None,
    compactness: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    if R_spatial is None or R_spatial != R_spatial:
        R_spatial = 0.35
    if compactness is None or compactness != compactness:
        compactness = 1.4
    n_index = (
        N_BASE
        + N_COMPACT * np.tanh(compactness - 1.0)
        + N_UNMIXED * (1.0 - um)
        + N_LOAD * (o - 0.5) ** 2
    )
    omega = CHAMBER_OMEGA * (0.92 + 0.16 * o)
    phase = float(np.cos(omega * tau))
    # Spatial overlap × time-lag phase. Compact, injector-local mixing on a
    # closed-closed 1L drives; delayed / spread mixing damps.
    rayleigh = float(n_index * R_spatial * phase)
    sigma = HEAT_RELEASE_SCALE * rayleigh - ACOUSTIC_DAMPING
    if sigma >= 0:
        S = float(np.exp(3.2 * sigma))
    else:
        S = float(1.0 / (1.0 - 4.0 * sigma))
    return float(n_index), float(omega), rayleigh, float(sigma), float(S), phase


def _result_from_report(
    x: dict[str, float],
    report: MixingReport,
    rng: np.random.Generator,
    noise: bool,
) -> SimulationResult:
    if not report.Cconv:
        return SimulationResult(
            x=x, Ap=float("nan"), f_dom=float("nan"), R=float("nan"), Um=float("nan"),
            Qw=float("nan"), Cconv=False, S=float("nan"), sigma=float("nan"),
            n_index=float("nan"), tau=float("nan"), stable=False,
            notes=report.notes, backend=report.backend,
            R_spatial=report.R_spatial, compactness=report.compactness,
            x_q=report.x_q,
        )
    n_index, omega, rayleigh, sigma, S, phase = _acoustics(
        report.tau, report.Um, x["o"], report.R_spatial, report.compactness
    )
    sigma_obs = float(sigma + (rng.normal(0.0, 0.018) if noise else 0.0))
    if sigma_obs >= 0:
        Ap = float(np.exp(3.2 * sigma_obs))
    else:
        Ap = float(1.0 / (1.0 - 4.0 * sigma_obs))
    Qw = float(np.clip(WALL_HEAT_BASE + 0.25 * x["o"] + 0.18 * Ap / (1.0 + Ap), 0.1, 1.4))
    return SimulationResult(
        x=x, Ap=Ap, f_dom=float(omega / (2.0 * np.pi)), R=float(rayleigh),
        Um=report.Um, Qw=Qw, Cconv=True, S=float(Ap), sigma=sigma_obs,
        n_index=n_index, tau=report.tau, stable=Ap < STABILITY_THRESHOLD,
        notes="", backend=report.backend,
        R_spatial=report.R_spatial, compactness=report.compactness,
        x_q=report.x_q, phase=phase,
    )


def simulate(
    x: Mapping[str, float],
    rng: np.random.Generator | None = None,
    backend: str = "atlas",
    n_iter: int = 100,
) -> SimulationResult:
    rng = rng or np.random.default_rng()
    x = clip_params(x)
    if backend == "openfoam":
        report = run_mixer(x, n_iter=n_iter)
    else:
        report = mix_cfd(x)
    return _result_from_report(x, report, rng, noise=(backend != "openfoam"))


def true_stability(x: Mapping[str, float]) -> bool:
    x = clip_params(x)
    report = mix_cfd(x)
    if not report.Cconv:
        return False
    *rest, S, _phase = _acoustics(report.tau, report.Um, x["o"], report.R_spatial, report.compactness)
    return S < STABILITY_THRESHOLD
