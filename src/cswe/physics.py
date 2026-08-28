"""Frozen Crocco n-τ acoustics on top of OpenFOAM injector mixing.

The chamber mode, damping, and stability threshold are fixed. The only
injector-dependent inputs are the CFD mixing delay τ and unmixedness Um.
There is no planted island and no algebraic swirl counterexample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cswe.mixing import mix as mix_cfd

PARAM_NAMES = ("g", "d", "a", "s", "o")
PARAM_BOUNDS = {
    "g": (0.0, 2.0),
    "d": (0.0, 1.0),
    "a": (0.0, 1.0),
    "s": (0.0, 1.0),
    "o": (0.0, 1.0),
}

# Acoustic constants are frozen. TAU_REF is the typical OpenFOAM mixing delay
# so ωτ sits near the Rayleigh sign change for this chamber.
CHAMBER_OMEGA = 11.0  # rad/s, first-longitudinal analog of the frozen chamber
ACOUSTIC_DAMPING = 0.07
STABILITY_THRESHOLD = 1.0
HEAT_RELEASE_SCALE = 1.05
WALL_HEAT_BASE = 0.45
TAU_REF = 0.18
N_BASE = 0.40
N_UNMIXED = 0.55
N_LOAD = 0.20


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
        }


def clip_params(x: Mapping[str, float]) -> dict[str, float]:
    return {name: float(np.clip(x[name], *PARAM_BOUNDS[name])) for name in PARAM_NAMES}


def vector_from_params(x: Mapping[str, float]) -> np.ndarray:
    x = clip_params(x)
    return np.array([x[n] for n in PARAM_NAMES], dtype=float)


def params_from_vector(v: np.ndarray) -> dict[str, float]:
    return clip_params({n: float(v[i]) for i, n in enumerate(PARAM_NAMES)})


def _acoustics(tau: float, um: float, o: float) -> tuple[float, float, float, float, float]:
    n_index = N_BASE + N_UNMIXED * (1.0 - um) + N_LOAD * (o - 0.5) ** 2
    omega = CHAMBER_OMEGA * (0.92 + 0.16 * o)
    rayleigh = n_index * np.cos(omega * tau)
    sigma = HEAT_RELEASE_SCALE * rayleigh - ACOUSTIC_DAMPING
    if sigma >= 0:
        S = float(np.exp(3.2 * sigma))
    else:
        S = float(1.0 / (1.0 - 4.0 * sigma))
    return float(n_index), float(omega), float(rayleigh), float(sigma), float(S)


def simulate(x: Mapping[str, float], rng: np.random.Generator | None = None) -> SimulationResult:
    rng = rng or np.random.default_rng()
    x = clip_params(x)
    report = mix_cfd(x)
    if not report.Cconv:
        return SimulationResult(
            x=x, Ap=float("nan"), f_dom=float("nan"), R=float("nan"), Um=float("nan"),
            Qw=float("nan"), Cconv=False, S=float("nan"), sigma=float("nan"),
            n_index=float("nan"), tau=float("nan"), stable=False,
            notes=report.notes, backend=report.backend,
        )
    n_index, omega, rayleigh, sigma, S = _acoustics(report.tau, report.Um, x["o"])
    sigma_obs = float(sigma + rng.normal(0.0, 0.02))
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
    )


def true_stability(x: Mapping[str, float]) -> bool:
    x = clip_params(x)
    report = mix_cfd(x)
    if not report.Cconv:
        return False
    *_, S = _acoustics(report.tau, report.Um, x["o"])
    return S < STABILITY_THRESHOLD
