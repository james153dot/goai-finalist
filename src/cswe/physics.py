"""Nondimensional Crocco n-τ combustion-acoustic model.

The chamber geometry, propellant family, acoustic boundary conditions, and
stability criterion are frozen. Injector-related variables and a single
operating-condition analog are explorable.

All quantities are abstract and nondimensional. They are not engineering
specifications for a flight injector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

PARAM_NAMES = ("g", "d", "a", "s", "o")
PARAM_BOUNDS = {
    "g": (0.0, 2.0),  # injector-pattern class: like-on-like → unlike → swirl-coaxial analog
    "d": (0.0, 1.0),  # orifice-size spread
    "a": (0.0, 1.0),  # impingement / injection-interaction analog
    "s": (0.0, 1.0),  # swirl intensity analog
    "o": (0.0, 1.0),  # operating-condition analog (mixture ratio / load)
}

# Frozen environment (not explorable).
# Frequency and lag are scaled so ωτ spans roughly π/2 to π and the
# Rayleigh criterion actually changes sign inside the domain.
CHAMBER_OMEGA = 1.42  # first-longitudinal acoustic frequency, nondim
ACOUSTIC_DAMPING = 0.12
STABILITY_THRESHOLD = 1.0  # S < 1 is stable
HEAT_RELEASE_SCALE = 0.80
WALL_HEAT_BASE = 0.45


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

    def as_dict(self) -> dict:
        out = {
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
        }
        return out


def clip_params(x: Mapping[str, float]) -> dict[str, float]:
    clipped = {}
    for name in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[name]
        clipped[name] = float(np.clip(x[name], lo, hi))
    return clipped


def vector_from_params(x: Mapping[str, float]) -> np.ndarray:
    x = clip_params(x)
    return np.array([x[n] for n in PARAM_NAMES], dtype=float)


def params_from_vector(v: np.ndarray) -> dict[str, float]:
    return clip_params({n: float(v[i]) for i, n in enumerate(PARAM_NAMES)})


def _injector_delay(g: float, d: float, a: float, s: float) -> float:
    """Atomization / mixing time lag. Higher swirl shortens the lag.

    The lag is large enough that ωτ can sit near π (out of phase, stable)
    or drop toward π/2 (in phase, unstable) when swirl is high.
    """
    pattern = 0.20 * (2.0 - g)  # unlike/swirl-coaxial mix faster than like-on-like
    spread = 0.35 * d
    impinge = 0.18 * np.cos(np.pi * a)
    swirl = 0.95 * (1.0 - s) ** 1.15
    return 0.48 + pattern + spread + impinge + swirl


def _interaction_index(g: float, d: float, a: float, s: float, o: float) -> float:
    """Crocco interaction index n. Stronger coupling raises growth rate."""
    base = 0.48 + 0.10 * g
    swirl_coupling = 0.22 * s * (1.0 - 0.65 * d)  # swirl helps only if orifices are uniform
    spread_penalty = 0.28 * d * (0.4 + 0.6 * o)
    impinge = 0.16 * np.sin(1.4 * np.pi * (a - 0.15))
    load = 0.12 * (o - 0.5) ** 2
    return base + swirl_coupling + spread_penalty + impinge + load


def _hidden_island(g: float, d: float, a: float, s: float, o: float) -> float:
    """Disconnected unstable pocket: a secondary-mode coupling spike.

    Like-on-like analog injectors at moderate impingement and high load
    excite a 1L/2T-like coupling that one-factor-at-a-time sweeps miss.
    """
    center = np.array([0.35, 0.55, 0.32, 0.25, 0.72])
    point = np.array([g, d, a, s, o])
    scale = np.array([0.55, 0.28, 0.18, 0.35, 0.12])
    dist = np.sum(((point - center) / scale) ** 2)
    return 0.95 * np.exp(-dist)


def _mixture_uniformity(g: float, d: float, a: float, s: float) -> float:
    return float(
        np.clip(
            0.38 + 0.22 * g + 0.28 * s - 0.34 * d + 0.12 * np.sin(np.pi * a),
            0.05,
            0.98,
        )
    )


def simulate(x: Mapping[str, float], rng: np.random.Generator | None = None) -> SimulationResult:
    """Run one frozen-chamber evaluation.

    Numerical non-convergence is injected at extreme orifice spread so the
    agent must treat failed simulations as non-discoveries.
    """
    rng = rng or np.random.default_rng()
    x = clip_params(x)
    g, d, a, s, o = (x[n] for n in PARAM_NAMES)

    tau = _injector_delay(g, d, a, s)
    n_index = _interaction_index(g, d, a, s, o)
    omega = CHAMBER_OMEGA * (0.92 + 0.16 * o)

    # Crocco n-τ growth rate of the first longitudinal mode.
    rayleigh = n_index * np.cos(omega * tau)
    island = _hidden_island(g, d, a, s, o)
    sigma = HEAT_RELEASE_SCALE * (rayleigh + island) - ACOUSTIC_DAMPING

    # Rare solver failures near poorly mixed, high-spread injectors.
    fail_prob = 0.04 * d**2 * (1.0 - s)
    converged = rng.random() > fail_prob

    if not converged:
        return SimulationResult(
            x=x,
            Ap=float("nan"),
            f_dom=float("nan"),
            R=float("nan"),
            Um=float("nan"),
            Qw=float("nan"),
            Cconv=False,
            S=float("nan"),
            sigma=float("nan"),
            n_index=float(n_index),
            tau=float(tau),
            stable=False,
            notes="numerical_nonconvergence",
        )

    # Observation noise is small; discoveries must survive it.
    noise = rng.normal(0.0, 0.03)
    sigma_obs = float(sigma + noise)

    # Map growth rate to a positive amplitude-like metric S.
    # S < 1 → stable; S >= 1 → unstable.
    if sigma_obs >= 0:
        Ap = float(np.exp(3.2 * sigma_obs))
    else:
        Ap = float(1.0 / (1.0 - 4.0 * sigma_obs))
    S = float(Ap)

    Um = _mixture_uniformity(g, d, a, s)
    Qw = float(np.clip(WALL_HEAT_BASE + 0.25 * o + 0.18 * Ap / (1.0 + Ap), 0.1, 1.4))
    f_dom = float(omega / (2.0 * np.pi))
    R = float(rayleigh)

    return SimulationResult(
        x=x,
        Ap=Ap,
        f_dom=f_dom,
        R=R,
        Um=Um,
        Qw=Qw,
        Cconv=True,
        S=S,
        sigma=sigma_obs,
        n_index=float(n_index),
        tau=float(tau),
        stable=S < STABILITY_THRESHOLD,
        notes="",
    )


def true_stability(x: Mapping[str, float]) -> bool:
    """Noise-free stability label for evaluation only (not shown to the agent)."""
    x = clip_params(x)
    g, d, a, s, o = (x[n] for n in PARAM_NAMES)
    tau = _injector_delay(g, d, a, s)
    n_index = _interaction_index(g, d, a, s, o)
    omega = CHAMBER_OMEGA * (0.92 + 0.16 * o)
    rayleigh = n_index * np.cos(omega * tau)
    island = _hidden_island(g, d, a, s, o)
    sigma = HEAT_RELEASE_SCALE * (rayleigh + island) - ACOUSTIC_DAMPING
    if sigma >= 0:
        S = float(np.exp(3.2 * sigma))
    else:
        S = float(1.0 / (1.0 - 4.0 * sigma))
    return S < STABILITY_THRESHOLD
