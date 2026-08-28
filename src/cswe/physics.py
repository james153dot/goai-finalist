"""Closed-closed 1L acoustics driven by an OpenFOAM mixing field.

OpenFOAM determines spatial and temporal mixing features from a fixed-geometry
2-D laminar dual-jet mixing environment. The acoustic model converts those
features into a hypothesis-level Rayleigh stability indicator σ_analog(z).
The project therefore evaluates an autonomous exploration method for
combustion-stability *analogs*, not predictive stability of a real rocket
combustor.

The design / exploration vector is z = [g, d, a, s, o]. Axial chamber
position is x. Pipeline: z → OpenFOAM → q_proxy(x) → σ_analog(z).

Heat-release proxy q_proxy(x) = Var_y[Z](x) is the cross-stream variance of
mixture fraction Z (OpenFOAM field name: T). Large Var_y[Z] means the two
streams are still unmixed at that station, so mixing-limited reaction could
still occur there. Fully mixed stations (Var → 0) contribute no further
proxy heat release. This is not a finite-rate flame.

The chamber pressure mode is the declared first longitudinal of a
closed-closed duct, p(x) = cos(π x / L), so the injector face is a pressure
antinode. R_spatial = ∫ q_proxy p dx / ∫ q_proxy dx. Mixing delay τ is
defined from the same field (first axial bin with Var_y[Z] < 0.045, then
τ = x_m / U_b). There is no planted island.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping

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

# Fixed chamber acoustics: closed-closed 1L analog. ω is not an explorable.
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
            "sigma_analog": self.sigma,
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
    damping: float | None = None,
    omega0: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Map mixing features to the Rayleigh analog σ_analog.

    OpenFOAM supplies τ, unmixedness, R_spatial, compactness. This function
    only converts those features into a hypothesis-level stability indicator.
    """
    if R_spatial is None or R_spatial != R_spatial:
        R_spatial = 0.35
    if compactness is None or compactness != compactness:
        compactness = 1.4
    damping = ACOUSTIC_DAMPING if damping is None else float(damping)
    omega0 = CHAMBER_OMEGA if omega0 is None else float(omega0)
    n_index = (
        N_BASE
        + N_COMPACT * np.tanh(compactness - 1.0)
        + N_UNMIXED * (1.0 - um)
        + N_LOAD * (o - 0.5) ** 2
    )
    omega = omega0 * (0.92 + 0.16 * o)
    phase = float(np.cos(omega * tau))
    rayleigh = float(n_index * R_spatial * phase)
    sigma = HEAT_RELEASE_SCALE * rayleigh - damping
    if sigma >= 0:
        S = float(np.exp(3.2 * sigma))
    else:
        S = float(1.0 / (1.0 - 4.0 * sigma))
    return float(n_index), float(omega), rayleigh, float(sigma), float(S), phase


@contextmanager
def analog_constants(*, damping: float | None = None, omega0: float | None = None) -> Iterator[None]:
    """Temporarily replace analog damping / base frequency. Mixing fields stay fixed."""
    global ACOUSTIC_DAMPING, CHAMBER_OMEGA
    old = ACOUSTIC_DAMPING, CHAMBER_OMEGA
    if damping is not None:
        ACOUSTIC_DAMPING = float(damping)
    if omega0 is not None:
        CHAMBER_OMEGA = float(omega0)
    try:
        yield
    finally:
        ACOUSTIC_DAMPING, CHAMBER_OMEGA = old


def relabel_mixing_row(row: dict, damping: float | None = None, omega0: float | None = None) -> dict:
    """Recompute σ_analog from stored OpenFOAM mixing features."""
    out = dict(row)
    if not row.get("Cconv"):
        return out
    n_index, omega, rayleigh, sigma, S, phase = _acoustics(
        row["tau"], row["Um"], row["o"], row.get("R_spatial"), row.get("compactness"),
        damping=damping, omega0=omega0,
    )
    out["n_index"] = n_index
    out["R"] = rayleigh
    out["sigma"] = sigma
    out["sigma_analog"] = sigma
    out["S"] = S
    out["phase"] = phase
    out["stable"] = int(S < STABILITY_THRESHOLD)
    return out


def unstable_runs_1d(values: list[float], sigma: list[float]) -> list[tuple[float, float]]:
    """Inclusive [g_lo, g_hi] intervals where σ_analog > 0 along a sorted 1-D slice."""
    pairs = sorted(zip(values, sigma), key=lambda t: t[0])
    runs: list[tuple[float, float]] = []
    start = None
    last = None
    for g, s in pairs:
        if s > 0:
            if start is None:
                start = g
            last = g
        elif start is not None:
            runs.append((float(start), float(last)))
            start = last = None
    if start is not None:
        runs.append((float(start), float(last)))
    return runs


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
