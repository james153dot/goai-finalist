"""Nondimensional injector layout shared by OpenFOAM and the in-repo mixer.

Chamber length, height, and acoustic boundaries stay fixed. Only the two
inlet slots move. Coordinates are SI meters for OpenFOAM; the exploration
vector z = (g, d, a, s, o) remains abstract.
"""

from __future__ import annotations

from dataclasses import dataclass

H = 0.020  # chamber height [m]
L = 0.080  # chamber length [m]
W = 0.002  # 2-D extrusion [m]
NU = 1.0e-5  # laminar viscosity [m^2/s]
U_REF = 0.18  # reference jet speed [m/s]
T_DIFFUSIVITY = 2.5e-5  # mixture-fraction diffusivity [m^2/s]


@dataclass(frozen=True)
class JetLayout:
    y0_lo: float
    y0_hi: float
    y1_lo: float
    y1_hi: float
    u0: tuple[float, float]
    u1: tuple[float, float]
    u_ref: float


CLASSICAL_INJECTORS: dict[str, dict[str, float]] = {
    "like_on_like": {"g": 0.15, "d": 0.08, "a": 0.55, "s": 0.08, "o": 0.45},
    "unlike_impinging": {"g": 1.00, "d": 0.08, "a": 0.78, "s": 0.10, "o": 0.50},
    "swirl_coaxial": {"g": 1.85, "d": 0.10, "a": 0.28, "s": 0.82, "o": 0.55},
}


def _lerp(g: float, low: float, mid: float, high: float) -> float:
    if g <= 1.0:
        t = g
        return (1.0 - t) * low + t * mid
    t = g - 1.0
    return (1.0 - t) * mid + t * high


def jet_layout(g: float, d: float, a: float, s: float, o: float) -> JetLayout:
    """Map the explorable design vector z onto two inlet slots in a fixed-geometry 2-D chamber."""
    h0 = 0.13 * H * (1.0 + 0.55 * d)
    h1 = 0.13 * H * (1.0 - 0.55 * d)
    c0 = _lerp(g, 0.22 * H, 0.28 * H, 0.50 * H)
    c1 = _lerp(g, 0.40 * H, 0.72 * H, 0.84 * H)
    # Swirl analog: offset the pair and add opposing tangential speed.
    c0 = c0 - 0.06 * H * s
    c1 = c1 + 0.04 * H * s

    def slot(c: float, h: float) -> tuple[float, float]:
        lo = max(0.004 * H, c - 0.5 * h)
        hi = min(H - 0.004 * H, c + 0.5 * h)
        if hi - lo < 0.03 * H:
            hi = min(H - 0.004 * H, lo + 0.03 * H)
        return lo, hi

    y0_lo, y0_hi = slot(c0, h0)
    y1_lo, y1_hi = slot(c1, h1)
    gap = 0.02 * H
    if y1_lo < y0_hi + gap:
        y1_lo = min(H - 0.06 * H, y0_hi + gap)
        y1_hi = min(H - 0.004 * H, y1_lo + max(0.03 * H, h1))
        if y1_hi <= y1_lo:
            y1_hi = min(H - 0.002 * H, y1_lo + 0.03 * H)

    theta = (0.12 + 0.70 * a) * 0.70  # radians, ~7–47 deg
    spin = 0.45 * s
    u0x = U_REF * (0.70 + 0.30 * o)
    u1x = U_REF * (1.30 - 0.30 * o)
    u0y = u0x * (theta - spin)
    u1y = -u1x * (theta + 0.35 * spin)
    return JetLayout(y0_lo, y0_hi, y1_lo, y1_hi, (u0x, u0y), (u1x, u1y), U_REF)
