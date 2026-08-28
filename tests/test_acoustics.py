"""Acoustics unit tests that do not require the mixing atlas or OpenFOAM."""

from cswe.geometry import jet_layout
from cswe.physics import _acoustics


def test_short_delay_drives_closed_closed_1L():
    n_index, omega, rayleigh, sigma, S, phase = _acoustics(
        tau=0.086, um=0.94, o=0.45, R_spatial=0.73, compactness=5.3
    )
    assert phase > 0.4
    assert sigma > 0
    assert S > 1.0


def test_long_delay_damps():
    n_index, omega, rayleigh, sigma, S, phase = _acoustics(
        tau=0.231, um=0.77, o=0.55, R_spatial=0.58, compactness=2.6
    )
    assert phase < 0
    assert sigma < 0
    assert S < 1.0


def test_front_loaded_overlap_is_more_driving():
    *_, sigma_front, _, _ = _acoustics(0.10, 0.9, 0.5, R_spatial=0.75, compactness=4.0)
    *_, sigma_aft, _, _ = _acoustics(0.10, 0.9, 0.5, R_spatial=0.35, compactness=4.0)
    assert sigma_front > sigma_aft


def test_jet_slots_are_ordered():
    layout = jet_layout(1.0, 0.2, 0.5, 0.4, 0.5)
    assert 0 < layout.y0_lo < layout.y0_hi < layout.y1_lo < layout.y1_hi < 0.021
    lo = jet_layout(0.15, 0.08, 0.55, 0.08, 0.45)
    hi = jet_layout(1.85, 0.10, 0.28, 0.82, 0.55)
    # Swirl-coaxial analog offsets the pair farther apart than like-on-like.
    assert (hi.y1_lo - hi.y0_hi) > (lo.y1_lo - lo.y0_hi)
