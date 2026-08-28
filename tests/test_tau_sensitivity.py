import numpy as np

from cswe.openfoam import mixing_delay_from_variance, resample_axial_profile
from cswe.tau_sensitivity import relabel_from_profile, run_tau_sensitivity


def test_mixing_delay_first_bin_below_threshold():
    x = [0.01, 0.02, 0.03, 0.04]
    v = [0.20, 0.10, 0.04, 0.01]
    tau, x_m = mixing_delay_from_variance(x, v, u_bulk=0.2, thresh=0.045)
    assert x_m == 0.03
    assert abs(tau - 0.03 / 0.2) < 1e-12


def test_mixing_delay_fallback_is_last_bin():
    x = [0.01, 0.02, 0.03]
    v = [0.2, 0.2, 0.2]
    tau, x_m = mixing_delay_from_variance(x, v, u_bulk=0.1, thresh=0.045)
    assert x_m == 0.03
    assert abs(tau - 0.3) < 1e-12


def test_resample_identity_at_24():
    x = np.linspace(0.002, 0.078, 24)
    v = np.linspace(0.2, 0.0, 24)
    xr, vr = resample_axial_profile(x, v, 24)
    assert np.allclose(x, xr)
    assert np.allclose(v, vr)


def test_tau_sensitivity_nominal_matches_stored_row():
    row = {
        "Cconv": True,
        "g": 0.15,
        "d": 0.08,
        "a": 0.55,
        "s": 0.08,
        "o": 0.45,
        "Um": 0.9,
        "R_spatial": 0.71,
        "compactness": 5.3,
        "q_profile": [0.11, 0.08, 0.05, 0.03, 0.02] + [0.01] * 19,
        "x_profile": list(np.linspace(0.0025, 0.0775, 24)),
    }
    lab = relabel_from_profile(row, v_crit=0.045, n_bins=24)
    assert lab is not None
    # First bin with V < 0.045 is index 3 (0.03).
    assert lab["x_m"] == row["x_profile"][3]


def test_tau_sensitivity_artifact_qualitative():
    payload = run_tau_sensitivity()
    assert payload["cases"]
    assert payload["classical_ordering_survives"]
    nominal = next(c for c in payload["cases"] if c["nominal"])
    assert nominal["g_slice"]["n_unstable_intervals"] == 2
    # Stricter mixing threshold V_crit=0.035 is the disclosed failure: the
    # high-g interval on the 1-D slice disappears.
    assert payload["g_slice_two_intervals_fail_at"]
    assert any("0.035" in name for name in payload["g_slice_two_intervals_fail_at"])
