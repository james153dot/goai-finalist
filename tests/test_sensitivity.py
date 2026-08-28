from cswe.physics import relabel_mixing_row, unstable_runs_1d


def test_unstable_runs_two_intervals():
    g = [0.05, 0.29, 0.53, 0.76, 1.00, 1.47, 1.71, 1.95]
    sig = [0.5, 0.2, -0.1, -0.3, -0.2, 0.08, 0.08, -0.06]
    runs = unstable_runs_1d(g, sig)
    assert len(runs) == 2
    assert runs[0][0] == 0.05
    assert runs[1][0] == 1.47


def test_relabel_respects_damping():
    row = {
        "Cconv": True,
        "tau": 0.086,
        "Um": 0.94,
        "o": 0.45,
        "R_spatial": 0.73,
        "compactness": 5.3,
    }
    lo = relabel_mixing_row(row, damping=0.04)
    hi = relabel_mixing_row(row, damping=0.20)
    assert lo["sigma"] > hi["sigma"]
