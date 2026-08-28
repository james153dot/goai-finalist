from cswe.metrics import campaign_diagnostics, is_unstable, score_against_test


def _row(t, sigma, **kw):
    r = {
        "t": t,
        "Cconv": 1,
        "S": 1.2 if sigma > 0 else 0.6,
        "sigma": sigma,
        "g": 0.2,
        "d": 0.1,
        "a": 0.5,
        "s": 0.1,
        "o": 0.4,
        "stable": int(sigma <= 0),
    }
    r.update(kw)
    return r


def test_diagnostics_time_to_first_unstable():
    rows = [_row(0, -0.2), _row(1, -0.1), _row(2, 0.3)]
    d = campaign_diagnostics(rows)
    assert d["time_to_first_unstable"] == 2
    assert d["n_unstable_found"] == 1
    assert is_unstable(rows[2])
    assert not is_unstable(rows[0])


def test_score_against_test_handles_tiny_campaign():
    camp = [_row(i, -0.12 + 0.1 * i, g=0.15 * i) for i in range(3)]
    test = [_row(0, 0.2, g=0.4)]
    out = score_against_test(camp, test)
    assert out["n_unstable_found"] == 1
    assert out["volume_accuracy"] is None  # too few points to fit a GP
    assert out["near_boundary_sigma_mae"] is None
    assert out["boundary_mae"] is None


def test_near_boundary_mae_aliases_boundary_mae():
    camp = [_row(i, -0.3 + 0.12 * i, g=0.08 * i, s=0.1 * (i % 3)) for i in range(8)]
    test = [
        _row(0, 0.05, g=0.2, s=0.1),
        _row(1, -0.10, g=0.4, s=0.2),
        _row(2, 0.40, g=0.8, s=0.5),
    ]
    out = score_against_test(camp, test)
    assert out["near_boundary_sigma_mae"] is not None
    assert out["boundary_mae"] == out["near_boundary_sigma_mae"]
