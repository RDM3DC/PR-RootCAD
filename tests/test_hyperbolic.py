import importlib
import math

from adaptivecad.geom import full_turn_deg, pi_a_over_pi, rotate_cmd


def test_full_turn_deg():
    r = 0.4
    kappa = 1.0
    ratio = pi_a_over_pi(r, kappa)
    # expected ratio from the table is about 1.0269
    assert math.isclose(ratio, 1.0268808145, rel_tol=1e-6)
    d_full = full_turn_deg(r, kappa)
    assert math.isclose(d_full, 360.0 * ratio, rel_tol=1e-6)


def test_rotate_cmd_full_turn():
    r = 0.5
    kappa = 1.0
    d_full = full_turn_deg(r, kappa)
    rad = rotate_cmd(d_full, r, kappa)
    assert math.isclose(rad, 2 * math.pi, rel_tol=1e-6)


def test_pi_a_over_pi_euclidean_limit():
    assert math.isclose(pi_a_over_pi(0.0, 2.0), 1.0)
    assert math.isclose(pi_a_over_pi(2.0, 0.0), 1.0)


def test_pi_a_over_pi_large_ratio_fallback():
    assert math.isclose(pi_a_over_pi(800.0, 1.0), 1.0)


def test_math_sinh_not_monkeypatched():
    import adaptivecad.geom.hyperbolic as hyperbolic

    original = math.sinh
    importlib.reload(hyperbolic)
    assert math.sinh is original
    assert math.isclose(hyperbolic._stable_sinh(0.5), original(0.5))
