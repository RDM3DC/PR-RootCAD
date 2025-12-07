import math

import numpy as np

from adaptivecad.pi.kernel import PiAParams, adaptive_arc_length, make_adaptive_circle, pi_a


def test_pi_reduces_to_plain_when_kappa_zero():
    params = PiAParams(beta=0.5, s0=1.0, clamp=0.3)
    assert abs(pi_a(0.0, 1.0, params) - math.pi) < 1e-12


def test_clamp_limits_fractional_change():
    params = PiAParams(beta=10.0, s0=1.0, clamp=0.1)
    pa = pi_a(kappa=100.0, scale=10.0, params=params)
    frac = abs(pa / math.pi - 1.0)
    assert frac <= 0.1000001


def test_adaptive_arc_length_scales_linearly():
    params = PiAParams(beta=0.25, s0=1.0, clamp=0.3)
    r = 2.0
    ang = math.pi / 3
    L0 = ang * r
    L1 = adaptive_arc_length(r, ang, kappa=0.5, scale=1.0, params=params)
    # not equal unless kappa==0; positive scaling expected
    assert L1 > 0.9 * L0 and L1 < 1.5 * L0


def test_make_circle_shape():
    pts = make_adaptive_circle(radius=1.0, n=128, kappa=0.0, scale=1.0)
    assert pts.shape == (128, 2)
    # approximate radius from mean distance
    r_est = np.mean(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2))
    assert abs(r_est - 1.0) < 1e-2
