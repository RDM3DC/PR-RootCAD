from adaptivecad.aacore import math as m


def test_vec_norm_unit():
    v = m.Vec3.of(3, 4, 0)
    assert abs(v.norm() - 5.0) < 1e-12
    u = v.unit()
    assert abs(u.norm() - 1.0) < 1e-12


def test_interval_basic():
    I = m.Interval(3, 1)
    assert I.lo == 1 and I.hi == 3
    assert I.contains(2)
    assert I.intersect(m.Interval(2, 4)).lo == 2
