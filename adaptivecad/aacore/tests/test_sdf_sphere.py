from adaptivecad.aacore.math import Xform
from adaptivecad.aacore.sdf import Prim, Scene


def test_sphere_sdf_signs():
    sc = Scene()
    sc.add(Prim("sphere", [0, 0, 0, 1.0], Xform(), beta=0.0, pid=1))
    d0, _, _ = sc.sdf((0.0, 0.0, 0.0))
    d1, _, _ = sc.sdf((2.0, 0.0, 0.0))
    assert d0 < 0.0 and d1 > 0.0
