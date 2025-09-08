import numpy as np
from adaptivecad.aacore.sdf import Scene, Prim, ray_march
from adaptivecad.aacore.math import Xform

def test_basic_sphere_hit():
    sc = Scene()
    sc.add(Prim('sphere', [0,0,0,1.0], pid=1))
    h = ray_march(sc, np.array([0,0,3.0]), np.array([0,0,-1.0]))
    assert h is not None
    assert h.id == 1
    assert abs(np.linalg.norm(h.p) - 1.0) < 1e-2

def test_difference():
    sc = Scene()
    sc.add(Prim('sphere', [0,0,0,1.0], pid=1))
    # subtract smaller sphere offset slightly
    sc.add(Prim('sphere', [0,0,0,0.5], pid=2, op='subtract'))
    h = ray_march(sc, np.array([0,0,3.0]), np.array([0,0,-1.0]))
    assert h is not None
