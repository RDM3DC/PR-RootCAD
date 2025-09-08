import os, sys, numpy as np

# Allow running directly from repo root without installing adaptivecad
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from adaptivecad.aacore.sdf import Scene, Prim
from adaptivecad.aacore.math import Xform
from adaptivecad.aacore.extract.marching_export import export_isosurface_to_stl

if __name__ == "__main__":
    sc = Scene(); sc.global_beta = 0.12
    sc.add(Prim('sphere', [0,0,0,1.0], Xform.translate(-0.6,0,0), beta=0.05, pid=1))
    sc.add(Prim('torus',  [1.2,0.35],  Xform.translate(0.9,0.0,0.0),             pid=2))
    sc.add(Prim('sphere', [0,0,0,0.7], Xform.translate(0.3,0.2,0.0), beta=0.00, pid=3, op='subtract'))
    path = export_isosurface_to_stl(sc, "aacore_demo.stl", bbox=((-2,-2,-2),(2,2,2)), res=64)
    print("Wrote:", path)
