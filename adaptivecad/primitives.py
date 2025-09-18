from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeTorus,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeRevol,
)
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
import math

def make_ball(center, radius):
    c = gp_Pnt(*center)
    return BRepPrimAPI_MakeSphere(c, radius).Shape()

def make_torus(center, major_radius, minor_radius, axis=(0,0,1)):
    c = gp_Pnt(*center)
    ax = gp_Ax2(c, gp_Dir(*axis))
    return BRepPrimAPI_MakeTorus(ax, major_radius, minor_radius).Shape()

def make_cone(center, radius1, radius2, height, axis=(0,0,1)):
    """Create a truncated cone shape."""
    c = gp_Pnt(*center)
    ax = gp_Ax2(c, gp_Dir(*axis))
    return BRepPrimAPI_MakeCone(ax, radius1, radius2, height).Shape()

def make_revolve(profile, axis_point, axis_dir=(0,0,1), angle_deg=360):
    """Revolve a profile edge/wire about an axis to create a solid."""
    ax = gp_Ax2(gp_Pnt(*axis_point), gp_Dir(*axis_dir))
    angle_rad = math.radians(angle_deg)
    return BRepPrimAPI_MakeRevol(profile, ax, angle_rad).Shape()


def make_mobius(center, major_radius=30.0, width=8.0, twists=1, nu=160, nv=16):
    """Create an approximated Möbius strip as a compound of triangular faces.

    The Möbius parameterization used is:
      X = (R + v * cos(u/2)) * cos(u)
      Y = (R + v * cos(u/2)) * sin(u)
      Z = v * sin(u/2)

    where u in [0, 2*pi), v in [-w/2, w/2].
    """
    cx, cy, cz = center
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    import math

    def pt(u, v):
        x = (major_radius + v * math.cos(u / 2.0)) * math.cos(u)
        y = (major_radius + v * math.cos(u / 2.0)) * math.sin(u)
        z = v * math.sin(u / 2.0)
        return gp_Pnt(cx + x, cy + y, cz + z)

    for i in range(nu):
        u0 = 2.0 * math.pi * i / nu
        u1 = 2.0 * math.pi * ((i + 1) % nu) / nu
        for j in range(nv - 1):
            v0 = -width * 0.5 + width * j / (nv - 1)
            v1 = -width * 0.5 + width * (j + 1) / (nv - 1)

            p00 = pt(u0, v0)
            p01 = pt(u0, v1)
            p10 = pt(u1, v0)
            p11 = pt(u1, v1)

            # first triangle p00, p10, p11
            e1 = BRepBuilderAPI_MakeEdge(p00, p10).Edge()
            e2 = BRepBuilderAPI_MakeEdge(p10, p11).Edge()
            e3 = BRepBuilderAPI_MakeEdge(p11, p00).Edge()
            wire = BRepBuilderAPI_MakeWire()
            wire.Add(e1); wire.Add(e2); wire.Add(e3)
            try:
                f1 = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
                builder.Add(comp, f1)
            except Exception:
                pass

            # second triangle p00, p11, p01
            e1 = BRepBuilderAPI_MakeEdge(p00, p11).Edge()
            e2 = BRepBuilderAPI_MakeEdge(p11, p01).Edge()
            e3 = BRepBuilderAPI_MakeEdge(p01, p00).Edge()
            wire = BRepBuilderAPI_MakeWire()
            wire.Add(e1); wire.Add(e2); wire.Add(e3)
            try:
                f2 = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
                builder.Add(comp, f2)
            except Exception:
                pass

    return comp
