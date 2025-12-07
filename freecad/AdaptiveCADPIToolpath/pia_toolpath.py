# FreeCAD macro to generate πₐ-adaptive "circle" and Euclidean reference
import math
import FreeCAD as App, FreeCADGui as Gui
import Part


def trC_xy(x, y):
    g1 = math.exp(-((x - 15) ** 2 + (y - 5) ** 2) / (2 * 8**2))
    g2 = -0.8 * math.exp(-((x + 10) ** 2 + (y + 12) ** 2) / (2 * 10**2))
    return 0.6 * g1 + 0.6 * g2


def generate_points(R0=30.0, N=720, lam_field=0.04):
    pi = math.pi
    ds = 2 * pi * R0 / N
    theta = 0.0
    pts = []
    for k in range(N):
        x = R0 * math.cos(theta)
        y = R0 * math.sin(theta)
        pts.append((x, y))
        pia_loc = pi * (1 + lam_field * trC_xy(x, y))
        dtheta = (pi / pia_loc) * (ds / R0)
        theta += dtheta
    pts.append(pts[0])
    return pts


def makeWire(name, pts):
    vecs = [App.Vector(x, y, 0) for (x, y) in pts]
    w = Part.makePolygon(vecs)
    obj = App.ActiveDocument.addObject("Part::Feature", name)
    obj.Shape = w
    App.ActiveDocument.recompute()
    return obj


def run(radius=30.0, steps=720, lam=0.04):
    pts = generate_points(radius, steps, lam)
    makeWire("piA_Path", pts)
    ref = [
        (radius * math.cos(2 * math.pi * i / 256), radius * math.sin(2 * math.pi * i / 256))
        for i in range(257)
    ]
    makeWire("Euclid_Circle", ref)


if __name__ == "__main__":
    if App.ActiveDocument is None:
        App.newDocument()
    run()
    Gui.SendMsgToActiveView("ViewFit")
