
# freecad/AdaptiveCADPIToolpath/commands/CommandAnisoDistance.py
import math, json
try:
    import FreeCAD as App, FreeCADGui as Gui
except Exception:
    App = None; Gui = None

from adaptive_pi.aniso_fmm import anisotropic_fmm, trace_geodesic, metric_const_aniso

class _CmdAnisoDistance:
    def GetResources(self):
        return {
            "Pixmap": "",
            "MenuText": "Anisotropic Distance (FMM-lite)",
            "ToolTip": "Compute an anisotropic distance map on a small grid and report sample values",
        }
    def Activated(self):
        if Gui is None:
            print("[AnisoDistance] FreeCAD GUI not available.")
            return
        try:
            from PySide6 import QtWidgets
        except Exception:
            from PyQt5 import QtWidgets  # type: ignore
        w = QtWidgets.QInputDialog
        a, ok = w.getDouble(None, "Metric a", "a (G11):", 1.3, 0.01, 100.0, 3)
        if not ok: return
        b, ok = w.getDouble(None, "Metric b", "b (G22):", 1.0, 0.01, 100.0, 3)
        if not ok: return
        nx = ny = 129
        G = metric_const_aniso(nx, ny, a=a, b=b)
        src = (nx//2, ny//2)
        T = anisotropic_fmm(G, src, use_diagonals=True)
        mid = src[1]
        xs = [0,5,10,20,30,40]
        row_x = [round(float(T[src[0]+k, mid]), 4) for k in xs]
        msg = f"[AnisoDistance] T(+x)={row_x}  (sqrt(a)~{math.sqrt(a):.4f}, sqrt(b)~{math.sqrt(b):.4f})\n"
        try:
            Gui.PrintMessage(msg + "\n")
        except Exception:
            print(msg)
    def IsActive(self): return True

GuiCommand = _CmdAnisoDistance
