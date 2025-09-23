"""Analytic SDF OpenGL viewport & control panel (PySide6 version).

Refactored to standardize on PySide6 (no PyQt6 mixing) and allow
injection of a shared AACore analytic scene so multiple view components
can operate on the same underlying primitive set.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider,
    QComboBox, QCheckBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QScrollArea, QSizePolicy
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat, QMouseEvent, QWheelEvent, QPainter, QPen, QColor
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QGuiApplication
from OpenGL.GL import *
import numpy as np
from pathlib import Path
from adaptivecad.sketch.model import SketchDocument
from adaptivecad.sketch.units import Units
from adaptivecad.aacore.sdf import (
    Scene as AACoreScene, Prim, KIND_SPHERE, KIND_BOX, KIND_CAPSULE, KIND_TORUS,
    KIND_MOBIUS, KIND_SUPERELLIPSOID, KIND_QUASICRYSTAL, KIND_TORUS4D, KIND_MANDELBULB, KIND_KLEIN, KIND_MENGER, KIND_HYPERBOLIC, KIND_GYROID, KIND_TREFOIL, OP_SOLID, OP_SUBTRACT, MAX_PRIMS
)
import time, os
import json
import logging
log = logging.getLogger("adaptivecad.gui")
try:
    from PIL import Image  # type: ignore
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

# Optional radial tool wheel overlay (used as a HUD over the viewport)
try:
    from adaptivecad.ui.radial_tool_wheel import RadialToolWheelOverlay, ToolSpec
    _HAVE_WHEEL = True
except Exception:
    RadialToolWheelOverlay = None  # type: ignore
    ToolSpec = None  # type: ignore
    _HAVE_WHEEL = False

    # --- 2D Sketch Overlay (Polyline + Snaps) ---------------------------------
    SNAP_PX = 10  # pixel radius for snapping

    class SketchEntity:
        def __init__(self):
            self.construction = False
        def draw(self, p: QPainter, view): ...
        def snap_points(self): return []  # list of (x,y) in world (Z=0)
        def snap_points_typed(self):
            return [(pt, 'other') for pt in self.snap_points()]

    class Polyline2D(SketchEntity):
        def __init__(self, pts=None):
            import numpy as _np
            super().__init__()
            self.pts = [ _np.array(p, dtype=_np.float32) for p in (pts or []) ]  # (x,y)
            self.closed = False
        def snap_points(self):
            import numpy as _np
            out = []
            out += self.pts
            for a,b in zip(self.pts[:-1], self.pts[1:]):
                out.append( (a+b)/2.0 )
            if self.closed and len(self.pts) >= 2:
                a = self.pts[-1]; b = self.pts[0]
                out.append((a+b)/2.0)
            return out
        def snap_points_typed(self):
            out=[]
            if self.pts:
                out.append((self.pts[0],'end'))
                if len(self.pts)>1:
                    out.append((self.pts[-1],'end'))
            for a,b in zip(self.pts[:-1], self.pts[1:]):
                out.append(((a+b)/2.0, 'mid'))
            if self.closed and len(self.pts)>=2:
                a=self.pts[-1]; b=self.pts[0]
                out.append(((a+b)/2.0, 'mid'))
            return out
        def draw(self, p: QPainter, view):
            import numpy as _np
            if not self.pts:
                return
            pts2 = [ view._world_to_screen(_np.array([x,y,0.0], _np.float32)) for (x,y) in self.pts ]
            if len(pts2) >= 2:
                pen = QPen(QColor(220,220,220)); pen.setWidth(2)
                if getattr(self,'construction',False): pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                for a,b in zip(pts2[:-1], pts2[1:]):
                    p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
                if self.closed:
                    a=pts2[-1]; b=pts2[0]
                    p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            pen = QPen(QColor(255,200,120)); pen.setWidth(6); p.setPen(pen)
            for s in pts2: p.drawPoint(int(s[0]), int(s[1]))

    class Line2D(SketchEntity):
        def __init__(self, p0, p1=None):
            import numpy as _np
            super().__init__()
            self.p0 = _np.array(p0, dtype=_np.float32)
            self.p1 = _np.array(p1, dtype=_np.float32) if p1 is not None else None
        def snap_points(self):
            import numpy as _np
            pts = [self.p0]
            if self.p1 is not None:
                pts.append(self.p1)
                pts.append((self.p0 + self.p1)/2.0)
            return pts
        def draw(self, p: QPainter, view):
            import numpy as _np
            s0 = view._world_to_screen(_np.array([self.p0[0], self.p0[1], 0.0], _np.float32))
            pen = QPen(QColor(220,220,220)); pen.setWidth(2)
            if getattr(self,'construction',False): pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            if self.p1 is not None:
                s1 = view._world_to_screen(_np.array([self.p1[0], self.p1[1], 0.0], _np.float32))
                p.drawLine(int(s0[0]), int(s0[1]), int(s1[0]), int(s1[1]))
            pen = QPen(QColor(255,200,140)); pen.setWidth(6); p.setPen(pen); p.drawPoint(int(s0[0]), int(s0[1]))
            if self.p1 is not None:
                p.drawPoint(int(s1[0]), int(s1[1]))

    class Circle2D(SketchEntity):
        def __init__(self, center, radius=0.0):
            import numpy as _np
            super().__init__()
            self.center = _np.array(center, dtype=_np.float32)
            self.radius = float(radius)
        def snap_points(self):
            import numpy as _np, math as _m
            pts = [self.center]
            if self.radius > 1e-6:
                for ang in (0,90,180,270):
                    a = _m.radians(ang)
                    pts.append(self.center + self.radius * _np.array([_np.cos(a), _np.sin(a)], _np.float32))
            return pts
        def draw(self, p: QPainter, view):
            import numpy as _np, math as _m
            if self.radius <= 0: return
            segs = 40
            pts = []
            for i in range(segs+1):
                a = 2*_m.pi * i / segs
                w = self.center + self.radius * _np.array([_np.cos(a), _np.sin(a)], _np.float32)
                s = view._world_to_screen(_np.array([w[0], w[1], 0.0], _np.float32))
                pts.append(s)
            pen = QPen(QColor(180,220,255)); pen.setWidth(2)
            if getattr(self,'construction',False): pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            for a,b in zip(pts[:-1], pts[1:]):
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            # center point
            c = view._world_to_screen(_np.array([self.center[0], self.center[1], 0.0], _np.float32))
            pen = QPen(QColor(255,180,120)); pen.setWidth(6); p.setPen(pen); p.drawPoint(int(c[0]), int(c[1]))

    class Rect2D(SketchEntity):
        def __init__(self, p0, p1=None):
            import numpy as _np
            super().__init__()
            self.p0 = _np.array(p0, dtype=_np.float32)
            self.p1 = _np.array(p1, dtype=_np.float32) if p1 is not None else None
        def corners(self):
            import numpy as _np
            if self.p1 is None:
                return []
            x0,y0 = self.p0
            x1,y1 = self.p1
            mnx,mxx = min(x0,x1), max(x0,x1)
            mny,mxy = min(y0,y1), max(y0,y1)
            return [ _np.array([mnx,mny], _np.float32), _np.array([mxx,mny], _np.float32), _np.array([mxx,mxy], _np.float32), _np.array([mnx,mxy], _np.float32) ]
        def snap_points(self):
            import numpy as _np
            cs = self.corners()
            pts = cs[:]
            if len(cs)==4:
                # edge midpoints
                for a,b in zip(cs, cs[1:]+cs[:1]):
                    pts.append( (a+b)/2.0 )
                # center
                cx = sum(c[0] for c in cs)/4.0; cy = sum(c[1] for c in cs)/4.0
                pts.append(_np.array([cx,cy], _np.float32))
            return pts
        def draw(self, p: QPainter, view):
            import numpy as _np
            cs = self.corners()
            if len(cs) < 2:
                # just draw p0
                s0 = view._world_to_screen(_np.array([self.p0[0], self.p0[1], 0.0], _np.float32))
                pen = QPen(QColor(255,200,140)); pen.setWidth(6); p.setPen(pen); p.drawPoint(int(s0[0]), int(s0[1]))
                return
            pts2 = [ view._world_to_screen(_np.array([c[0], c[1], 0.0], _np.float32)) for c in cs ]
            pen = QPen(QColor(200,255,180)); pen.setWidth(2)
            if getattr(self,'construction',False): pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            for a,b in zip(pts2, pts2[1:]+pts2[:1]):
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            pen = QPen(QColor(255,200,140)); pen.setWidth(6); p.setPen(pen)
            for s in pts2:
                p.drawPoint(int(s[0]), int(s[1]))

    class Arc2D(SketchEntity):
        def __init__(self, p0=None, p1=None, p2=None):
            import numpy as _np
            super().__init__()
            self.p0 = _np.array(p0, dtype=_np.float32) if p0 is not None else None
            self.p1 = _np.array(p1, dtype=_np.float32) if p1 is not None else None
            self.p2 = _np.array(p2, dtype=_np.float32) if p2 is not None else None
        def _circle_from_3pts(self):
            import numpy as _np
            if self.p0 is None or self.p1 is None or self.p2 is None:
                return None
            x1,y1 = self.p0; x2,y2 = self.p1; x3,y3 = self.p2
            a = x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2
            if abs(a) < 1e-9: return None
            b = (x1*x1 + y1*y1)*(y3 - y2) + (x2*x2 + y2*y2)*(y1 - y3) + (x3*x3 + y3*y3)*(y2 - y1)
            c = (x1*x1 + y1*y1)*(x2 - x3) + (x2*x2 + y2*y2)*(x3 - x1) + (x3*x3 + y3*y3)*(x1 - x2)
            cx = -b/(2*a); cy = -c/(2*a)
            r = float(_np.hypot(x1-cx, y1-cy))
            return _np.array([cx,cy], _np.float32), r
        def snap_points(self):
            import numpy as _np
            pts=[]
            if self.p0 is not None: pts.append(self.p0)
            if self.p2 is not None: pts.append(self.p2)
            circ = self._circle_from_3pts()
            if circ is not None:
                c,_ = circ; pts.append(c)
            if self.p0 is not None and self.p2 is not None:
                pts.append((self.p0 + self.p2)/2.0)
            return pts
        def draw(self, p: QPainter, view):
            import numpy as _np, math as _m
            if self.p0 is None:
                return
            pen=QPen(QColor(200,240,255)); pen.setWidth(2)
            if getattr(self,'construction',False): pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            if self.p1 is None or self.p2 is None:
                s0=view._world_to_screen(_np.array([self.p0[0], self.p0[1],0.0], _np.float32))
                p.drawPoint(int(s0[0]), int(s0[1]))
                if self.p1 is not None:
                    s1=view._world_to_screen(_np.array([self.p1[0], self.p1[1],0.0], _np.float32))
                    p.drawLine(int(s0[0]), int(s0[1]), int(s1[0]), int(s1[1]))
                return
            circ=self._circle_from_3pts()
            if circ is None:
                s0=view._world_to_screen(_np.array([self.p0[0], self.p0[1],0.0], _np.float32))
                s2=view._world_to_screen(_np.array([self.p2[0], self.p2[1],0.0], _np.float32))
                p.drawLine(int(s0[0]), int(s0[1]), int(s2[0]), int(s2[1]))
                return
            c,r=circ
            a0=_m.atan2(self.p0[1]-c[1], self.p0[0]-c[0])
            a1=_m.atan2(self.p1[1]-c[1], self.p1[0]-c[0])
            a2=_m.atan2(self.p2[1]-c[1], self.p2[0]-c[0])
            def ang_dist(a,b):
                d=(b-a+_m.pi)%(2*_m.pi)-_m.pi
                return d
            d1=abs(ang_dist(a0,a1))+abs(ang_dist(a1,a2))
            d2=abs(ang_dist(a0,a2))
            cw=False if d1<d2 else True
            segs=48; pts=[]
            if not cw:
                step=ang_dist(a0,a2)/segs; k=0
                while k<=segs:
                    ang=a0+step*k; k+=1
                    w=c+r*_np.array([_np.cos(ang), _np.sin(ang)], _np.float32)
                    s=view._world_to_screen(_np.array([w[0],w[1],0.0], _np.float32)); pts.append(s)
            else:
                step=ang_dist(a2,a0)/segs; k=0
                while k<=segs:
                    ang=a2+step*k; k+=1
                    w=c+r*_np.array([_np.cos(ang), _np.sin(ang)], _np.float32)
                    s=view._world_to_screen(_np.array([w[0],w[1],0.0], _np.float32)); pts.append(s)
            for a,b in zip(pts[:-1], pts[1:]): p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            pen_pts=QPen(QColor(255,200,140)); pen_pts.setWidth(6); p.setPen(pen_pts)
            s0=view._world_to_screen(_np.array([self.p0[0], self.p0[1],0.0], _np.float32)); p.drawPoint(int(s0[0]), int(s0[1]))
            s2=view._world_to_screen(_np.array([self.p2[0], self.p2[1],0.0], _np.float32)); p.drawPoint(int(s2[0]), int(s2[1]))
            sc=view._world_to_screen(_np.array([c[0], c[1],0.0], _np.float32)); p.drawPoint(int(sc[0]), int(sc[1]))

    class Line2D(SketchEntity):
        def __init__(self, p0, p1=None):
            import numpy as _np
            super().__init__()
            self.p0 = _np.array(p0, dtype=_np.float32)
            self.p1 = _np.array(p1, dtype=_np.float32) if p1 is not None else None
        def snap_points(self):
            import numpy as _np
            pts = [self.p0]
            if self.p1 is not None:
                pts.append(self.p1)
                pts.append((self.p0 + self.p1)/2.0)
            return pts
        def snap_points_typed(self):
            out=[(self.p0,'end')]
            if self.p1 is not None:
                out.append((self.p1,'end'))
                out.append(((self.p0+self.p1)/2.0,'mid'))
            return out
        def draw(self, p: QPainter, view):
            import numpy as _np
            s0 = view._world_to_screen(_np.array([self.p0[0], self.p0[1], 0.0], _np.float32))
            pen = QPen(QColor(220,220,220)); pen.setWidth(2)
            if getattr(self,'construction',False): pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            if self.p1 is not None:
                s1 = view._world_to_screen(_np.array([self.p1[0], self.p1[1], 0.0], _np.float32))
                p.drawLine(int(s0[0]), int(s0[1]), int(s1[0]), int(s1[1]))
            pen = QPen(QColor(255,200,140)); pen.setWidth(6); p.setPen(pen); p.drawPoint(int(s0[0]), int(s0[1]))
            if self.p1 is not None:
                p.drawPoint(int(s1[0]), int(s1[1]))

    class Arc2D(SketchEntity):
        def __init__(self, p0=None, p1=None, p2=None):
            import numpy as _np
            super().__init__()
            self.p0 = _np.array(p0, dtype=_np.float32) if p0 is not None else None
            self.p1 = _np.array(p1, dtype=_np.float32) if p1 is not None else None
            self.p2 = _np.array(p2, dtype=_np.float32) if p2 is not None else None
        def _circle_from_3pts(self):
            import numpy as _np
            if self.p0 is None or self.p1 is None or self.p2 is None:
                return None
            x1,y1 = self.p0; x2,y2 = self.p1; x3,y3 = self.p2
            a = x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2
            if abs(a) < 1e-9: return None
            b = (x1*x1 + y1*y1)*(y3 - y2) + (x2*x2 + y2*y2)*(y1 - y3) + (x3*x3 + y3*y3)*(y2 - y1)
            c = (x1*x1 + y1*y1)*(x2 - x3) + (x2*x2 + y2*y2)*(x3 - x1) + (x3*x3 + y3*y3)*(x1 - x2)
            cx = -b/(2*a); cy = -c/(2*a)
            r = float(_np.hypot(x1-cx, y1-cy))
            return _np.array([cx,cy], _np.float32), r
        def snap_points(self):
            import numpy as _np
            pts=[]
            if self.p0 is not None: pts.append(self.p0)
            if self.p2 is not None: pts.append(self.p2)
            circ = self._circle_from_3pts()
            if circ is not None:
                c,_ = circ; pts.append(c)
            if self.p0 is not None and self.p2 is not None:
                pts.append((self.p0 + self.p2)/2.0)
            return pts
        def snap_points_typed(self):
            out=[]
            if self.p0 is not None: out.append((self.p0,'end'))
            if self.p2 is not None: out.append((self.p2,'end'))
            circ = self._circle_from_3pts()
            if circ is not None:
                c,_ = circ; out.append((c,'center'))
            if self.p0 is not None and self.p2 is not None:
                out.append(((self.p0+self.p2)/2.0,'mid'))
            return out
        def draw(self, p: QPainter, view):
            import numpy as _np, math as _m
            if self.p0 is None:
                return
            pen = QPen(QColor(200,240,255)); pen.setWidth(2)
            if getattr(self,'construction',False): pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            # if not complete, draw provisional segment(s)
            if self.p1 is None or self.p2 is None:
                s0 = view._world_to_screen(_np.array([self.p0[0], self.p0[1], 0.0], _np.float32))
                p.drawPoint(int(s0[0]), int(s0[1]))
                if self.p1 is not None:
                    s1 = view._world_to_screen(_np.array([self.p1[0], self.p1[1], 0.0], _np.float32))
                    p.drawLine(int(s0[0]), int(s0[1]), int(s1[0]), int(s1[1]))
                return
            circ = self._circle_from_3pts()
            if circ is None:
                # fallback draw polyline
                s0 = view._world_to_screen(_np.array([self.p0[0], self.p0[1], 0.0], _np.float32))
                s2 = view._world_to_screen(_np.array([self.p2[0], self.p2[1], 0.0], _np.float32))
                p.drawLine(int(s0[0]), int(s0[1]), int(s2[0]), int(s2[1]))
                return
            c, r = circ
            # compute angles
            a0 = _m.atan2(self.p0[1]-c[1], self.p0[0]-c[0])
            a1 = _m.atan2(self.p1[1]-c[1], self.p1[0]-c[0])
            a2 = _m.atan2(self.p2[1]-c[1], self.p2[0]-c[0])
            # pick direction that passes near a1 (shorter arc through a1)
            def ang_dist(a,b):
                d = (b - a + _m.pi) % (2*_m.pi) - _m.pi
                return d
            d1 = abs(ang_dist(a0, a1)) + abs(ang_dist(a1, a2))
            d2 = abs(ang_dist(a0, a2))
            cw = False if d1 < d2 else True
            # sample arc
            segs = 48
            pts=[]
            if not cw:
                step = ang_dist(a0, a2)/segs
                k = 0
                while k <= segs:
                    ang = a0 + step*k; k += 1
                    w = c + r * _np.array([_np.cos(ang), _np.sin(ang)], _np.float32)
                    s = view._world_to_screen(_np.array([w[0], w[1], 0.0], _np.float32))
                    pts.append(s)
            else:
                # reverse direction
                step = ang_dist(a2, a0)/segs
                k = 0
                while k <= segs:
                    ang = a2 + step*k; k += 1
                    w = c + r * _np.array([_np.cos(ang), _np.sin(ang)], _np.float32)
                    s = view._world_to_screen(_np.array([w[0], w[1], 0.0], _np.float32))
                    pts.append(s)
            for a,b in zip(pts[:-1], pts[1:]):
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            # endpoints & optional center point
            pen_pts = QPen(QColor(255,200,140)); pen_pts.setWidth(6); p.setPen(pen_pts)
            s0 = view._world_to_screen(_np.array([self.p0[0], self.p0[1], 0.0], _np.float32)); p.drawPoint(int(s0[0]), int(s0[1]))
            s2 = view._world_to_screen(_np.array([self.p2[0], self.p2[1], 0.0], _np.float32)); p.drawPoint(int(s2[0]), int(s2[1]))
            sc = view._world_to_screen(_np.array([c[0], c[1], 0.0], _np.float32)); p.drawPoint(int(sc[0]), int(sc[1]))

    class SketchLayer:
        def __init__(self):
            self.entities: list[SketchEntity] = []
            self.active_poly: Polyline2D | None = None
            self.active_shape: SketchEntity | None = None  # circle/rect WIP
            self.hover_snap = None  # (x,y)
            self.dimensions = []  # list of DimOverlay
            self._dim_pick = None  # first ref for dimension tools
            self._dim_preview = None  # transient preview overlay
            self._dim_placing = False  # after picking A and B, placing offset
            self._dim_place_data = None  # {'type': str, 'a': ref, 'b': ref}
        def all_snap_points(self):
            pts = []
            for e in self.entities:
                pts += e.snap_points()
            if self.active_poly:
                pts += self.active_poly.snap_points()
            if self.active_shape and hasattr(self.active_shape, 'snap_points'):
                pts += self.active_shape.snap_points()
            return pts
        def all_snap_points_typed(self):
            pts = []
            for e in self.entities:
                if hasattr(e,'snap_points_typed'):
                    pts += e.snap_points_typed()
                else:
                    pts += [(pt,'other') for pt in e.snap_points()]
            if self.active_poly:
                pts += self.active_poly.snap_points_typed()
            if self.active_shape and hasattr(self.active_shape, 'snap_points_typed'):
                pts += self.active_shape.snap_points_typed()
            return pts
        def draw(self, p: QPainter, view):
            import numpy as _np
            for e in self.entities:
                e.draw(p, view)
            if self.active_poly:
                self.active_poly.draw(p, view)
            if self.active_shape:
                self.active_shape.draw(p, view)
            # draw dimensions
            for d in getattr(self, 'dimensions', []):
                try:
                    d.draw(p, view)
                except Exception:
                    pass
            # draw preview dim if any
            try:
                if self._dim_preview is not None:
                    self._dim_preview.draw(p, view)
            except Exception:
                pass
            if self.hover_snap is not None:
                s = view._world_to_screen(_np.array([self.hover_snap[0], self.hover_snap[1], 0.0], _np.float32))
                pen = QPen(QColor(120,255,160)); pen.setWidth(2); p.setPen(pen)
                p.drawLine(int(s[0]-6), int(s[1]), int(s[0]+6), int(s[1]))
                p.drawLine(int(s[0]), int(s[1]-6), int(s[0]), int(s[1]+6))

    # --- Dimension overlays ---
    class DimBase2D:
        def __init__(self):
            self.label_override = None  # when set, displayed instead of measured
        def _fmt(self, val: float, panel) -> str:
            try:
                prec = int(getattr(panel, '_dim_precision', 2))
            except Exception:
                prec = 2
            return f"{val:.{prec}f}"
        def draw_text(self, p: QPainter, view, pos_xy, text):
            import numpy as _np
            s = view._world_to_screen(_np.array([pos_xy[0], pos_xy[1], 0.0], _np.float32))
            p.setPen(QPen(QColor(250, 250, 210)))
            p.drawText(int(s[0]+6), int(s[1]-6), text)

    class DimLinear2D(DimBase2D):
        def __init__(self, ref_a, ref_b, panel, offset: float = 0.0):
            super().__init__(); self.ref_a = ref_a; self.ref_b = ref_b; self.panel = panel; self.offset = float(offset)
        def _xy(self, ref):
            if isinstance(ref, tuple) and len(ref)==2 and isinstance(ref[0], Polyline2D):
                ent, idx = ref; return ent.pts[idx]
            if isinstance(ref, tuple) and len(ref)==2 and isinstance(ref[0], Line2D):
                ent, idx = ref
                if idx == 0: return ent.p0
                if idx == 1 and ent.p1 is not None: return ent.p1
                return None
            if isinstance(ref, tuple) and len(ref)==2 and isinstance(ref[0], Rect2D):
                ent, idx = ref
                cs = ent.corners()
                if 0 <= idx < len(cs): return cs[idx]
                return None
            if isinstance(ref, tuple) and ref[0]=='circle' and isinstance(ref[1], Circle2D):
                return ref[1].center
            # allow raw points [[x,y]] or numpy arrays
            try:
                import numpy as _np
                if isinstance(ref, (list, tuple)) and len(ref)==2 and all(isinstance(v, (int,float)) for v in ref):
                    return _np.array([float(ref[0]), float(ref[1])], _np.float32)
                if hasattr(ref, 'shape') and getattr(ref, 'shape', None) is not None and len(ref.shape)==1 and ref.shape[0]==2:
                    return ref
            except Exception:
                pass
            return None
        def draw(self, p: QPainter, view):
            import numpy as _np
            a = self._xy(self.ref_a); b = self._xy(self.ref_b)
            if a is None or b is None: return
            # baseline and normal
            v = b - a
            L = float(np.linalg.norm(v))
            if L < 1e-6: return
            n = _np.array([-v[1], v[0]], _np.float32) / L
            # shifted dimension line
            p0 = a + n * self.offset
            p1 = b + n * self.offset
            sa = view._world_to_screen(_np.array([p0[0], p0[1], 0.0], _np.float32))
            sb = view._world_to_screen(_np.array([p1[0], p1[1], 0.0], _np.float32))
            # extension lines from anchors to dim line
            ea0 = view._world_to_screen(_np.array([a[0], a[1], 0.0], _np.float32))
            eb0 = view._world_to_screen(_np.array([b[0], b[1], 0.0], _np.float32))
            pen = QPen(QColor(180, 200, 220)); pen.setWidth(1); p.setPen(pen)
            p.drawLine(int(ea0[0]), int(ea0[1]), int(sa[0]), int(sa[1]))
            p.drawLine(int(eb0[0]), int(eb0[1]), int(sb[0]), int(sb[1]))
            # main dimension line
            pen = QPen(QColor(255, 220, 180)); pen.setWidth(2); p.setPen(pen)
            p.drawLine(int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1]))
            # label
            val = float(np.linalg.norm(b - a))
            text = self.label_override if self.label_override is not None else self._fmt(val, self.panel)
            mid = (p0 + p1) * 0.5
            self.draw_text(p, view, mid, text)

    class DimHorizontal2D(DimLinear2D):
        def draw(self, p: QPainter, view):
            import numpy as _np
            a = self._xy(self.ref_a); b = self._xy(self.ref_b)
            if a is None or b is None: return
            # normal for horizontal is +Y
            n = _np.array([0.0, 1.0], _np.float32)
            p0 = _np.array([a[0], a[1]], _np.float32) + n * self.offset
            p1 = _np.array([b[0], b[1]], _np.float32) + n * self.offset
            sa = view._world_to_screen(_np.array([p0[0], p0[1], 0.0], _np.float32))
            sb = view._world_to_screen(_np.array([p1[0], p1[1], 0.0], _np.float32))
            # extensions
            ea0 = view._world_to_screen(_np.array([a[0], a[1], 0.0], _np.float32))
            eb0 = view._world_to_screen(_np.array([b[0], b[1], 0.0], _np.float32))
            pen = QPen(QColor(180, 200, 220)); pen.setWidth(1); p.setPen(pen)
            p.drawLine(int(ea0[0]), int(ea0[1]), int(sa[0]), int(sa[1]))
            p.drawLine(int(eb0[0]), int(eb0[1]), int(sb[0]), int(sb[1]))
            pen = QPen(QColor(255, 220, 180)); pen.setWidth(2); p.setPen(pen)
            p.drawLine(int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1]))
            val = abs(float(b[0] - a[0]))
            text = self.label_override if self.label_override is not None else self._fmt(val, self.panel)
            mid = (p0 + p1) * 0.5
            self.draw_text(p, view, mid, text)

    class DimVertical2D(DimLinear2D):
        def draw(self, p: QPainter, view):
            import numpy as _np
            a = self._xy(self.ref_a); b = self._xy(self.ref_b)
            if a is None or b is None: return
            # normal for vertical is +X
            n = _np.array([1.0, 0.0], _np.float32)
            p0 = _np.array([a[0], a[1]], _np.float32) + n * self.offset
            p1 = _np.array([b[0], b[1]], _np.float32) + n * self.offset
            sa = view._world_to_screen(_np.array([p0[0], p0[1], 0.0], _np.float32))
            sb = view._world_to_screen(_np.array([p1[0], p1[1], 0.0], _np.float32))
            # extensions
            ea0 = view._world_to_screen(_np.array([a[0], a[1], 0.0], _np.float32))
            eb0 = view._world_to_screen(_np.array([b[0], b[1], 0.0], _np.float32))
            pen = QPen(QColor(180, 200, 220)); pen.setWidth(1); p.setPen(pen)
            p.drawLine(int(ea0[0]), int(ea0[1]), int(sa[0]), int(sa[1]))
            p.drawLine(int(eb0[0]), int(eb0[1]), int(sb[0]), int(sb[1]))
            pen = QPen(QColor(255, 220, 180)); pen.setWidth(2); p.setPen(pen)
            p.drawLine(int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1]))
            val = abs(float(b[1] - a[1]))
            text = self.label_override if self.label_override is not None else self._fmt(val, self.panel)
            mid = (p0 + p1) * 0.5
            self.draw_text(p, view, mid, text)

    class DimRadius2D(DimBase2D):
        def __init__(self, circle_ref, panel):
            super().__init__(); self.circle_ref = circle_ref; self.panel = panel
        def draw(self, p: QPainter, view):
            if not (isinstance(self.circle_ref, tuple) and self.circle_ref[0]=='circle' and isinstance(self.circle_ref[1], Circle2D)):
                return
            c = self.circle_ref[1]
            val = float(c.radius)
            text = self.label_override if self.label_override is not None else self._fmt(val, self.panel)
            # place text slightly above center
            pos = c.center + np.array([0.0, c.radius + 0.0], np.float32)
            pen = QPen(QColor(255, 220, 180)); pen.setWidth(1); p.setPen(pen)
            self.draw_text(p, view, pos, f"R {text}")

    class DimDiameter2D(DimRadius2D):
        def draw(self, p: QPainter, view):
            if not (isinstance(self.circle_ref, tuple) and self.circle_ref[0]=='circle' and isinstance(self.circle_ref[1], Circle2D)):
                return
            c = self.circle_ref[1]
            val = float(c.radius*2.0)
            text = self.label_override if self.label_override is not None else self._fmt(val, self.panel)
            pos = c.center + np.array([0.0, c.radius + 0.0], np.float32)
            pen = QPen(QColor(255, 220, 180)); pen.setWidth(1); p.setPen(pen)
            self.draw_text(p, view, pos, f"⌀ {text}")

    class Circle2D(SketchEntity):
        def __init__(self, center, radius=0.0):
            import numpy as _np
            super().__init__()
            self.center = _np.array(center, dtype=_np.float32)
            self.radius = float(radius)
        def snap_points(self):
            import numpy as _np, math as _m
            pts = [self.center]
            if self.radius > 1e-6:
                for ang in (0,90,180,270):
                    a = _m.radians(ang)
                    pts.append(self.center + self.radius * _np.array([_np.cos(a), _np.sin(a)], _np.float32))
            return pts
        def snap_points_typed(self):
            out=[(self.center,'center')]
            return out
        def draw(self, p: QPainter, view):
            import numpy as _np, math as _m
            if self.radius <= 0: return
            segs = 40
            pts = []
            for i in range(segs+1):
                a = 2*_m.pi * i / segs
                w = self.center + self.radius * _np.array([_np.cos(a), _np.sin(a)], _np.float32)
                s = view._world_to_screen(_np.array([w[0], w[1], 0.0], _np.float32))
                pts.append(s)
            pen = QPen(QColor(180,220,255)); pen.setWidth(2)
            if getattr(self,'construction',False): pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            for a,b in zip(pts[:-1], pts[1:]):
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            # center point
            c = view._world_to_screen(_np.array([self.center[0], self.center[1], 0.0], _np.float32))
            pen = QPen(QColor(255,180,120)); pen.setWidth(6); p.setPen(pen); p.drawPoint(int(c[0]), int(c[1]))

    class Rect2D(SketchEntity):
        def __init__(self, p0, p1=None):
            import numpy as _np
            super().__init__()
            self.p0 = _np.array(p0, dtype=_np.float32)
            self.p1 = _np.array(p1, dtype=_np.float32) if p1 is not None else None
        def corners(self):
            import numpy as _np
            if self.p1 is None:
                return []
            x0,y0 = self.p0
            x1,y1 = self.p1
            mnx,mxx = min(x0,x1), max(x0,x1)
            mny,mxy = min(y0,y1), max(y0,y1)
            return [ _np.array([mnx,mny], _np.float32), _np.array([mxx,mny], _np.float32), _np.array([mxx,mxy], _np.float32), _np.array([mnx,mxy], _np.float32) ]
        def snap_points(self):
            import numpy as _np
            cs = self.corners()
            pts = cs[:]
            if len(cs)==4:
                # edge midpoints
                for a,b in zip(cs, cs[1:]+cs[:1]):
                    pts.append( (a+b)/2.0 )
                # center
                cx = sum(c[0] for c in cs)/4.0; cy = sum(c[1] for c in cs)/4.0
                pts.append(_np.array([cx,cy], _np.float32))
            return pts
        def snap_points_typed(self):
            import numpy as _np
            cs = self.corners()
            out=[(c,'end') for c in cs]
            if len(cs)==4:
                for a,b in zip(cs, cs[1:]+cs[:1]): out.append(((a+b)/2.0,'mid'))
                cx = sum(c[0] for c in cs)/4.0; cy = sum(c[1] for c in cs)/4.0
                out.append((_np.array([cx,cy], _np.float32),'center'))
            return out
        def draw(self, p: QPainter, view):
            import numpy as _np
            cs = self.corners()
            if len(cs) < 2:
                # just draw p0
                s0 = view._world_to_screen(_np.array([self.p0[0], self.p0[1], 0.0], _np.float32))
                pen = QPen(QColor(255,200,140)); pen.setWidth(6); p.setPen(pen); p.drawPoint(int(s0[0]), int(s0[1]))
                return
            pts2 = [ view._world_to_screen(_np.array([c[0], c[1], 0.0], _np.float32)) for c in cs ]
            pen = QPen(QColor(200,255,180)); pen.setWidth(2)
            if getattr(self,'construction',False): pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            for a,b in zip(pts2, pts2[1:]+pts2[:1]):
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            pen = QPen(QColor(255,200,140)); pen.setWidth(6); p.setPen(pen)
            for s in pts2:
                p.drawPoint(int(s[0]), int(s[1]))

# --- Gizmo helper functions (move/rotate/scale snapping) ---
import math
def _deg2rad(d): return d * math.pi / 180.0
def _snap_angle(deg, step_deg):
    if step_deg <= 0: return deg
    return round(deg / step_deg) * step_deg
def _snap_scale(v, step):
    if step <= 0: return v
    return round(v / step) * step
def _make_rot(axis, deg):
    a = _deg2rad(deg)
    c, s = math.cos(a), math.sin(a)
    R = np.eye(4, dtype=np.float32)
    if axis == 'x':
        R[1,1]=c; R[1,2]=-s; R[2,1]=s; R[2,2]=c
    elif axis == 'y':
        R[0,0]=c; R[0,2]=s; R[2,0]=-s; R[2,2]=c
    else: # 'z'
        R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c
    return R
def _make_scale(axis, k):
    S = np.eye(4, dtype=np.float32)
    if axis == 'x': S[0,0] = k
    elif axis == 'y': S[1,1] = k
    elif axis == 'z': S[2,2] = k
    else: S[0,0] = S[1,1] = S[2,2] = k
    return S
def _get_local_axes_from_xform(M):
    rx = M[:3,0]; ry = M[:3,1]; rz = M[:3,2]
    def _norm(v):
        n = np.linalg.norm(v); return v if n < 1e-9 else v / n
    return _norm(rx), _norm(ry), _norm(rz)

# --- Ensure Sketch Overlay classes exist even if PIL import succeeded ---
try:
    SketchLayer
except NameError:  # define overlay classes (PIL present path)
    SNAP_PX = 10
    class SketchEntity:
        def __init__(self):
            self.construction = False
        def draw(self, p: QPainter, view): ...
        def snap_points(self): return []
        def snap_points_typed(self):
            return [(pt, 'other') for pt in self.snap_points()]
    class Polyline2D(SketchEntity):
        def __init__(self, pts=None):
            import numpy as _np
            self.pts = [ _np.array(p, dtype=_np.float32) for p in (pts or []) ]
        def snap_points(self):
            import numpy as _np
            out = [] + self.pts
            for a,b in zip(self.pts[:-1], self.pts[1:]):
                out.append((a+b)/2.0)
            return out
        def draw(self, p: QPainter, view):
            import numpy as _np
            if not self.pts: return
            pts2 = [ view._world_to_screen(_np.array([x,y,0.0], _np.float32)) for (x,y) in self.pts ]
            if len(pts2) >= 2:
                pen = QPen(QColor(220,220,220)); pen.setWidth(2); p.setPen(pen)
                for a,b in zip(pts2[:-1], pts2[1:]):
                    p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            pen = QPen(QColor(255,200,120)); pen.setWidth(6); p.setPen(pen)
            for s in pts2: p.drawPoint(int(s[0]), int(s[1]))
    class Line2D(SketchEntity):
        def __init__(self, p0, p1=None):
            import numpy as _np
            self.p0 = _np.array(p0, dtype=_np.float32)
            self.p1 = _np.array(p1, dtype=_np.float32) if p1 is not None else None
        def snap_points(self):
            import numpy as _np
            pts = [self.p0]
            if self.p1 is not None:
                pts.append(self.p1)
                pts.append((self.p0 + self.p1)/2.0)
            return pts
        def snap_points_typed(self):
            out=[(self.p0,'end')]
            if self.p1 is not None:
                out.append((self.p1,'end'))
                out.append(((self.p0+self.p1)/2.0,'mid'))
            return out
        def draw(self, p: QPainter, view):
            import numpy as _np
            s0 = view._world_to_screen(_np.array([self.p0[0], self.p0[1], 0.0], _np.float32))
            pen = QPen(QColor(220,220,220)); pen.setWidth(2); p.setPen(pen)
            if self.p1 is not None:
                s1 = view._world_to_screen(_np.array([self.p1[0], self.p1[1], 0.0], _np.float32))
                p.drawLine(int(s0[0]), int(s0[1]), int(s1[0]), int(s1[1]))
            pen = QPen(QColor(255,200,140)); pen.setWidth(6); p.setPen(pen); p.drawPoint(int(s0[0]), int(s0[1]))
            if self.p1 is not None:
                p.drawPoint(int(s1[0]), int(s1[1]))
    class Circle2D(SketchEntity):
        def __init__(self, center, radius=0.0):
            import numpy as _np
            self.center = _np.array(center, dtype=_np.float32)
            self.radius = float(radius)
        def snap_points(self):
            import numpy as _np, math as _m
            pts = [self.center]
            if self.radius>1e-6:
                for ang in (0,90,180,270):
                    a=_m.radians(ang)
                    pts.append(self.center + self.radius * _np.array([_np.cos(a), _np.sin(a)], _np.float32))
            return pts
        def snap_points_typed(self):
            return [(self.center,'center')]
        def draw(self, p: QPainter, view):
            import numpy as _np, math as _m
            if self.radius <= 0: return
            segs=40; pts=[]
            for i in range(segs+1):
                a=2*_m.pi*i/segs
                w=self.center + self.radius*_np.array([_np.cos(a), _np.sin(a)], _np.float32)
                s=view._world_to_screen(_np.array([w[0],w[1],0.0], _np.float32))
                pts.append(s)
            pen = QPen(QColor(180,220,255)); pen.setWidth(2); p.setPen(pen)
            for a,b in zip(pts[:-1], pts[1:]): p.drawLine(int(a[0]),int(a[1]),int(b[0]),int(b[1]))
            c=view._world_to_screen(_np.array([self.center[0],self.center[1],0.0], _np.float32))
            pen = QPen(QColor(255,180,120)); pen.setWidth(6); p.setPen(pen); p.drawPoint(int(c[0]), int(c[1]))
    class Rect2D(SketchEntity):
        def __init__(self, p0, p1=None):
            import numpy as _np
            self.p0=_np.array(p0, dtype=_np.float32)
            self.p1=_np.array(p1, dtype=_np.float32) if p1 is not None else None
        def corners(self):
            import numpy as _np
            if self.p1 is None: return []
            x0,y0=self.p0; x1,y1=self.p1
            mnx,mxx=min(x0,x1),max(x0,x1); mny,mxy=min(y0,y1),max(y0,y1)
            return [ _np.array([mnx,mny],_np.float32), _np.array([mxx,mny],_np.float32), _np.array([mxx,mxy],_np.float32), _np.array([mnx,mxy],_np.float32) ]
        def snap_points(self):
            import numpy as _np
            cs=self.corners(); pts=cs[:]
            if len(cs)==4:
                for a,b in zip(cs, cs[1:]+cs[:1]): pts.append((a+b)/2.0)
                cx=sum(c[0] for c in cs)/4.0; cy=sum(c[1] for c in cs)/4.0
                pts.append(_np.array([cx,cy], _np.float32))
            return pts
        def snap_points_typed(self):
            import numpy as _np
            cs = self.corners()
            out=[(c,'end') for c in cs]
            if len(cs)==4:
                for a,b in zip(cs, cs[1:]+cs[:1]): out.append(((a+b)/2.0,'mid'))
                cx = sum(c[0] for c in cs)/4.0; cy = sum(c[1] for c in cs)/4.0
                out.append((_np.array([cx,cy], _np.float32),'center'))
            return out
        def draw(self, p: QPainter, view):
            import numpy as _np
            cs=self.corners()
            if len(cs)<2:
                s0=view._world_to_screen(_np.array([self.p0[0], self.p0[1],0.0], _np.float32))
                pen=QPen(QColor(255,200,140)); pen.setWidth(6); p.setPen(pen); p.drawPoint(int(s0[0]), int(s0[1])); return
            pts2=[ view._world_to_screen(_np.array([c[0],c[1],0.0], _np.float32)) for c in cs ]
            pen=QPen(QColor(200,255,180)); pen.setWidth(2); p.setPen(pen)
            for a,b in zip(pts2, pts2[1:]+pts2[:1]): p.drawLine(int(a[0]),int(a[1]),int(b[0]),int(b[1]))
            pen=QPen(QColor(255,200,140)); pen.setWidth(6); p.setPen(pen)
            for s in pts2: p.drawPoint(int(s[0]), int(s[1]))
    class Arc2D(SketchEntity):
        def __init__(self, p0=None, p1=None, p2=None):
            import numpy as _np
            self.p0 = _np.array(p0, dtype=_np.float32) if p0 is not None else None
            self.p1 = _np.array(p1, dtype=_np.float32) if p1 is not None else None
            self.p2 = _np.array(p2, dtype=_np.float32) if p2 is not None else None
        def _circle_from_3pts(self):
            import numpy as _np
            if self.p0 is None or self.p1 is None or self.p2 is None:
                return None
            x1,y1 = self.p0; x2,y2 = self.p1; x3,y3 = self.p2
            a = x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2
            if abs(a) < 1e-9: return None
            b = (x1*x1 + y1*y1)*(y3 - y2) + (x2*x2 + y2*y2)*(y1 - y3) + (x3*x3 + y3*y3)*(y2 - y1)
            c = (x1*x1 + y1*y1)*(x2 - x3) + (x2*x2 + y2*y2)*(x3 - x1) + (x3*x3 + y3*y3)*(x1 - x2)
            cx = -b/(2*a); cy = -c/(2*a)
            r = float(_np.hypot(x1-cx, y1-cy))
            return _np.array([cx,cy], _np.float32), r
        def snap_points(self):
            import numpy as _np
            pts=[]
            if self.p0 is not None: pts.append(self.p0)
            if self.p2 is not None: pts.append(self.p2)
            circ = self._circle_from_3pts()
            if circ is not None:
                c,_ = circ; pts.append(c)
            if self.p0 is not None and self.p2 is not None:
                pts.append((self.p0 + self.p2)/2.0)
            return pts
        def snap_points_typed(self):
            out=[]
            if self.p0 is not None: out.append((self.p0,'end'))
            if self.p2 is not None: out.append((self.p2,'end'))
            circ = self._circle_from_3pts()
            if circ is not None:
                c,_ = circ; out.append((c,'center'))
            if self.p0 is not None and self.p2 is not None:
                out.append(((self.p0+self.p2)/2.0,'mid'))
            return out
        def draw(self, p: QPainter, view):
            import numpy as _np, math as _m
            if self.p0 is None:
                return
            pen=QPen(QColor(200,240,255)); pen.setWidth(2); p.setPen(pen)
            if self.p1 is None or self.p2 is None:
                s0=view._world_to_screen(_np.array([self.p0[0], self.p0[1],0.0], _np.float32))
                p.drawPoint(int(s0[0]), int(s0[1]))
                if self.p1 is not None:
                    s1=view._world_to_screen(_np.array([self.p1[0], self.p1[1],0.0], _np.float32))
                    p.drawLine(int(s0[0]), int(s0[1]), int(s1[0]), int(s1[1]))
                return
            circ=self._circle_from_3pts()
            if circ is None:
                s0=view._world_to_screen(_np.array([self.p0[0], self.p0[1],0.0], _np.float32))
                s2=view._world_to_screen(_np.array([self.p2[0], self.p2[1],0.0], _np.float32))
                p.drawLine(int(s0[0]), int(s0[1]), int(s2[0]), int(s2[1]))
                return
            c,r=circ
            a0=_m.atan2(self.p0[1]-c[1], self.p0[0]-c[0])
            a1=_m.atan2(self.p1[1]-c[1], self.p1[0]-c[0])
            a2=_m.atan2(self.p2[1]-c[1], self.p2[0]-c[0])
            def ang_dist(a,b):
                d=(b-a+_m.pi)%(2*_m.pi)-_m.pi
                return d
            d1=abs(ang_dist(a0,a1))+abs(ang_dist(a1,a2))
            d2=abs(ang_dist(a0,a2))
            cw=False if d1<d2 else True
            segs=48; pts=[]
            if not cw:
                step=ang_dist(a0,a2)/segs; k=0
                while k<=segs:
                    ang=a0+step*k; k+=1
                    w=c+r*_np.array([_np.cos(ang), _np.sin(ang)], _np.float32)
                    s=view._world_to_screen(_np.array([w[0],w[1],0.0], _np.float32)); pts.append(s)
            else:
                step=ang_dist(a2,a0)/segs; k=0
                while k<=segs:
                    ang=a2+step*k; k+=1
                    w=c+r*_np.array([_np.cos(ang), _np.sin(ang)], _np.float32)
                    s=view._world_to_screen(_np.array([w[0],w[1],0.0], _np.float32)); pts.append(s)
            for a,b in zip(pts[:-1], pts[1:]): p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            pen_pts=QPen(QColor(255,200,140)); pen_pts.setWidth(6); p.setPen(pen_pts)
            s0=view._world_to_screen(_np.array([self.p0[0], self.p0[1],0.0], _np.float32)); p.drawPoint(int(s0[0]), int(s0[1]))
            s2=view._world_to_screen(_np.array([self.p2[0], self.p2[1],0.0], _np.float32)); p.drawPoint(int(s2[0]), int(s2[1]))
            sc=view._world_to_screen(_np.array([c[0], c[1],0.0], _np.float32)); p.drawPoint(int(sc[0]), int(sc[1]))
    # --- Dimension overlays (fallback path) ---
    class DimBase2D:
        def __init__(self):
            self.label_override = None
        def _fmt(self, val: float, panel) -> str:
            try:
                prec = int(getattr(panel, '_dim_precision', 2))
            except Exception:
                prec = 2
            return f"{val:.{prec}f}"
        def draw_text(self, p: QPainter, view, pos_xy, text):
            import numpy as _np
            s = view._world_to_screen(_np.array([pos_xy[0], pos_xy[1], 0.0], _np.float32))
            p.setPen(QPen(QColor(250, 250, 210)))
            p.drawText(int(s[0]+6), int(s[1]-6), text)

    class DimLinear2D(DimBase2D):
        def __init__(self, ref_a, ref_b, panel, offset: float = 0.0):
            super().__init__(); self.ref_a = ref_a; self.ref_b = ref_b; self.panel = panel; self.offset = float(offset)
        def _xy(self, ref):
            if isinstance(ref, tuple) and len(ref)==2 and isinstance(ref[0], Polyline2D):
                ent, idx = ref; return ent.pts[idx]
            if isinstance(ref, tuple) and len(ref)==2 and isinstance(ref[0], Line2D):
                ent, idx = ref
                if idx == 0: return ent.p0
                if idx == 1 and ent.p1 is not None: return ent.p1
                return None
            if isinstance(ref, tuple) and len(ref)==2 and isinstance(ref[0], Rect2D):
                ent, idx = ref
                cs = ent.corners()
                if 0 <= idx < len(cs): return cs[idx]
                return None
            if isinstance(ref, tuple) and ref[0]=='circle' and isinstance(ref[1], Circle2D):
                return ref[1].center
            try:
                import numpy as _np
                if isinstance(ref, (list, tuple)) and len(ref)==2 and all(isinstance(v, (int,float)) for v in ref):
                    return _np.array([float(ref[0]), float(ref[1])], _np.float32)
                if hasattr(ref, 'shape') and getattr(ref, 'shape', None) is not None and len(ref.shape)==1 and ref.shape[0]==2:
                    return ref
            except Exception:
                pass
            return None
        def draw(self, p: QPainter, view):
            import numpy as _np, numpy as np
            a = self._xy(self.ref_a); b = self._xy(self.ref_b)
            if a is None or b is None: return
            v = b - a
            L = float(np.linalg.norm(v))
            if L < 1e-6: return
            n = _np.array([-v[1], v[0]], _np.float32) / L
            p0 = a + n * self.offset
            p1 = b + n * self.offset
            sa = view._world_to_screen(_np.array([p0[0], p0[1], 0.0], _np.float32))
            sb = view._world_to_screen(_np.array([p1[0], p1[1], 0.0], _np.float32))
            ea0 = view._world_to_screen(_np.array([a[0], a[1], 0.0], _np.float32))
            eb0 = view._world_to_screen(_np.array([b[0], b[1], 0.0], _np.float32))
            pen = QPen(QColor(180, 200, 220)); pen.setWidth(1); p.setPen(pen)
            p.drawLine(int(ea0[0]), int(ea0[1]), int(sa[0]), int(sa[1]))
            p.drawLine(int(eb0[0]), int(eb0[1]), int(sb[0]), int(sb[1]))
            pen = QPen(QColor(255, 220, 180)); pen.setWidth(2); p.setPen(pen)
            p.drawLine(int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1]))
            val = float(np.linalg.norm(b - a))
            text = self.label_override if self.label_override is not None else self._fmt(val, self.panel)
            mid = (p0 + p1) * 0.5
            self.draw_text(p, view, mid, text)

    class DimHorizontal2D(DimLinear2D):
        def draw(self, p: QPainter, view):
            import numpy as _np
            a = self._xy(self.ref_a); b = self._xy(self.ref_b)
            if a is None or b is None: return
            n = _np.array([0.0, 1.0], _np.float32)
            p0 = _np.array([a[0], a[1]], _np.float32) + n * self.offset
            p1 = _np.array([b[0], b[1]], _np.float32) + n * self.offset
            sa = view._world_to_screen(_np.array([p0[0], p0[1], 0.0], _np.float32))
            sb = view._world_to_screen(_np.array([p1[0], p1[1], 0.0], _np.float32))
            ea0 = view._world_to_screen(_np.array([a[0], a[1], 0.0], _np.float32))
            eb0 = view._world_to_screen(_np.array([b[0], b[1], 0.0], _np.float32))
            pen = QPen(QColor(180, 200, 220)); pen.setWidth(1); p.setPen(pen)
            p.drawLine(int(ea0[0]), int(ea0[1]), int(sa[0]), int(sa[1]))
            p.drawLine(int(eb0[0]), int(eb0[1]), int(sb[0]), int(sb[1]))
            pen = QPen(QColor(255, 220, 180)); pen.setWidth(2); p.setPen(pen)
            p.drawLine(int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1]))
            val = abs(float(b[0] - a[0]))
            text = self.label_override if self.label_override is not None else self._fmt(val, self.panel)
            mid = (p0 + p1) * 0.5
            self.draw_text(p, view, mid, text)

    class DimVertical2D(DimLinear2D):
        def draw(self, p: QPainter, view):
            import numpy as _np
            a = self._xy(self.ref_a); b = self._xy(self.ref_b)
            if a is None or b is None: return
            n = _np.array([1.0, 0.0], _np.float32)
            p0 = _np.array([a[0], a[1]], _np.float32) + n * self.offset
            p1 = _np.array([b[0], b[1]], _np.float32) + n * self.offset
            sa = view._world_to_screen(_np.array([p0[0], p0[1], 0.0], _np.float32))
            sb = view._world_to_screen(_np.array([p1[0], p1[1], 0.0], _np.float32))
            ea0 = view._world_to_screen(_np.array([a[0], a[1], 0.0], _np.float32))
            eb0 = view._world_to_screen(_np.array([b[0], b[1], 0.0], _np.float32))
            pen = QPen(QColor(180, 200, 220)); pen.setWidth(1); p.setPen(pen)
            p.drawLine(int(ea0[0]), int(ea0[1]), int(sa[0]), int(sa[1]))
            p.drawLine(int(eb0[0]), int(eb0[1]), int(sb[0]), int(sb[1]))
            pen = QPen(QColor(255, 220, 180)); pen.setWidth(2); p.setPen(pen)
            p.drawLine(int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1]))
            val = abs(float(b[1] - a[1]))
            text = self.label_override if self.label_override is not None else self._fmt(val, self.panel)
            mid = (p0 + p1) * 0.5
            self.draw_text(p, view, mid, text)

    class DimRadius2D(DimBase2D):
        def __init__(self, circle_ref, panel):
            super().__init__(); self.circle_ref = circle_ref; self.panel = panel
        def draw(self, p: QPainter, view):
            if not (isinstance(self.circle_ref, tuple) and self.circle_ref[0]=='circle' and isinstance(self.circle_ref[1], Circle2D)):
                return
            c = self.circle_ref[1]
            val = float(c.radius)
            text = self.label_override if self.label_override is not None else self._fmt(val, self.panel)
            pos = c.center + np.array([0.0, c.radius + 0.0], np.float32)
            pen = QPen(QColor(255, 220, 180)); pen.setWidth(1); p.setPen(pen)
            self.draw_text(p, view, pos, f"R {text}")

    class DimDiameter2D(DimRadius2D):
        def draw(self, p: QPainter, view):
            if not (isinstance(self.circle_ref, tuple) and self.circle_ref[0]=='circle' and isinstance(self.circle_ref[1], Circle2D)):
                return
            c = self.circle_ref[1]
            val = float(c.radius*2.0)
            text = self.label_override if self.label_override is not None else self._fmt(val, self.panel)
            pos = c.center + np.array([0.0, c.radius + 0.0], np.float32)
            pen = QPen(QColor(255, 220, 180)); pen.setWidth(1); p.setPen(pen)
            self.draw_text(p, view, pos, f"⌀ {text}")

    class SketchLayer:
        def __init__(self):
            self.entities=[]; self.active_poly=None; self.active_shape=None; self.hover_snap=None
            self.dimensions = []
            self._dim_pick = None
            self._dim_preview = None
            self._dim_placing = False
            self._dim_place_data = None
        def all_snap_points(self):
            pts=[]
            for e in self.entities: pts+=e.snap_points()
            if self.active_poly: pts+=self.active_poly.snap_points()
            if self.active_shape and hasattr(self.active_shape,'snap_points'): pts+=self.active_shape.snap_points()
            return pts
        def all_snap_points_typed(self):
            pts=[]
            for e in self.entities:
                if hasattr(e,'snap_points_typed'):
                    pts += e.snap_points_typed()
                else:
                    pts += [(pt,'other') for pt in e.snap_points()]
            if self.active_poly and hasattr(self.active_poly,'snap_points_typed'):
                pts += self.active_poly.snap_points_typed()
            if self.active_shape and hasattr(self.active_shape,'snap_points_typed'):
                pts += self.active_shape.snap_points_typed()
            return pts
        def draw(self, p: QPainter, view):
            for e in self.entities: e.draw(p, view)
            if self.active_poly: self.active_poly.draw(p, view)
            if self.active_shape: self.active_shape.draw(p, view)
            # draw dimensions
            for d in getattr(self, 'dimensions', []):
                try:
                    d.draw(p, view)
                except Exception:
                    pass
            # draw preview dim if any
            try:
                if self._dim_preview is not None:
                    self._dim_preview.draw(p, view)
            except Exception:
                pass
            if self.hover_snap is not None:
                import numpy as _np
                s=view._world_to_screen(_np.array([self.hover_snap[0], self.hover_snap[1],0.0], _np.float32))
                pen=QPen(QColor(120,255,160)); pen.setWidth(2); p.setPen(pen)
                p.drawLine(int(s[0]-6), int(s[1]), int(s[0]+6), int(s[1]))
                p.drawLine(int(s[0]), int(s[1]-6), int(s[0]), int(s[1]+6))

def load_text(p):
    return Path(p).read_text(encoding="utf-8")

class AnalyticViewport(QOpenGLWidget):
    def __init__(self, parent=None, aacore_scene: AACoreScene | None = None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3,3); fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSamples(4)  # 4x MSAA
        self.setFormat(fmt)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Shared AACore scene (may be injected)
        self.scene = aacore_scene if aacore_scene is not None else AACoreScene()
        # camera
        self.cam_target = np.array([0.0,0.0,0.0], np.float32)
        self.distance = 6.0
        self.yaw=0.0; self.pitch=0.0
        self.cam_pos = np.array([0.0,0.0,self.distance], np.float32)
        self._last_mouse = None
        self._last_buttons = Qt.MouseButton.NoButton
        # shader handles
        self.prog = None
        # debug / feature toggles (could later expose via UI)
        self.debug_mode = 0          # 0 beauty
        self.use_analytic_aa = 1
        self.use_toon = 0
        self.toon_levels = 4.0
        self.curv_strength = 1.0
        self.use_foveated = 1
        self.far_plane = 200.0
        self.show_beta_overlay = 0
        self.beta_overlay_intensity = 0.65
        self.beta_scale = 1.0
        self.beta_cmap = 0  # 0 legacy,1 viridis,2 plasma,3 diverge
        self._settings_path = os.path.join(os.path.expanduser('~'), '.adaptivecad_analytic.json')
        self._load_settings()
        self._beta_lut_tex = None
        self._curv_lut_tex = None
        self.use_curv_lut = 1
        # listen for scene edits (debounced via timer-less simple flag)
        self._scene_dirty = True
        try:
            self.scene.on_changed(self._on_scene_changed)
        except Exception:
            pass
        if aacore_scene is None and len(self.scene.prims) == 0:
            self._demo_scene()
        # selection
        self.selected_index = -1
        self._picking_fbo = None
        self._pick_tex = None
        self._depth_rb = None
        # micro pick FBO
        self._pick_fbo = None
        # translation drag state
        self._drag_move_active = False
        self._drag_last_pos = None
        # Overlay editing state
        self._sk_point_sel = None  # (entity, index) for Polyline2D point or ('circle', entity) for circle center
        self._sk_dragging = False
        # Lightweight 2D polyline overlays (for algorithm visuals)
        # Each entry: { 'pts': [(x,y),...], 'color': (r,g,b), 'width': int, 'scale': (sx,sy), 'offset': (ox,oy) }
        self._poly_overlays = []

    # Resolve a sketch reference used by dimension tools into a world XY np.array([x,y], float32)
    def _ref_to_xy(self, ref):
        import numpy as _np
        try:
            if isinstance(ref, tuple) and len(ref)==2 and isinstance(ref[0], Polyline2D):
                ent, idx = ref; return ent.pts[idx]
            if isinstance(ref, tuple) and len(ref)==2 and isinstance(ref[0], Line2D):
                ent, idx = ref
                if idx == 0: return ent.p0
                if idx == 1 and ent.p1 is not None: return ent.p1
                return None
            if isinstance(ref, tuple) and len(ref)==2 and isinstance(ref[0], Rect2D):
                ent, idx = ref
                cs = ent.corners()
                if 0 <= idx < len(cs): return cs[idx]
                return None
            if isinstance(ref, tuple) and ref[0]=='circle' and isinstance(ref[1], Circle2D):
                return ref[1].center
            if isinstance(ref, (list, tuple)) and len(ref)==2 and all(isinstance(v, (int,float)) for v in ref):
                return _np.array([float(ref[0]), float(ref[1])], _np.float32)
            if hasattr(ref, 'shape') and getattr(ref, 'shape', None) is not None and len(ref.shape)==1 and ref.shape[0]==2:
                return ref
        except Exception:
            return None
        return None

    def _on_scene_changed(self):
        self._scene_dirty = True
        self.update()

    def _demo_scene(self):
        self.scene.add(Prim(KIND_SPHERE, [1.2,0,0,0], beta=0.12, color=(0.9,0.4,0.3)))
        self.scene.add(Prim(KIND_CAPSULE, [0.4,2.0,0,0], beta=0.0, color=(0.3,0.7,0.9)))
        self.scene.add(Prim(KIND_TORUS, [1.2,0.25,0,0], beta=0.05, color=(0.6,0.9,0.5)))

    def sizeHint(self): return QSize(800,600)

    # --- OpenGL lifecycle ---
    def initializeGL(self):
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)  # Enable MSAA
        shader_dir = Path(__file__).parent.parent / "analytic" / "shaders"
        vs = load_text(shader_dir / "sdf.vert")
        fs = load_text(shader_dir / "sdf.frag")
        self.prog = glCreateProgram()
        vsh = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vsh, vs); glCompileShader(vsh)
        if not glGetShaderiv(vsh, GL_COMPILE_STATUS): raise RuntimeError(glGetShaderInfoLog(vsh).decode())
        fsh = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fsh, fs); glCompileShader(fsh)
        if not glGetShaderiv(fsh, GL_COMPILE_STATUS): raise RuntimeError(glGetShaderInfoLog(fsh).decode())
        glAttachShader(self.prog, vsh); glAttachShader(self.prog, fsh); glLinkProgram(self.prog)
        if not glGetProgramiv(self.prog, GL_LINK_STATUS): raise RuntimeError(glGetProgramInfoLog(self.prog).decode())
        glDeleteShader(vsh); glDeleteShader(fsh)
        # fullscreen quad
        self._vao = glGenVertexArrays(1); glBindVertexArray(self._vao)
        v = np.array([-1,-1,  1,-1, -1, 1,   1,1], dtype=np.float32)
        self._vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, v.nbytes, v, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        # create simple FBO for ID picking
        self._init_picking()
        # --- 1x1 FBO for micro picking ---
        try:
            self._pick_fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self._pick_fbo)
            self._pick_tex_micro = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self._pick_tex_micro)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._pick_tex_micro, 0)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        except Exception as e:
            log.debug(f"Micro pick FBO init failed: {e}")

    def _init_picking(self):
        try:
            if self._picking_fbo:
                return
            self._picking_fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self._picking_fbo)
            self._pick_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self._pick_tex)
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,self.width(),self.height(),0,GL_RGBA,GL_UNSIGNED_BYTE,None)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
            glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,self._pick_tex,0)
            self._depth_rb = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER,self._depth_rb)
            glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8,self.width(),self.height())
            glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,self._depth_rb)
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                log.debug("Picking FBO incomplete")
            glBindFramebuffer(GL_FRAMEBUFFER,0)
        except Exception as e:
            log.debug(f"Picking FBO init failed: {e}")

    def resizeGL(self, w, h):
        # resize picking attachments
        try:
            if self._pick_tex:
                glBindTexture(GL_TEXTURE_2D, self._pick_tex)
                glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,None)
            if self._depth_rb:
                glBindRenderbuffer(GL_RENDERBUFFER,self._depth_rb)
                glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8,w,h)
        except Exception:
            pass

    def _cam_basis(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        # forward in -Z
        f = np.array([sy*cp, sp, -cy*cp], np.float32)
        r = np.array([ cy, 0.0,  sy], np.float32)
        u = np.cross(r, f)
        R = np.stack([r,u,-f], axis=1).astype(np.float32)  # columns
        return R

    def _update_camera(self):
        # forward vector points from cam to target, so place camera at target - f*distance
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        f = np.array([sy*cp, sp, -cy*cp], np.float32)
        self.cam_pos = self.cam_target - f * self.distance
        self.update()

    def paintGL(self):
        glViewport(0,0,self.width(), self.height())
        glClearColor(*self.scene.bg_color, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self._render_scene_internal(debug_override=None)
        # Sketch overlay
        try:
            panel = self.parent()
        except Exception:
            panel = None
        if panel is not None and getattr(panel,'_sketch_on',False):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            # Draw grid if enabled
            try:
                if getattr(panel, '_sketch_grid_on', False):
                    # Draw grid with constant on-screen spacing (pixels) and contrast color
                    self._draw_grid(p, panel)
            except Exception:
                pass
            panel._sketch.draw(p, self)
            p.end()
        # Algorithm overlays (polylines)
        if self._poly_overlays:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            for ov in self._poly_overlays:
                try:
                    pts = ov.get('pts') or []
                    if len(pts) < 2:
                        continue
                    col = ov.get('color', (120,255,160))
                    width = int(ov.get('width', 2))
                    sx, sy = ov.get('scale', (1.0, 1.0))
                    ox, oy = ov.get('offset', (0.0, 0.0))
                    pen = QPen(QColor(int(col[0]), int(col[1]), int(col[2]))); pen.setWidth(width)
                    p.setPen(pen)
                    # world z fixed at 0 for overlay
                    last = None
                    for (x,y) in pts:
                        wx = float(x)*float(sx) + float(ox)
                        wy = float(y)*float(sy) + float(oy)
                        spt = self._world_to_screen(np.array([wx, wy, 0.0], np.float32))
                        if last is not None:
                            p.drawLine(int(last[0]), int(last[1]), int(spt[0]), int(spt[1]))
                        last = spt
                except Exception:
                    continue
            p.end()

    # --- Sketch projection helpers ---
    def _world_to_screen(self, p3):
        # Simple orthographic-ish projection using camera distance for scale
        w,h = float(self.width()), float(self.height())
        # derive right/up from camera basis
        R = self._cam_basis()
        right = R[:,0]; up = R[:,1]
        rel = p3 - self.cam_pos
        x = np.dot(rel, right)
        y = np.dot(rel, up)
        s = 80.0 / max(1e-5, self.distance)
        return np.array([w*0.5 + x*s, h*0.5 - y*s], np.float32)

    def _screen_to_world_grid_snap(self, sx, sy, step):
        wpt = self._screen_to_world_xy(sx, sy)
        if step <= 0:
            return wpt
        wpt_snapped = wpt.copy()
        wpt_snapped[0] = round(wpt[0] / step) * step
        wpt_snapped[1] = round(wpt[1] / step) * step
        return wpt_snapped

    def _grid_step_world_for_panel(self, panel) -> float:
        """Return effective grid step in world units. If panel requests screen-locked grid,
        interpret panel._sketch_grid_step as pixel spacing and convert using current zoom."""
        try:
            step_cfg = float(getattr(panel, '_sketch_grid_step', 10.0))
        except Exception:
            step_cfg = 10.0
        lock_to_screen = bool(getattr(panel, '_sketch_grid_screen_lock', True))
        if lock_to_screen:
            # _world_to_screen uses s = 80 / distance, so 1 world unit = s pixels
            s = 80.0 / max(1e-5, float(self.distance))
            # desired pixel spacing -> world spacing
            return max(1e-8, float(step_cfg) / float(s))
        return max(1e-8, float(step_cfg))

    def _draw_grid(self, painter: QPainter, panel):
        # Compute world step from panel settings (optionally screen-locked)
        step_world = self._grid_step_world_for_panel(panel)
        if step_world <= 0:
            return
        w, h = self.width(), self.height()
        # Compute world bbox corners at Z≈0
        tl = self._screen_to_world_xy(0,0)
        br = self._screen_to_world_xy(w,h)
        xmin, xmax = sorted([tl[0], br[0]])
        ymin, ymax = sorted([tl[1], br[1]])
        import math
        x0 = math.floor(xmin/step_world)*step_world
        y0 = math.floor(ymin/step_world)*step_world
        # Choose grid color as high-contrast inverse of background
        try:
            br_, bg_, bb_ = self.scene.bg_color
        except Exception:
            br_, bg_, bb_ = (0.1, 0.1, 0.12)
        # Luminance to decide black/white, then apply slight transparency
        lum = 0.2126*br_ + 0.7152*bg_ + 0.0722*bb_
        if lum < 0.5:
            # dark background -> light grid
            cr, cg, cb, ca = 240, 240, 240, 110
        else:
            # light background -> dark grid
            cr, cg, cb, ca = 20, 20, 20, 110
        pen = QPen(QColor(int(cr), int(cg), int(cb), int(ca))); pen.setWidth(1)
        painter.setPen(pen)
        # verticals
        x = x0
        while x <= xmax:
            s1 = self._world_to_screen(np.array([x, ymin, 0.0], np.float32))
            s2 = self._world_to_screen(np.array([x, ymax, 0.0], np.float32))
            painter.drawLine(int(s1[0]), int(s1[1]), int(s2[0]), int(s2[1]))
            x += step_world
        # horizontals
        y = y0
        while y <= ymax:
            s1 = self._world_to_screen(np.array([xmin, y, 0.0], np.float32))
            s2 = self._world_to_screen(np.array([xmax, y, 0.0], np.float32))
            painter.drawLine(int(s1[0]), int(s1[1]), int(s2[0]), int(s2[1]))
            y += step_world

    def _screen_to_world_xy(self, sx, sy):
        # Map screen to world on Z≈0 plane
        w,h = float(self.width()), float(self.height())
        R = self._cam_basis(); right = R[:,0]; up = R[:,1]; forward = -R[:,2]
        s = 80.0 / max(1e-5, self.distance)
        dx = (sx - w*0.5)/s
        dy = -(sy - h*0.5)/s
        # construct point relative to camera
        pos = self.cam_pos + right*dx + up*dy
        # project to z=0 along forward
        if abs(forward[2]) > 1e-5:
            t = -pos[2]/forward[2]
            pos = pos + forward*t
        pos[2] = 0.0
        return pos

    def _render_scene_internal(self, debug_override=None):
        if not self.prog:
            return
        glUseProgram(self.prog)
        glUniform3f(glGetUniformLocation(self.prog,"u_cam_pos"), *self.cam_pos)
        R = self._cam_basis(); glUniformMatrix3fv(glGetUniformLocation(self.prog,"u_cam_rot"), 1, GL_FALSE, R.T)
        glUniform2f(glGetUniformLocation(self.prog,"u_res"), float(self.width()), float(self.height()))
        glUniform3f(glGetUniformLocation(self.prog,"u_env"), *self.scene.env_light)
        glUniform3f(glGetUniformLocation(self.prog,"u_bg"), *self.scene.bg_color)
        pack = self.scene.to_gpu_structs(max_prims=MAX_PRIMS)
        n = int(pack['count'])
        if n>0:
            try:
                import logging
                logging.getLogger("adaptivecad.gui").debug(
                    f"GPU upload prim0 xform (col-major packed) translation={pack['xform'][0][12:15].tolist()} raw_first16={pack['xform'][0].tolist()}")
            except Exception:
                pass
        U = lambda name: glGetUniformLocation(self.prog, name)
        glUniform1i(U("u_count"), n)
        glUniform1iv(U("u_kind"), n, pack['kind'])
        glUniform1iv(U("u_op"), n, pack['op'])
        glUniform1fv(U("u_beta"), n, pack['beta'])
        glUniform3fv(U("u_color"), n, pack['color'])
        glUniform4fv(U("u_params"), n, pack['params'])
        glUniformMatrix4fv(U("u_xform"), n, GL_FALSE, pack['xform'])
        if 'xform_inv' in pack:
            glUniformMatrix4fv(U("u_xform_inv"), n, GL_FALSE, pack['xform_inv'])
        glUniform1i(U("u_mode"), 0)
        mode_val = self.debug_mode if debug_override is None else debug_override
        glUniform1i(U("u_debug"), int(mode_val))
        glUniform1f(U("u_far"), float(self.far_plane))
        glUniform1i(U("u_use_analytic_aa"), int(self.use_analytic_aa))
        glUniform1i(U("u_use_toon"), int(self.use_toon))
        glUniform1f(U("u_toon_levels"), float(self.toon_levels))
        glUniform1f(U("u_curv_strength"), float(self.curv_strength))
        glUniform1i(U("u_use_foveated"), int(self.use_foveated))
        glUniform1i(U("u_show_beta_overlay"), int(self.show_beta_overlay))
        glUniform1f(U("u_beta_overlay_intensity"), float(self.beta_overlay_intensity))
        glUniform1f(U("u_beta_scale"), float(self.beta_scale))
        glUniform1i(U("u_beta_cmap"), int(self.beta_cmap))
        glUniform1i(U("u_selected"), int(self.selected_index))
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def _draw_frame(self, debug_override=None):  # retained for other callers
        glViewport(0,0,self.width(), self.height())
        glClearColor(*self.scene.bg_color, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self._render_scene_internal(debug_override)

    def keyPressEvent(self, event):
        k = event.key()
        changed = False
        # number keys for debug modes
        if k in (Qt.Key_0, Qt.Key.Key_0): self.debug_mode=0; changed=True
        elif k in (Qt.Key_1, Qt.Key.Key_1): self.debug_mode=1; changed=True
        elif k in (Qt.Key_2, Qt.Key.Key_2): self.debug_mode=2; changed=True
        elif k in (Qt.Key_3, Qt.Key.Key_3): self.debug_mode=3; changed=True
        elif k in (Qt.Key_4, Qt.Key.Key_4): self.debug_mode=4; changed=True
        elif k in (Qt.Key_5, Qt.Key.Key_5): self.debug_mode=5; changed=True  # adaptive heatmap
        elif k == Qt.Key_Space:
            self.debug_mode = (self.debug_mode + 1) % 6; changed=True
        elif k == Qt.Key_A: self.use_analytic_aa ^=1; changed=True
        elif k == Qt.Key_F: self.use_foveated ^=1; changed=True
        elif k == Qt.Key_T: self.use_toon ^=1; changed=True
        elif k == Qt.Key_Plus or k == Qt.Key_Equal: # '+' (with shift) or '='
            self.toon_levels = min(self.toon_levels+1.0, 8.0); changed=True
        elif k == Qt.Key_Minus: self.toon_levels = max(self.toon_levels-1.0, 1.0); changed=True
        elif k == Qt.Key_C:
            # cycle curvature strength
            if self.curv_strength < 0.25: self.curv_strength = 0.5
            elif self.curv_strength < 0.75: self.curv_strength = 1.0
            else: self.curv_strength = 0.0
            changed=True
        elif k == Qt.Key_G: self.save_gbuffers(); return
        elif k == Qt.Key_H:
            log.info(self._shortcuts_help())
        elif k == Qt.Key_B:
            self.show_beta_overlay ^=1; changed=True
        elif k == Qt.Key_Period: # increase beta overlay intensity
            self.beta_overlay_intensity = min(1.0, self.beta_overlay_intensity+0.05); changed=True
        elif k == Qt.Key_Comma:  # decrease intensity
            self.beta_overlay_intensity = max(0.0, self.beta_overlay_intensity-0.05); changed=True
        elif k == Qt.Key_Slash:  # increase beta scale
            self.beta_scale = min(8.0, self.beta_scale * 1.25); changed=True
        elif k == Qt.Key_Question: # SHIFT + '/' may map same; ensure alternative
            self.beta_scale = min(8.0, self.beta_scale * 1.25); changed=True
        elif k == Qt.Key_Backslash: # decrease beta scale
            self.beta_scale = max(0.125, self.beta_scale / 1.25); changed=True
        elif k == Qt.Key_V: # cycle beta color map
            self.beta_cmap = (self.beta_cmap + 1) % 4; changed=True
        if changed:
            self._update_title(); self.update(); self._save_settings()
        else:
            try:
                panel = self.parent()
            except Exception:
                panel = None
            # Sketch polyline finalize / undo
            if panel is not None and getattr(panel,'_sketch_on',False) and panel._sketch_mode=='polyline':
                if k in (Qt.Key_Return, Qt.Key_Enter):
                    if panel._sketch.active_poly and len(panel._sketch.active_poly.pts)>=2:
                        panel._sketch.entities.append(panel._sketch.active_poly)
                    panel._sketch.active_poly = None
                    self.update(); return
                elif k == Qt.Key_Escape:
                    if panel._sketch.active_poly and panel._sketch.active_poly.pts:
                        panel._sketch.active_poly.pts.pop()
                        if not panel._sketch.active_poly.pts:
                            panel._sketch.active_poly = None
                        self.update(); return
            # Cancel dimension placing with ESC
            if panel is not None and getattr(panel,'_sketch_on',False) and k == Qt.Key_Escape:
                try:
                    panel._sketch._dim_placing = False; panel._sketch._dim_place_data = None; panel._sketch._dim_preview = None
                except Exception:
                    pass
                self.update(); return
            # Sketch delete selected vertex/entity
            if panel is not None and getattr(panel,'_sketch_on',False) and k in (Qt.Key_Delete, Qt.Key_Backspace):
                if self._delete_selected_sketch():
                    try:
                        if hasattr(panel, '_update_sketch_info'):
                            panel._update_sketch_info(self._sk_point_sel)
                    except Exception:
                        pass
                    self.update(); return
            # Gizmo mode shortcuts
            if panel is not None:
                if k == Qt.Key_R:
                    panel._gizmo_mode = 'rotate'; self.update(); return
                elif k == Qt.Key_S and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                    panel._gizmo_mode = 'scale'; self.update(); return
                elif k == Qt.Key_M:
                    panel._gizmo_mode = 'move'; self.update(); return
                elif k == Qt.Key_X:
                    panel._axis_lock = 'x'; return
                elif k == Qt.Key_Y:
                    panel._axis_lock = 'y'; return
                elif k == Qt.Key_Z:
                    panel._axis_lock = 'z'; return
                elif k == Qt.Key_Escape and not (getattr(panel,'_sketch_on',False) and panel._sketch_mode=='polyline'):
                    panel._axis_lock = None; return
            # --- Keyboard nudges (view-aligned + world-aligned) ---
            pr = self._current_prim()
            if pr is not None:
                # determine step
                step_widget = getattr(self.parent(), '_nudge_step', None)
                try:
                    step = float(step_widget.value()) if step_widget is not None else 1.0
                except Exception:
                    step = 1.0
                moved = False
                basis = self._cam_basis()
                pos_vec = pr.xform.M[:3,3].astype(np.float32)
                # View-aligned (WASD + Q/E)
                if k == Qt.Key_W: pos_vec += basis[:,1]*step; moved=True
                elif k == Qt.Key_S: pos_vec -= basis[:,1]*step; moved=True
                elif k == Qt.Key_D: pos_vec += basis[:,0]*step; moved=True
                elif k == Qt.Key_A: pos_vec -= basis[:,0]*step; moved=True
                elif k == Qt.Key_E: pos_vec -= basis[:,2]*step; moved=True  # forward (-Z in basis third column sign depending)
                elif k == Qt.Key_Q: pos_vec += basis[:,2]*step; moved=True
                # World-aligned (arrows + PgUp/PgDn)
                elif k == Qt.Key_Up: pos_vec += np.array([0,step,0], np.float32); moved=True
                elif k == Qt.Key_Down: pos_vec -= np.array([0,step,0], np.float32); moved=True
                elif k == Qt.Key_Right: pos_vec += np.array([step,0,0], np.float32); moved=True
                elif k == Qt.Key_Left: pos_vec -= np.array([step,0,0], np.float32); moved=True
                elif k == Qt.Key_PageUp: pos_vec += np.array([0,0,step], np.float32); moved=True
                elif k == Qt.Key_PageDown: pos_vec -= np.array([0,0,step], np.float32); moved=True
                if moved:
                    self._apply_panel_move(pos_vec)
                    return
            super().keyPressEvent(event)

    # --- Polyline overlay API ---
    def clear_polyline_overlays(self):
        try:
            self._poly_overlays.clear()
        except Exception:
            self._poly_overlays = []
        self.update()

    def add_polyline_overlay(self, pts, color=(120,255,160), width=3, scale=(1.0,1.0), offset=(0.0,0.0)):
        try:
            self._poly_overlays.append({
                'pts': list(pts) if pts is not None else [],
                'color': tuple(color),
                'width': int(width),
                'scale': tuple(scale),
                'offset': tuple(offset),
            })
        except Exception:
            pass
        self.update()

    # --- Mouse Interaction ---
    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse = event.position()
        self._last_buttons = event.buttons()
        # Plain LMB (no modifiers) -> pick
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            # Use full-resolution FBO picking for accurate coordinate-based selection.
            try:
                self._perform_pick(int(event.position().x()), int(event.position().y()))
                pid = self.selected_index
                if pid >= 0:
                    try:
                        if hasattr(self.parent(), '_select_prim'):
                            self.parent()._select_prim(pid)
                    except Exception:
                        pass
                self.update()
            except Exception as e:
                log.debug(f"mousePressEvent pick failed: {e}")
        # Shift + LMB -> begin move-drag if a primitive selected
        if (event.button() == Qt.MouseButton.LeftButton and \
            (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) and \
            self._current_prim() is not None):
            self._drag_move_active = True
            self._drag_last_pos = event.position()
        # Sketch polyline placement (before default)
        try:
            panel = self.parent()
        except Exception:
            panel = None
        if panel is not None and getattr(panel,'_sketch_on',False) and event.button()==Qt.MouseButton.LeftButton:
            mode = panel._sketch_mode
            if getattr(panel, '_sketch_snap_grid', False):
                step = self._grid_step_world_for_panel(panel)
                w3 = self._screen_to_world_grid_snap(event.position().x(), event.position().y(), step)
            else:
                w3 = self._screen_to_world_xy(event.position().x(), event.position().y())
            pt_world2 = panel._sketch.hover_snap if panel._sketch.hover_snap is not None else w3[:2]
            if mode=='polyline':
                if panel._sketch.active_poly is None:
                    panel._sketch.active_poly = Polyline2D([])
                panel._sketch.active_poly.pts.append(np.array([pt_world2[0],pt_world2[1]], np.float32))
                self.update(); return
            elif mode=='circle':
                if panel._sketch.active_shape is None:
                    panel._sketch.active_shape = Circle2D(pt_world2, 0.0)
                else:
                    # finalize
                    if isinstance(panel._sketch.active_shape, Circle2D) and panel._sketch.active_shape.radius>0:
                        panel._sketch.entities.append(panel._sketch.active_shape)
                        # Optional kernel sync: add a thin torus ring at same center/radius
                        try:
                            if bool(getattr(panel, '_sketch_circle_to_ring', False)):
                                c = panel._sketch.active_shape
                                R = float(max(1e-6, c.radius))
                                r = max(0.01, R * 0.05)
                                pr = Prim(KIND_TORUS, [R, r, 0.0, 0.0], beta=0.0, color=(0.9,0.7,0.3))
                                if hasattr(pr, 'set_transform'):
                                    pr.set_transform(pos=[float(c.center[0]), float(c.center[1]), 0.0], euler=[0.0,0.0,0.0], scale=[1.0,1.0,1.0])
                                else:
                                    pr.xform.M[:3,3] = np.array([float(c.center[0]), float(c.center[1]), 0.0], np.float32)
                                self.scene.add(pr)
                        except Exception as _e:
                            try:
                                log.debug(f"circle→kernel sync failed: {_e}")
                            except Exception:
                                pass
                    panel._sketch.active_shape = None
                self.update(); return
            elif mode=='rect':
                if panel._sketch.active_shape is None:
                    panel._sketch.active_shape = Rect2D(pt_world2)
                else:
                    if isinstance(panel._sketch.active_shape, Rect2D):
                        panel._sketch.active_shape.p1 = np.array([pt_world2[0],pt_world2[1]], np.float32)
                        if panel._sketch.active_shape.p1 is not None:
                            panel._sketch.entities.append(panel._sketch.active_shape)
                    panel._sketch.active_shape = None
                self.update(); return
            elif mode=='line':
                if panel._sketch.active_shape is None:
                    panel._sketch.active_shape = Line2D(pt_world2)
                else:
                    if isinstance(panel._sketch.active_shape, Line2D):
                        panel._sketch.active_shape.p1 = np.array([pt_world2[0],pt_world2[1]], np.float32)
                        panel._sketch.entities.append(panel._sketch.active_shape)
                    panel._sketch.active_shape = None
                self.update(); return
            elif mode=='arc3':
                if panel._sketch.active_shape is None:
                    panel._sketch.active_shape = Arc2D(pt_world2, None, None)
                elif isinstance(panel._sketch.active_shape, Arc2D) and panel._sketch.active_shape.p1 is None:
                    panel._sketch.active_shape.p1 = np.array([pt_world2[0],pt_world2[1]], np.float32)
                else:
                    if isinstance(panel._sketch.active_shape, Arc2D):
                        panel._sketch.active_shape.p2 = np.array([pt_world2[0],pt_world2[1]], np.float32)
                        panel._sketch.entities.append(panel._sketch.active_shape)
                    panel._sketch.active_shape = None
                self.update(); return
            elif mode=='dimension':
                # Typed placement: Linear/H/V = pick A, pick B, then click to set offset; Radius/Diameter use circle center
                # If we are in placing phase, compute offset and finalize
                if panel._sketch._dim_placing and isinstance(panel._sketch._dim_place_data, dict):
                    dt = panel._sketch._dim_place_data.get('type','linear')
                    a = panel._sketch._dim_place_data.get('a'); b = panel._sketch._dim_place_data.get('b')
                    # current mouse world point
                    hit = self._screen_to_world_xy(event.position().x(), event.position().y())
                    wpt = np.array([hit[0], hit[1]], np.float32)
                    # derive offset along normal
                    if dt == 'horizontal':
                        pa = self._ref_to_xy(a); pb = self._ref_to_xy(b)
                        if pa is None or pb is None:
                            panel._sketch._dim_placing = False; panel._sketch._dim_place_data = None; return
                        off = float(wpt[1] - pa[1])
                        panel._sketch.dimensions.append(DimHorizontal2D(a, b, panel, offset=off))
                    elif dt == 'vertical':
                        pa = self._ref_to_xy(a); pb = self._ref_to_xy(b)
                        if pa is None or pb is None:
                            panel._sketch._dim_placing = False; panel._sketch._dim_place_data = None; return
                        off = float(wpt[0] - pa[0])
                        panel._sketch.dimensions.append(DimVertical2D(a, b, panel, offset=off))
                    else:
                        # general: offset is signed distance from mouse to AB line along its normal
                        pa = self._ref_to_xy(a); pb = self._ref_to_xy(b)
                        if pa is None or pb is None:
                            panel._sketch._dim_placing = False; panel._sketch._dim_place_data = None; return
                        v = pb - pa; L = float(np.linalg.norm(v))
                        if L < 1e-6:
                            panel._sketch._dim_placing = False; panel._sketch._dim_place_data = None; return
                        n = np.array([-v[1], v[0]], np.float32) / L
                        off = float(np.dot(wpt - pa, n))
                        panel._sketch.dimensions.append(DimLinear2D(a, b, panel, offset=off))
                    # reset state
                    panel._sketch._dim_pick = None; panel._sketch._dim_preview = None
                    panel._sketch._dim_placing = False; panel._sketch._dim_place_data = None
                    self.update(); return
                # Otherwise proceed with picking references
                sel = self._pick_nearest_sketch_vertex(event.position().x(), event.position().y())
                if sel is None:
                    return
                dt = getattr(panel, '_dim_type', 'linear')
                if dt in ('radius','diameter'):
                    if isinstance(sel, tuple) and sel[0]=='circle' and isinstance(sel[1], Circle2D):
                        if dt=='radius':
                            panel._sketch.dimensions.append(DimRadius2D(('circle', sel[1]), panel))
                        else:
                            panel._sketch.dimensions.append(DimDiameter2D(('circle', sel[1]), panel))
                        self.update(); return
                    else:
                        return
                # two-point dims: pick A then B, then enter placing phase
                if panel._sketch._dim_pick is None:
                    panel._sketch._dim_pick = sel
                else:
                    a = panel._sketch._dim_pick; b = sel
                    panel._sketch._dim_place_data = {'type': dt, 'a': a, 'b': b}
                    panel._sketch._dim_placing = True
                    # create a preview with zero offset first
                    if dt=='horizontal':
                        panel._sketch._dim_preview = DimHorizontal2D(a, b, panel, offset=0.0)
                    elif dt=='vertical':
                        panel._sketch._dim_preview = DimVertical2D(a, b, panel, offset=0.0)
                    else:
                        panel._sketch._dim_preview = DimLinear2D(a, b, panel, offset=0.0)
                    self.update(); return
            elif mode=='insert_vertex':
                # Insert on nearest polyline segment or split a line
                try:
                    w2 = np.array([pt_world2[0], pt_world2[1]], np.float32)
                    best_px = 1e9; best_ent = None; best_idx = -1; best_point = None
                    for ent in panel._sketch.entities:
                        if isinstance(ent, Polyline2D) and len(ent.pts) >= 2:
                            for i in range(len(ent.pts)-1):
                                a = ent.pts[i]; b = ent.pts[i+1]
                                ab = b - a; t = 0.0
                                denom = float(np.dot(ab, ab));
                                if denom > 1e-9:
                                    t = max(0.0, min(1.0, float(np.dot(w2 - a, ab) / denom)))
                                p = a + t*ab
                                s = self._world_to_screen(np.array([p[0], p[1], 0.0], np.float32))
                                dpx = float(np.hypot(s[0]-event.position().x(), s[1]-event.position().y()))
                                if dpx < best_px:
                                    best_px = dpx; best_ent = ent; best_idx = i+1; best_point = p
                        elif isinstance(ent, Line2D) and ent.p1 is not None:
                            a = ent.p0; b = ent.p1
                            ab = b - a; t = 0.0
                            denom = float(np.dot(ab, ab));
                            if denom > 1e-9:
                                t = max(0.0, min(1.0, float(np.dot(w2 - a, ab) / denom)))
                            p = a + t*ab
                            s = self._world_to_screen(np.array([p[0], p[1], 0.0], np.float32))
                            dpx = float(np.hypot(s[0]-event.position().x(), s[1]-event.position().y()))
                            if dpx < best_px:
                                best_px = dpx; best_ent = ent; best_idx = 1; best_point = p
                    # threshold in pixels
                    if best_ent is None or best_px > 12.0:
                        return
                    if isinstance(best_ent, Polyline2D):
                        ins = best_point if getattr(panel, '_insert_midpoint', False) else w2
                        best_ent.pts.insert(best_idx, ins.astype(np.float32))
                    elif isinstance(best_ent, Line2D):
                        # split into two lines
                        a = best_ent.p0; b = best_ent.p1
                        mid = (best_point if getattr(panel, '_insert_midpoint', False) else w2).copy()
                        panel._sketch.entities.remove(best_ent)
                        panel._sketch.entities.append(Line2D(a, mid))
                        panel._sketch.entities.append(Line2D(mid, b))
                    self.update(); return
                except Exception:
                    return
            # Selection of nearest sketch vertex (Ctrl+Click)
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                sel = self._pick_nearest_sketch_vertex(event.position().x(), event.position().y())
                self._sk_point_sel = sel
                self._sk_dragging = sel is not None
                try:
                    if hasattr(panel, '_update_sketch_info'):
                        panel._update_sketch_info(sel)
                except Exception:
                    pass
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_mouse is None:
            self._last_mouse = event.position()
        dx = event.position().x() - self._last_mouse.x()
        dy = event.position().y() - self._last_mouse.y()
        # Sketch hover snapping
        try:
            panel = self.parent()
        except Exception:
            panel = None
        if panel is not None and getattr(panel,'_sketch_on',False):
            hit = self._screen_to_world_xy(event.position().x(), event.position().y())
            wpt = np.array([hit[0], hit[1]], np.float32)
            snap = None; best = 1e9
            for sp in panel._sketch.all_snap_points():
                scr_sp = self._world_to_screen(np.array([sp[0],sp[1],0.0], np.float32))
                dpx = float(np.hypot(scr_sp[0]-event.position().x(), scr_sp[1]-event.position().y()))
                if dpx < SNAP_PX and dpx < best:
                    best = dpx; snap = sp
            panel._sketch.hover_snap = snap if snap is not None else wpt
            # live update active shape dimensions
            if panel._sketch_mode=='circle' and isinstance(panel._sketch.active_shape, Circle2D):
                c = panel._sketch.active_shape.center
                panel._sketch.active_shape.radius = float(np.linalg.norm(wpt - c))
            elif panel._sketch_mode=='rect' and isinstance(panel._sketch.active_shape, Rect2D):
                panel._sketch.active_shape.p1 = wpt.copy()
            elif panel._sketch_mode=='line' and isinstance(panel._sketch.active_shape, Line2D):
                panel._sketch.active_shape.p1 = wpt.copy()
            elif panel._sketch_mode=='arc3' and isinstance(panel._sketch.active_shape, Arc2D):
                if panel._sketch.active_shape.p1 is None:
                    panel._sketch.active_shape.p1 = wpt.copy()
                else:
                    panel._sketch.active_shape.p2 = wpt.copy()
            # dimension live preview
            if panel._sketch_mode=='dimension':
                # If in placing phase, update preview offset dynamically
                if panel._sketch._dim_placing and isinstance(panel._sketch._dim_place_data, dict):
                    hit = self._screen_to_world_xy(event.position().x(), event.position().y())
                    wpt2 = np.array([hit[0], hit[1]], np.float32)
                    dt = panel._sketch._dim_place_data.get('type','linear')
                    a = panel._sketch._dim_place_data.get('a'); b = panel._sketch._dim_place_data.get('b')
                    if dt == 'horizontal':
                        pa = self._ref_to_xy(a)
                        off = float(wpt2[1] - (pa[1] if pa is not None else wpt2[1]))
                        panel._sketch._dim_preview = DimHorizontal2D(a, b, panel, offset=off)
                    elif dt == 'vertical':
                        pa = self._ref_to_xy(a)
                        off = float(wpt2[0] - (pa[0] if pa is not None else wpt2[0]))
                        panel._sketch._dim_preview = DimVertical2D(a, b, panel, offset=off)
                    else:
                        pa = self._ref_to_xy(a); pb = self._ref_to_xy(b)
                        if pa is not None and pb is not None:
                            v = pb - pa; L = float(np.linalg.norm(v))
                            if L > 1e-6:
                                n = np.array([-v[1], v[0]], np.float32) / L
                                off = float(np.dot(wpt2 - pa, n))
                                panel._sketch._dim_preview = DimLinear2D(a, b, panel, offset=off)
                else:
                    sel = self._pick_nearest_sketch_vertex(event.position().x(), event.position().y())
                    dt = getattr(panel, '_dim_type', 'linear')
                    panel._sketch._dim_preview = None
                    if dt in ('radius','diameter'):
                        if isinstance(sel, tuple) and sel[0]=='circle' and isinstance(sel[1], Circle2D):
                            if dt=='radius':
                                panel._sketch._dim_preview = DimRadius2D(('circle', sel[1]), panel)
                            else:
                                panel._sketch._dim_preview = DimDiameter2D(('circle', sel[1]), panel)
                    else:
                        a = panel._sketch._dim_pick
                        b = sel if sel is not None else (panel._sketch.hover_snap if panel._sketch.hover_snap is not None else wpt)
                        if a is not None and b is not None:
                            if dt=='horizontal':
                                panel._sketch._dim_preview = DimHorizontal2D(a, b, panel)
                            elif dt=='vertical':
                                panel._sketch._dim_preview = DimVertical2D(a, b, panel)
                            else:
                                panel._sketch._dim_preview = DimLinear2D(a, b, panel)
            self.update()
            # Drag selected vertex (if any)
            if self._sk_dragging and self._sk_point_sel is not None and (event.buttons() & Qt.MouseButton.LeftButton):
                if getattr(panel, '_sketch_snap_grid', False):
                    step = self._grid_step_world_for_panel(panel)
                    w3 = self._screen_to_world_grid_snap(event.position().x(), event.position().y(), step)
                else:
                    w3 = self._screen_to_world_xy(event.position().x(), event.position().y())
                self._move_selected_sketch_vertex(np.array([w3[0], w3[1]], np.float32))
                try:
                    if hasattr(panel, '_update_sketch_info'):
                        panel._update_sketch_info(self._sk_point_sel)
                except Exception:
                    pass
                self.update()
        # Gizmo rotate / scale (invisible gizmo) using horizontal drag
        try:
            panel = self.parent()
        except Exception:
            panel = None
        if panel is not None and hasattr(panel, '_gizmo_mode') and panel._gizmo_mode in ("rotate","scale") and panel._current_sel >= 0:
            if event.buttons() & Qt.MouseButton.LeftButton:  # require LMB drag
                pr = self.scene.prims[panel._current_sel]
                axis = panel._axis_lock or 'z'
                local_space = bool(getattr(panel, '_space_local', True))
                if panel._gizmo_mode == 'rotate':
                    sens_rot = 0.25  # degrees per pixel
                    raw_deg = dx * sens_rot
                    use_snap = not (event.modifiers() & Qt.KeyboardModifier.AltModifier)
                    deg = _snap_angle(raw_deg, panel._snap_angle_deg) if use_snap else raw_deg
                    R = _make_rot(axis, deg)
                    if local_space:
                        pr.xform.M = pr.xform.M @ R
                    else:
                        T = np.eye(4, dtype=np.float32); T[:3,3] = pr.xform.M[:3,3]
                        Tinv = np.eye(4, dtype=np.float32); Tinv[:3,3] = -pr.xform.M[:3,3]
                        pr.xform.M = T @ R @ Tinv @ pr.xform.M
                    try: self.scene._notify()
                    except Exception: pass
                    try: panel._select_prim(panel._current_sel)
                    except Exception: pass
                    self.update(); self._last_mouse = event.position(); return
                elif panel._gizmo_mode == 'scale':
                    sens_scl = 0.005  # scale factor per pixel
                    raw = 1.0 + dx * sens_scl
                    raw = max(0.01, raw)
                    use_snap = not (event.modifiers() & Qt.KeyboardModifier.AltModifier)
                    k = _snap_scale(raw, panel._snap_scale_step) if use_snap else raw
                    S = _make_scale(axis, k)
                    if local_space:
                        pr.xform.M = pr.xform.M @ S
                    else:
                        T = np.eye(4, dtype=np.float32); T[:3,3] = pr.xform.M[:3,3]
                        Tinv = np.eye(4, dtype=np.float32); Tinv[:3,3] = -pr.xform.M[:3,3]
                        pr.xform.M = T @ S @ Tinv @ pr.xform.M
                    try: self.scene._notify()
                    except Exception: pass
                    try: panel._select_prim(panel._current_sel)
                    except Exception: pass
                    self.update(); self._last_mouse = event.position(); return
        # Screen-plane translate when active
        if self._drag_move_active and self._drag_last_pos is not None:
            dx2 = event.position().x() - self._drag_last_pos.x()
            dy2 = event.position().y() - self._drag_last_pos.y()
            R = self._cam_basis(); right = R[:,0]; up = R[:,1]
            step = self.distance * 0.0015
            dp = right * dx2 * step - up * dy2 * step
            pr = self._current_prim()
            if pr is not None:
                new_pos = pr.xform.M[:3,3] + dp.astype(np.float32)
                self._apply_panel_move(new_pos)
            self._drag_last_pos = event.position()
            return
        buttons = event.buttons() or self._last_buttons
        updated = False
        # Orbit (LMB) — disabled if sketch lock top is enabled
        lock_top = False
        try:
            panel = self.parent()
            lock_top = bool(getattr(panel, '_sketch_on', False) and getattr(panel, '_sketch_lock_top', False))
        except Exception:
            lock_top = False
        if (buttons & Qt.MouseButton.LeftButton) and not lock_top:
            sens = 0.005
            self.yaw   += dx * sens
            self.pitch += dy * sens
            # clamp pitch to avoid gimbal flip
            self.pitch = float(np.clip(self.pitch, -1.45, 1.45))
            self._update_camera()
            updated = True
        # Pan (RMB) — disabled if sketch lock top is enabled
        if (buttons & Qt.MouseButton.RightButton) and not lock_top:
            # derive right/up from current basis
            R = self._cam_basis()
            right = R[:,0]; up = R[:,1]
            pan_sens = self.distance * 0.0015
            self.cam_target -= right * dx * pan_sens
            self.cam_target += up * dy * pan_sens
            self._update_camera()
            updated = True
        self._last_mouse = event.position()
        if not updated:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_move_active = False
            self._drag_last_pos = None
            self._sk_dragging = False
        super().mouseReleaseEvent(event)

    def _pick_nearest_sketch_vertex(self, sx: float, sy: float):
        try:
            panel = self.parent()
        except Exception:
            return None
        if panel is None:
            return None
        best = 1e9; best_key = None
        # Search polyline points, line endpoints/midpoints, rect corners, circle centers
        for ent in getattr(panel, '_sketch').entities:
            if isinstance(ent, Polyline2D):
                for idx, pt in enumerate(ent.pts):
                    s = self._world_to_screen(np.array([pt[0], pt[1], 0.0], np.float32))
                    d = float(np.hypot(s[0]-sx, s[1]-sy))
                    if d < best and d <= 12.0:
                        best = d; best_key = (ent, idx)
            elif isinstance(ent, Line2D):
                for idx, pt in enumerate([ent.p0] + ([ent.p1] if ent.p1 is not None else [])):
                    s = self._world_to_screen(np.array([pt[0], pt[1], 0.0], np.float32))
                    d = float(np.hypot(s[0]-sx, s[1]-sy))
                    if d < best and d <= 12.0:
                        best = d; best_key = (ent, idx)
            elif isinstance(ent, Rect2D):
                for idx, pt in enumerate(ent.corners()):
                    s = self._world_to_screen(np.array([pt[0], pt[1], 0.0], np.float32))
                    d = float(np.hypot(s[0]-sx, s[1]-sy))
                    if d < best and d <= 12.0:
                        best = d; best_key = (ent, idx)
            elif isinstance(ent, Circle2D):
                c = ent.center
                s = self._world_to_screen(np.array([c[0], c[1], 0.0], np.float32))
                d = float(np.hypot(s[0]-sx, s[1]-sy))
                if d < best and d <= 12.0:
                    best = d; best_key = ('circle', ent)
        return best_key

    def _move_selected_sketch_vertex(self, new_xy: np.ndarray):
        key = self._sk_point_sel
        if key is None:
            return
        if isinstance(key, tuple) and len(key)==2 and isinstance(key[0], Polyline2D):
            ent, idx = key
            if 0 <= idx < len(ent.pts):
                ent.pts[idx] = new_xy.copy()
        elif isinstance(key, tuple) and key[0] == 'circle' and isinstance(key[1], Circle2D):
            key[1].center = new_xy.copy()

    def _delete_selected_sketch(self) -> bool:
        try:
            panel = self.parent()
        except Exception:
            panel = None
        if panel is None or not getattr(panel, '_sketch_on', False):
            return False
        key = self._sk_point_sel
        # Delete logic: if a polyline vertex is selected, remove it; drop entity if < 2 pts.
        # If a circle center selected, remove the circle entity.
        if key is None:
            # fallback: delete last entity if any
            if panel._sketch.entities:
                panel._sketch.entities.pop()
                return True
            return False
        if isinstance(key, tuple) and len(key)==2 and isinstance(key[0], Polyline2D):
            ent, idx = key
            try:
                if 0 <= idx < len(ent.pts):
                    ent.pts.pop(idx)
                    if len(ent.pts) < 2 and ent in panel._sketch.entities:
                        panel._sketch.entities.remove(ent)
                    self._sk_point_sel = None
                    return True
            except Exception:
                return False
        elif isinstance(key, tuple) and key[0] == 'circle' and isinstance(key[1], Circle2D):
            ent = key[1]
            try:
                if ent in panel._sketch.entities:
                    panel._sketch.entities.remove(ent)
                self._sk_point_sel = None
                return True
            except Exception:
                return False
        return False

    # --- Selection + panel sync helpers ---
    def _current_prim(self):
        try:
            panel = self.parent()
            if hasattr(panel, '_current_sel') and 0 <= panel._current_sel < len(self.scene.prims):
                return self.scene.prims[panel._current_sel]
        except Exception:
            return None
        return None

    def _apply_panel_move(self, new_pos):
        panel = self.parent()
        try:
            if hasattr(panel, '_sp_move'):
                for i,v in enumerate(new_pos):
                    panel._sp_move[i].blockSignals(True)
                    panel._sp_move[i].setValue(float(v))
                    panel._sp_move[i].blockSignals(False)
                if hasattr(panel, '_apply_edit'):
                    panel._apply_edit()
                # Surface position feedback (if main window status bar exists)
                try:
                    mw = panel.window()
                    if hasattr(mw, 'statusBar') and mw.statusBar():
                        mw.statusBar().showMessage(f"Moved prim {getattr(panel,'_current_sel',-1)} to ("+
                            ", ".join(f"{panel._sp_move[i].value():.3f}" for i in range(3))+")", 1500)
                except Exception:
                    pass
        except Exception:
            pass

    def _perform_pick(self, x:int, y:int):
        if not self.prog or self._picking_fbo is None:
            return
        try:
            # Ensure GL context is current for FBO operations outside paintGL
            try:
                self.makeCurrent()
            except Exception:
                pass
            dpr = self.devicePixelRatio() if hasattr(self, 'devicePixelRatio') else 1
            sx = int(x * dpr); sy = int(y * dpr)
            glBindFramebuffer(GL_FRAMEBUFFER, self._picking_fbo)
            glViewport(0,0,self.width(), self.height())
            glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.prog)
            # draw IDs
            glUniform3f(glGetUniformLocation(self.prog,"u_cam_pos"), *self.cam_pos)
            R = self._cam_basis(); glUniformMatrix3fv(glGetUniformLocation(self.prog,"u_cam_rot"), 1, GL_FALSE, R.T)
            glUniform2f(glGetUniformLocation(self.prog,"u_res"), float(self.width()), float(self.height()))
            glUniform3f(glGetUniformLocation(self.prog,"u_env"), *self.scene.env_light)
            glUniform3f(glGetUniformLocation(self.prog,"u_bg"), 0,0,0)
            pack = self.scene.to_gpu_structs(max_prims=MAX_PRIMS)
            n = int(pack['count'])
            glUniform1i(glGetUniformLocation(self.prog,"u_count"), n)
            def U(name): return glGetUniformLocation(self.prog, name)
            glUniform1iv(U("u_kind"), n, pack['kind'])
            glUniform1iv(U("u_op"),   n, pack['op'])
            glUniform1fv(U("u_beta"), n, pack['beta'])
            glUniform3fv(U("u_color"), n, pack['color'])
            glUniform4fv(U("u_params"), n, pack['params'])
            glUniformMatrix4fv(U("u_xform"), n, GL_FALSE, pack['xform'])
            if 'xform_inv' in pack:
                glUniformMatrix4fv(U("u_xform_inv"), n, GL_FALSE, pack['xform_inv'])
            glUniform1i(glGetUniformLocation(self.prog, "u_mode"), 0)
            glUniform1i(glGetUniformLocation(self.prog, "u_debug"), 2)  # ID mode
            glUniform1i(glGetUniformLocation(self.prog, "u_use_analytic_aa"), 0)
            glUniform1i(glGetUniformLocation(self.prog, "u_use_toon"), 0)
            glUniform1f(glGetUniformLocation(self.prog, "u_toon_levels"), 1.0)
            glUniform1f(glGetUniformLocation(self.prog, "u_curv_strength"), 0.0)
            glUniform1i(glGetUniformLocation(self.prog, "u_use_foveated"), 0)
            glUniform1i(glGetUniformLocation(self.prog, "u_show_beta_overlay"), 0)
            glUniform1f(glGetUniformLocation(self.prog, "u_beta_overlay_intensity"), 0.0)
            glUniform1f(glGetUniformLocation(self.prog, "u_beta_scale"), 1.0)
            glUniform1i(glGetUniformLocation(self.prog, "u_beta_cmap"), 0)
            glUniform1i(glGetUniformLocation(self.prog, "u_selected"), -1)
            glBindVertexArray(self._vao); glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            px = np.zeros((1,1,4), dtype=np.uint8)
            ry = self.height() - 1 - (sy if dpr!=1 else y)
            glReadPixels(sx if dpr!=1 else x, ry, 1,1, GL_RGBA, GL_UNSIGNED_BYTE, px)
            enc = int(px[0,0,0]) | (int(px[0,0,1])<<8) | (int(px[0,0,2])<<16)
            if enc==0:
                self.selected_index = -1
            else:
                self.selected_index = enc-1
            glBindFramebuffer(GL_FRAMEBUFFER,0)
            self.update()
            log.debug(f"pick raw={px[0,0].tolist()} enc={enc} sel={self.selected_index} at ({x},{y}) dpr={dpr}")
        except Exception as e:
            log.debug(f"pick failed (full-res) falling back to micro pick: {e}")
            try:
                pid = self._pick_id_at(x,y)
                self.selected_index = pid
            except Exception:
                pass

    def _pick_id_at(self, x:int, y:int) -> int:
        if not self.prog or self._pick_fbo is None:
            return -1
        try:
            # simplistic: render full scene in ID mode into 1x1 buffer (center ray)
            glBindFramebuffer(GL_FRAMEBUFFER, self._pick_fbo)
            glViewport(0,0,1,1)
            glClearColor(0,0,0,1)
            glClear(GL_COLOR_BUFFER_BIT)
            self._render_scene_internal(debug_override=2)
            data = glReadPixels(0,0,1,1,GL_RGBA,GL_UNSIGNED_BYTE)
            import numpy as _np
            px = _np.frombuffer(data, dtype=_np.uint8)
            enc = int(px[0])
            pid = enc-1
            glBindFramebuffer(GL_FRAMEBUFFER,0)
            return pid if 0 <= pid < len(self.scene.prims) else -1
        except Exception as e:
            log.debug(f"micro pick failed: {e}")
            try: glBindFramebuffer(GL_FRAMEBUFFER,0)
            except Exception: pass
            return -1

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y() / 120.0  # 1 per notch
        zoom_factor = 1.0 - delta * 0.1
        zoom_factor = max(0.1, min(2.0, zoom_factor))
        self.distance *= zoom_factor
        self.distance = float(np.clip(self.distance, 0.5, 200.0))
        self._update_camera()
        super().wheelEvent(event)

    def _update_title(self):
        try:
            mode_names = ['beauty','normals','id','depth','thickness']
            m = mode_names[self.debug_mode] if 0<=self.debug_mode < len(mode_names) else str(self.debug_mode)
            cmap_names = ['legacy','viridis','plasma','diverge']
            cm = cmap_names[self.beta_cmap]
            self.window().setWindowTitle(
                f"Analytic Viewport - {m} | AA={'on' if self.use_analytic_aa else 'off'} Toon={'on' if self.use_toon else 'off'} Lvl={int(self.toon_levels)} Curv={self.curv_strength:.1f} FovStep={'on' if self.use_foveated else 'off'} β={'on' if self.show_beta_overlay else 'off'} βI={self.beta_overlay_intensity:.2f} βS={self.beta_scale:.2f} {cm}"
            )
        except Exception:
            pass

    def _shortcuts_help(self):
        return (
            "Shortcuts:\n"
            "0..4 : debug modes (beauty,normals,id,depth,thickness)\n"
            "Space : cycle debug mode\n"
            "A : toggle analytic AA\n"
            "F : toggle foveated steps\n"
            "T : toggle toon\n"
            "+ / - : toon levels up/down\n"
            "C : cycle curvature strength (0,0.5,1.0,0)\n"
            "G : save G-buffers (beauty + normals/id/depth/thickness)\n"
            "Click : select primitive (highlight)\n"
            "H : print this help"
        )

    def save_gbuffers(self):
        # Ensure context current
        self.makeCurrent()
        out_dir = os.path.join(os.getcwd(), 'gbuffers')
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        targets = [(-1,'beauty'), (1,'normals'), (2,'id'), (3,'depth'), (4,'thickness')]
        w,h = self.width(), self.height()
        prev_mode = self.debug_mode
        # Render beauty first (override debug to 0 by setting debug_override=None with debug_mode=0)
        for mode,label in targets:
            if mode==-1:
                # beauty
                self.debug_mode=0
                self._draw_frame(debug_override=0)
            else:
                self._draw_frame(debug_override=mode)
            glFinish()
            data = glReadPixels(0,0,w,h, GL_RGBA, GL_UNSIGNED_BYTE)
            import numpy as _np
            img = _np.frombuffer(data, dtype=_np.uint8).reshape(h, w, 4)
            img = img[::-1,:,:]  # flip vertically
            path = os.path.join(out_dir, f"{timestamp}_{label}.png")
            if _HAVE_PIL:
                Image.fromarray(img, 'RGBA').save(path)
            else:
                # fallback raw dump
                img.tofile(path + '.raw')
            log.info(f"Saved {label} -> {path}")
        self.debug_mode = prev_mode
        self.update()


class AnalyticViewportPanel(QWidget):
    """Composite widget: analytic SDF viewport + control panel."""
    def __init__(self, parent=None, aacore_scene: AACoreScene | None = None):
        super().__init__(parent)
        # Radial wheel defaults (updated before controls are created)
        self._wheel_enabled: bool = True
        self._wheel_edge_fit: bool = True
        self._wheel_edge_overshoot: int = 14
        self._wheel_thickness: float = 0.10
        self._wheel_opacity: float = 1.0
        self._wheel_text_scale: float = 1.0
        self._wheel_auto_spin: bool = False
        self._wheel_auto_spin_speed: float = 0.0
        self._radial_wheel = None
        self._cb_wheel_enabled = None
        self._cb_wheel_edge_fit = None
        self._spin_wheel_overshoot = None
        self._slider_wheel_thickness = None
        self._label_wheel_thickness = None
        self._slider_wheel_opacity = None
        self._label_wheel_opacity = None
        self._slider_wheel_text = None
        self._label_wheel_text = None
        self._cb_wheel_autospin = None
        self._slider_wheel_spin = None
        self._label_wheel_spin = None

        self.view = AnalyticViewport(self, aacore_scene=aacore_scene)
        # --- Sketch layer state (overlay) ---
        self._sketch_on: bool = False
        self._sketch_mode: str | None = None  # 'polyline' or None
        self._sketch = SketchLayer()
        self._sketch_lock_top: bool = True
        # Gizmo / transform state
        self._gizmo_mode = 'move'   # 'move' | 'rotate' | 'scale'
        self._axis_lock = None      # 'x' | 'y' | 'z' | None
        self._space_local = True    # True local, False world
        self._snap_angle_deg = 5.0
        self._snap_scale_step = 0.1
        # References to select UI controls we want to drive programmatically
        self._cb_sketch = None
        self._btn_showcase = None
        self._btn_clear_scene = None
        self._build_ui()
        self.setWindowTitle("Analytic Viewport (Panel)")
        self.view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Ensure the analytic panel is visible on at least one screen.
        # Run after a short delay so top-level windows have a chance to map.
        try:
            QTimer.singleShot(150, self._ensure_on_screen)
        except Exception:
            pass

        # --- Radial Tool Wheel (overlay) ---
        if _HAVE_WHEEL:
            try:
                tools = [
                    ToolSpec("Polyline", "polyline"),
                    ToolSpec("Line", "line"),
                    ToolSpec("Arc 3pt", "arc3"),
                    ToolSpec("Circle", "circle"),
                    ToolSpec("Rect", "rect"),
                    ToolSpec("Insert Vtx", "insert_vertex"),
                    ToolSpec("Dim", "dimension"),
                    ToolSpec("Move", "move"),
                    ToolSpec("Rotate", "rotate"),
                    ToolSpec("Scale", "scale"),
                    ToolSpec("Center", "center_selected"),
                    ToolSpec("Reset", "reset_camera"),
                    ToolSpec("Undo", "delete_last"),
                    ToolSpec("Showcase", "showcase"),
                    ToolSpec("Clear", "clear"),
                ]
                self._radial_wheel = RadialToolWheelOverlay(
                    self.view,
                    tools=tools,
                    outer_radius_ratio=0.46,  # used when fit_to_viewport_edge=False
                    thickness_ratio=0.10,
                    fade_full_at_center_ratio=0.06,
                    fade_full_at_edge_ratio=0.48,
                    idle_rotate_deg_per_sec=0.0,
                    # New: expand diameter so the outer edge sits just
                    # beyond the viewport and is nicely clipped by it.
                    fit_to_viewport_edge=True,
                    edge_overshoot_px=14,
                )
                self._radial_wheel.toolActivated.connect(self._on_wheel_tool)
                self._apply_wheel_settings()
                self._sync_wheel_controls()
            except Exception as e:
                try:
                    log.debug(f"radial wheel init failed: {e}")
                except Exception:
                    pass

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4,4,4,4)
        root.addWidget(self.view, 1)
        # Put the side controls into a dedicated widget so we can wrap
        # them with a QScrollArea to provide vertical scrolling when
        # the panel is taller than the available window height.
        side_widget = QWidget()
        side = QVBoxLayout(side_widget)
        side.setContentsMargins(0,0,0,0)
        side.setSpacing(6)

        # --- Debug Modes ---
        gb_modes = QGroupBox("Debug / Modes"); fm = QFormLayout(); gb_modes.setLayout(fm)
        self.mode_box = QComboBox(); self.mode_box.addItems(["Beauty","Normals","ID","Depth","Thickness"])
        self.mode_box.currentIndexChanged.connect(self._on_mode); fm.addRow("Mode", self.mode_box)
        side.addWidget(gb_modes)

        # --- Feature Toggles ---
        gb_feat = QGroupBox("Features"); vf = QVBoxLayout(); gb_feat.setLayout(vf)
        self.cb_aa = QCheckBox("Analytic AA"); self.cb_aa.setChecked(True); self.cb_aa.stateChanged.connect(self._on_toggle)
        self.cb_fov = QCheckBox("Foveated Steps"); self.cb_fov.setChecked(True); self.cb_fov.stateChanged.connect(self._on_toggle)
        self.cb_toon = QCheckBox("Toon"); self.cb_toon.setChecked(False); self.cb_toon.stateChanged.connect(self._on_toggle)
        for w in (self.cb_aa, self.cb_fov, self.cb_toon): vf.addWidget(w)
        side.addWidget(gb_feat)

        # --- Parameters ---
        gb_sliders = QGroupBox("Parameters"); fs = QFormLayout(); gb_sliders.setLayout(fs)
        self.slider_toon = QSlider(Qt.Orientation.Horizontal); self.slider_toon.setRange(1,8); self.slider_toon.setValue(4); self.slider_toon.valueChanged.connect(self._on_toon_levels)
        self.slider_curv = QSlider(Qt.Orientation.Horizontal); self.slider_curv.setRange(0,100); self.slider_curv.setValue(100); self.slider_curv.valueChanged.connect(self._on_curv)
        self.cb_beta = QCheckBox("β Overlay"); self.cb_beta.setChecked(False); self.cb_beta.stateChanged.connect(self._on_toggle)
        self.slider_beta_int = QSlider(Qt.Orientation.Horizontal); self.slider_beta_int.setRange(0,100); self.slider_beta_int.setValue(int(self.view.beta_overlay_intensity*100)); self.slider_beta_int.valueChanged.connect(self._on_beta_int)
        self.slider_beta_scale = QSlider(Qt.Orientation.Horizontal); self.slider_beta_scale.setRange(1,800); self.slider_beta_scale.setValue(int(self.view.beta_scale*100)); self.slider_beta_scale.valueChanged.connect(self._on_beta_scale)
        self.cmap_box = QComboBox(); self.cmap_box.addItems(["Legacy","Viridis","Plasma","Diverge"]); self.cmap_box.currentIndexChanged.connect(self._on_cmap)
        fs.addRow("Toon Levels", self.slider_toon)
        fs.addRow("Curvature", self.slider_curv)
        fs.addRow(self.cb_beta)
        fs.addRow("β Intensity", self.slider_beta_int)
        fs.addRow("β Scale", self.slider_beta_scale)
        fs.addRow("β Colormap", self.cmap_box)
        side.addWidget(gb_sliders)

        if _HAVE_WHEEL:
            def _make_slider(min_v: int, max_v: int, value: int, formatter):
                row = QWidget()
                hl = QHBoxLayout(row)
                hl.setContentsMargins(0, 0, 0, 0)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(min_v, max_v)
                slider.setValue(value)
                slider.setSingleStep(1)
                slider.setTracking(True)
                label = QLabel()
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                label.setFixedWidth(64)
                def _update_label(val: int) -> None:
                    try:
                        label.setText(formatter(val))
                    except Exception:
                        label.setText(str(val))
                _update_label(value)
                slider.valueChanged.connect(_update_label)
                hl.addWidget(slider, 1)
                hl.addWidget(label)
                return slider, label, row

            gb_wheel = QGroupBox("HUD Wheel")
            fw = QFormLayout()
            gb_wheel.setLayout(fw)

            self._cb_wheel_enabled = QCheckBox("Enabled")
            self._cb_wheel_enabled.setChecked(self._wheel_enabled)
            self._cb_wheel_enabled.toggled.connect(self._on_wheel_enabled)
            fw.addRow(self._cb_wheel_enabled)

            self._cb_wheel_edge_fit = QCheckBox("Fit To Edge")
            self._cb_wheel_edge_fit.setChecked(self._wheel_edge_fit)
            self._cb_wheel_edge_fit.toggled.connect(self._on_wheel_edge_fit)
            fw.addRow(self._cb_wheel_edge_fit)

            self._spin_wheel_overshoot = QSpinBox()
            self._spin_wheel_overshoot.setRange(0, 64)
            self._spin_wheel_overshoot.setValue(int(self._wheel_edge_overshoot))
            self._spin_wheel_overshoot.valueChanged.connect(self._on_wheel_overshoot)
            fw.addRow("Overshoot (px)", self._spin_wheel_overshoot)

            thickness_val = max(6, min(40, int(round(self._wheel_thickness * 100))))
            self._slider_wheel_thickness, self._label_wheel_thickness, row_thickness = _make_slider(6, 40, thickness_val, lambda v: f"{v}%")
            self._slider_wheel_thickness.valueChanged.connect(self._on_wheel_thickness)
            fw.addRow("Thickness (%)", row_thickness)

            opacity_val = max(0, min(100, int(round(self._wheel_opacity * 100))))
            self._slider_wheel_opacity, self._label_wheel_opacity, row_opacity = _make_slider(0, 100, opacity_val, lambda v: f"{v}%")
            self._slider_wheel_opacity.valueChanged.connect(self._on_wheel_opacity)
            fw.addRow("Opacity", row_opacity)

            text_val = max(60, min(200, int(round(self._wheel_text_scale * 100))))
            self._slider_wheel_text, self._label_wheel_text, row_text = _make_slider(60, 200, text_val, lambda v: f"{v/100:.2f}x")
            self._slider_wheel_text.valueChanged.connect(self._on_wheel_text_scale)
            fw.addRow("Text Size", row_text)

            self._cb_wheel_autospin = QCheckBox("Auto Spin")
            self._cb_wheel_autospin.setChecked(self._wheel_auto_spin)
            self._cb_wheel_autospin.toggled.connect(self._on_wheel_autospin)
            fw.addRow(self._cb_wheel_autospin)

            spin_val = max(0, min(360, int(round(self._wheel_auto_spin_speed))))
            self._slider_wheel_spin, self._label_wheel_spin, row_spin = _make_slider(0, 360, spin_val, lambda v: f"{v} deg/s")
            self._slider_wheel_spin.valueChanged.connect(self._on_wheel_spin_speed)
            fw.addRow("Spin Speed", row_spin)

            side.addWidget(gb_wheel)
            self._sync_wheel_controls()

        # --- Gizmo / Transform Controls ---
        gb_gizmo = QGroupBox("Gizmo / Transform"); fg = QFormLayout(); gb_gizmo.setLayout(fg)
        # Mode buttons
        row_modes = QWidget(); hm = QHBoxLayout(row_modes); hm.setContentsMargins(0,0,0,0)
        btn_move = QPushButton("Move"); btn_rot = QPushButton("Rotate [R]"); btn_scl = QPushButton("Scale [S]")
        for b in (btn_move, btn_rot, btn_scl): b.setCheckable(True)
        btn_move.setChecked(True)
        def _set_mode(m):
            def f():
                btn_move.setChecked(m=='move'); btn_rot.setChecked(m=='rotate'); btn_scl.setChecked(m=='scale')
                self._gizmo_mode = m
            return f
        btn_move.clicked.connect(_set_mode('move'))
        btn_rot.clicked.connect(_set_mode('rotate'))
        btn_scl.clicked.connect(_set_mode('scale'))
        hm.addWidget(btn_move); hm.addWidget(btn_rot); hm.addWidget(btn_scl)
        fg.addRow("Mode", row_modes)
        # Axis lock buttons
        row_axes = QWidget(); ha = QHBoxLayout(row_axes); ha.setContentsMargins(0,0,0,0)
        ax_x = QPushButton("X"); ax_y = QPushButton("Y"); ax_z = QPushButton("Z"); ax_clr = QPushButton("Free")
        for b in (ax_x, ax_y, ax_z, ax_clr): b.setCheckable(True)
        ax_clr.setChecked(True)
        def _lock_axis(a):
            def f():
                for b,a2 in ((ax_x,'x'),(ax_y,'y'),(ax_z,'z')):
                    b.setChecked(a==a2)
                ax_clr.setChecked(a is None)
                self._axis_lock = a
            return f
        ax_x.clicked.connect(_lock_axis('x'))
        ax_y.clicked.connect(_lock_axis('y'))
        ax_z.clicked.connect(_lock_axis('z'))
        ax_clr.clicked.connect(_lock_axis(None))
        for b in (ax_x, ax_y, ax_z, ax_clr): ha.addWidget(b)
        fg.addRow("Axis", row_axes)
        # Space toggle
        cb_space = QCheckBox("Local Space"); cb_space.setChecked(True)
        def _space_toggle(_v): self._space_local = cb_space.isChecked()
        cb_space.stateChanged.connect(_space_toggle)
        fg.addRow("Space", cb_space)
        # Snap controls
        spin_ang = QDoubleSpinBox(); spin_ang.setRange(0.5,45.0); spin_ang.setSingleStep(0.5); spin_ang.setValue(self._snap_angle_deg); spin_ang.setSuffix(" °")
        spin_scl = QDoubleSpinBox(); spin_scl.setRange(0.01,10.0); spin_scl.setDecimals(3); spin_scl.setSingleStep(0.05); spin_scl.setValue(self._snap_scale_step)
        def _upd_ang(_v): self._snap_angle_deg = float(spin_ang.value())
        def _upd_scl(_v): self._snap_scale_step = float(spin_scl.value())
        spin_ang.valueChanged.connect(_upd_ang)
        spin_scl.valueChanged.connect(_upd_scl)
        fg.addRow("Angle Snap", spin_ang)
        fg.addRow("Scale Snap", spin_scl)
        side.addWidget(gb_gizmo)

        # --- Sketch (2D) overlay ---
        gb_sk = QGroupBox("Sketch (2D)"); fs = QFormLayout(); gb_sk.setLayout(fs)
        cb_sketch = QCheckBox("Enable Sketch Overlay"); cb_sketch.setChecked(False)
        def _toggle_sk(_):
            self._sketch_on = cb_sketch.isChecked()
            if not self._sketch_on:  # disable & clean up
                self._sketch_mode = None
                self._sketch.active_poly = None
            else:
                if getattr(self, '_sketch_lock_top', False):
                    try:
                        self.view.yaw = 0.0; self.view.pitch = 0.0
                        self.view._update_camera()
                    except Exception:
                        self.view.update()
                else:
                    self.view.update()
        cb_sketch.stateChanged.connect(_toggle_sk)
        # store for programmatic toggling
        self._cb_sketch = cb_sketch
        fs.addRow(cb_sketch)
        # Lock top view while sketching
        row_lock = QWidget(); hlk = QHBoxLayout(row_lock); hlk.setContentsMargins(0,0,0,0)
        cb_lock_top = QCheckBox("Lock Top View (XY)"); cb_lock_top.setChecked(True)
        def _apply_lock_top(_):
            self._sketch_lock_top = cb_lock_top.isChecked()
            if self._sketch_on and self._sketch_lock_top:
                # force yaw/pitch to top view and refresh
                try:
                    self.view.yaw = 0.0; self.view.pitch = 0.0
                    self.view._update_camera()
                except Exception:
                    pass
            else:
                self.view.update()
        cb_lock_top.stateChanged.connect(_apply_lock_top)
        hlk.addWidget(cb_lock_top)
        fs.addRow("View Lock", row_lock)
        # Grid & Snap
        self._sketch_grid_on = False
        self._sketch_snap_grid = False
        self._sketch_grid_step = 10.0
        # If True: interpret grid_step as pixel spacing (screen-locked); if False: world units
        self._sketch_grid_screen_lock = True
        row_grid = QWidget(); hg = QHBoxLayout(row_grid); hg.setContentsMargins(0,0,0,0)
        cb_grid = QCheckBox("Show Grid"); cb_grid.stateChanged.connect(lambda _v: setattr(self, '_sketch_grid_on', cb_grid.isChecked()) or self.view.update())
        cb_snapg= QCheckBox("Snap Grid"); cb_snapg.stateChanged.connect(lambda _v: setattr(self, '_sketch_snap_grid', cb_snapg.isChecked()))
        # Toggle: screen-lock (pixels) vs world units
        cb_lock = QCheckBox("Screen-locked spacing"); cb_lock.setChecked(self._sketch_grid_screen_lock)
        cb_lock.stateChanged.connect(lambda _v: setattr(self, '_sketch_grid_screen_lock', cb_lock.isChecked()) or self.view.update())
        sp_grid = QDoubleSpinBox(); sp_grid.setRange(0.1, 1000.0); sp_grid.setDecimals(2); sp_grid.setSingleStep(0.5); sp_grid.setValue(self._sketch_grid_step)
        sp_grid.valueChanged.connect(lambda v: setattr(self, '_sketch_grid_step', float(v)))
        hg.addWidget(cb_grid); hg.addWidget(cb_snapg); hg.addWidget(cb_lock); hg.addWidget(QLabel("Step")); hg.addWidget(sp_grid)
        fs.addRow("Grid/Snap", row_grid)
        # Kernel Sync (optional)
        row_sync = QWidget(); hsync = QHBoxLayout(row_sync); hsync.setContentsMargins(0,0,0,0)
        self._sketch_circle_to_ring = False
        cb_circle_sync = QCheckBox("Circle → Ring (kernel)")
        cb_circle_sync.setToolTip("When enabled, finalizing a sketch circle adds a thin torus at the same center/radius in the AACore scene.")
        cb_circle_sync.stateChanged.connect(lambda _v: setattr(self, '_sketch_circle_to_ring', cb_circle_sync.isChecked()))
        hsync.addWidget(cb_circle_sync)
        fs.addRow("Sync", row_sync)
        row_tools = QWidget(); ht = QHBoxLayout(row_tools); ht.setContentsMargins(0,0,0,0)
        btn_poly = QPushButton("Polyline"); btn_poly.setCheckable(True)
        btn_line = QPushButton("Line"); btn_line.setCheckable(True)
        btn_arc3 = QPushButton("Arc (3-pt)"); btn_arc3.setCheckable(True)
        btn_circle = QPushButton("Circle"); btn_circle.setCheckable(True)
        btn_rect = QPushButton("Rectangle"); btn_rect.setCheckable(True)
        btn_insv = QPushButton("Insert Vertex"); btn_insv.setCheckable(True)
        def _choose_poly():
            on = btn_poly.isChecked()
            if on:
                for b2 in (btn_line, btn_arc3, btn_circle, btn_rect): b2.setChecked(False)
                self._sketch_mode = 'polyline'
                self._sketch.active_shape = None
                if getattr(self, '_sketch_lock_top', False):
                    try: self.view.yaw=0.0; self.view.pitch=0.0; self.view._update_camera()
                    except Exception: pass
            else:
                if self._sketch_mode=='polyline':
                    self._sketch_mode = None
            self.view.update()
        btn_poly.clicked.connect(_choose_poly)
        def _choose_line():
            on = btn_line.isChecked()
            if on:
                for b2 in (btn_poly, btn_arc3, btn_circle, btn_rect): b2.setChecked(False)
                self._sketch_mode = 'line'
                self._sketch.active_poly = None
                self._sketch.active_shape = None
                if getattr(self, '_sketch_lock_top', False):
                    try: self.view.yaw=0.0; self.view.pitch=0.0; self.view._update_camera()
                    except Exception: pass
            else:
                if self._sketch_mode=='line': self._sketch_mode = None
            self.view.update()
        btn_line.clicked.connect(_choose_line)
        def _choose_arc3():
            on = btn_arc3.isChecked()
            if on:
                for b2 in (btn_poly, btn_line, btn_circle, btn_rect): b2.setChecked(False)
                self._sketch_mode = 'arc3'
                self._sketch.active_poly = None
                self._sketch.active_shape = None
                if getattr(self, '_sketch_lock_top', False):
                    try: self.view.yaw=0.0; self.view.pitch=0.0; self.view._update_camera()
                    except Exception: pass
            else:
                if self._sketch_mode=='arc3': self._sketch_mode = None
            self.view.update()
        btn_arc3.clicked.connect(_choose_arc3)
        def _choose_circle():
            on = btn_circle.isChecked()
            if on:
                for b2 in (btn_poly, btn_line, btn_arc3, btn_rect): b2.setChecked(False)
                self._sketch_mode = 'circle'
                self._sketch.active_poly = None
                self._sketch.active_shape = None
                if getattr(self, '_sketch_lock_top', False):
                    try: self.view.yaw=0.0; self.view.pitch=0.0; self.view._update_camera()
                    except Exception: pass
            else:
                if self._sketch_mode=='circle': self._sketch_mode = None
            self.view.update()
        btn_circle.clicked.connect(_choose_circle)
        def _choose_rect():
            on = btn_rect.isChecked()
            if on:
                for b2 in (btn_poly, btn_line, btn_arc3, btn_circle): b2.setChecked(False)
                self._sketch_mode = 'rect'
                self._sketch.active_poly = None
                self._sketch.active_shape = None
                if getattr(self, '_sketch_lock_top', False):
                    try: self.view.yaw=0.0; self.view.pitch=0.0; self.view._update_camera()
                    except Exception: pass
            else:
                if self._sketch_mode=='rect': self._sketch_mode = None
            self.view.update()
        btn_rect.clicked.connect(_choose_rect)
        # Insert Vertex / Midpoint option
        self._insert_midpoint = False
        cb_ins_mid = QCheckBox("Midpoint"); cb_ins_mid.stateChanged.connect(lambda _v: setattr(self, '_insert_midpoint', cb_ins_mid.isChecked()))
        def _choose_insv():
            on = btn_insv.isChecked()
            if on:
                for b2 in (btn_poly, btn_line, btn_arc3, btn_circle, btn_rect): b2.setChecked(False)
                self._sketch_mode = 'insert_vertex'
                self._sketch._dim_pick = None
            else:
                if self._sketch_mode=='insert_vertex': self._sketch_mode = None
            self.view.update()
        btn_insv.clicked.connect(_choose_insv)
        # Dimension tool
        btn_dim = QPushButton("Dimension"); btn_dim.setCheckable(True)
        def _choose_dim():
            on = btn_dim.isChecked()
            if on:
                for b2 in (btn_poly, btn_line, btn_arc3, btn_circle, btn_rect): b2.setChecked(False)
                self._sketch_mode = 'dimension'
                self._sketch._dim_pick = None
            else:
                if self._sketch_mode=='dimension':
                    self._sketch_mode = None
                    # reset any in-progress placing state
                    try:
                        self._sketch._dim_placing = False; self._sketch._dim_place_data = None; self._sketch._dim_preview = None
                    except Exception:
                        pass
            self.view.update()
        btn_dim.clicked.connect(_choose_dim)
        for b in (btn_poly, btn_line, btn_arc3, btn_circle, btn_rect, btn_insv, btn_dim, cb_ins_mid):
            ht.addWidget(b)
        fs.addRow("Tool", row_tools)
        # Dimension settings (type + precision)
        row_dim = QWidget(); hdim = QHBoxLayout(row_dim); hdim.setContentsMargins(0,0,0,0)
        from PySide6.QtWidgets import QComboBox as _QCB
        self._dim_type = 'linear'
        cb_dim_type = _QCB(); cb_dim_type.addItems(["Linear","Horizontal","Vertical","Radius","Diameter"])
        def _set_dim_type(idx):
            self._dim_type = [ 'linear','horizontal','vertical','radius','diameter' ][max(0,min(4,idx))]
        cb_dim_type.currentIndexChanged.connect(_set_dim_type)
        self._dim_precision = 2
        sp_dim_prec = QSpinBox(); sp_dim_prec.setRange(0,6); sp_dim_prec.setValue(self._dim_precision)
        sp_dim_prec.valueChanged.connect(lambda v: setattr(self, '_dim_precision', int(v)))
        hdim.addWidget(QLabel("Type")); hdim.addWidget(cb_dim_type)
        hdim.addWidget(QLabel("Precision")); hdim.addWidget(sp_dim_prec)
        fs.addRow("Dimensions", row_dim)
        # Selection info label
        self._sketch_info_label = QLabel("No selection")
        fs.addRow("Selection", self._sketch_info_label)
        # Save/Load Sketch JSON
        row_io = QWidget(); hio = QHBoxLayout(row_io); hio.setContentsMargins(0,0,0,0)
        btn_save_sk = QPushButton("Save Sketch JSON")
        btn_load_sk = QPushButton("Load Sketch JSON")
        btn_save_sk.clicked.connect(self._save_sketch_json)
        btn_load_sk.clicked.connect(self._load_sketch_json)
        # Clear button
        btn_clear_sk = QPushButton("Clear Sketch")
        def _clear_sk():
            self._sketch.entities.clear(); self._sketch.active_poly=None; self._sketch.active_shape=None
            self.view._sk_point_sel=None; self.view._sk_dragging=False
            self._update_sketch_info(None)
            self.view.update()
        btn_clear_sk.clicked.connect(_clear_sk)
        hio.addWidget(btn_save_sk); hio.addWidget(btn_load_sk)
        hio.addWidget(btn_clear_sk)
        fs.addRow("IO", row_io)
        side.addWidget(gb_sk)
        # Backing model
        self._sketch_doc = SketchDocument(units=Units.MM)

        # --- Actions ---
        gb_act = QGroupBox("Actions"); va = QVBoxLayout(); gb_act.setLayout(va)
        btn_save = QPushButton("Save G-Buffers [G]"); btn_save.setEnabled(False)
        btn_help = QPushButton("Print Shortcuts [H]"); btn_help.clicked.connect(lambda: log.info(self.view._shortcuts_help()))
        btn_showcase = QPushButton("Showcase Primitives")
        btn_clear_scene = QPushButton("Clear Scene")
        def _do_showcase():
            try:
                # Add a small curated set if budget allows
                if len(self.view.scene.prims) < MAX_PRIMS - 6:
                    self.view.scene.add(Prim(KIND_SPHERE, [0.7,0,0,0], beta=0.08, color=(0.95,0.45,0.35)))
                    self.view.scene.add(Prim(KIND_TORUS, [1.0,0.22,0,0], beta=0.03, color=(0.6,0.85,0.5)))
                    self.view.scene.add(Prim(KIND_MOBIUS, [1.4,0.25,0,0], beta=0.02, color=(0.7,0.55,0.95)))
                    self.view.scene.add(Prim(KIND_SUPERELLIPSOID, [1.1,3.5,0,0], beta=0.0, color=(0.95,0.9,0.85)))
                    self.view.scene.add(Prim(KIND_GYROID, [2.2,0.0,0.035,0], beta=0.0, color=(0.65,0.9,0.3)))
                    self.view.scene.add(Prim(KIND_TREFOIL, [0.8,0.08,96,0], beta=0.0, color=(0.95,0.6,0.25)))
                    # Offset for variety
                    try:
                        for i, pr in enumerate(self.view.scene.prims[-6:]):
                            pr.xform.M[:3,3] += np.array([ (i%3-1)*1.6, (i//3-0.5)*1.4, 0.0 ], np.float32)
                    except Exception:
                        pass
                    try: self.view.scene._notify()
                    except Exception: pass
                    self.view.update()
            except Exception as e:
                log.debug(f"showcase failed: {e}")
        def _do_clear_scene():
            try:
                self.view.scene.prims.clear()
                try: self.view.scene._notify()
                except Exception: pass
                try: self.view.clear_polyline_overlays()
                except Exception: pass
                self._refresh_prim_label(); self.view.update()
            except Exception:
                pass
        btn_showcase.clicked.connect(_do_showcase)
        btn_clear_scene.clicked.connect(_do_clear_scene)
        # keep references so the wheel can trigger them
        self._btn_showcase = btn_showcase
        self._btn_clear_scene = btn_clear_scene
        va.addWidget(btn_save); va.addWidget(btn_help); va.addWidget(btn_showcase); va.addWidget(btn_clear_scene); side.addWidget(gb_act)

        # --- Environment ---
        gb_env = QGroupBox("Environment"); fe = QFormLayout(); gb_env.setLayout(fe)
        self._bg_sliders = []
        for i, ch in enumerate(['R','G','B']):
            s = QSlider(Qt.Orientation.Horizontal); s.setRange(0,255); s.setValue(int(self.view.scene.bg_color[i]*255)); s.valueChanged.connect(self._on_bg_changed)
            fe.addRow(f"BG {ch}", s); self._bg_sliders.append(s)
        self._env_sliders = []
        for i, ch in enumerate(['X','Y','Z']):
            s = QSlider(Qt.Orientation.Horizontal); s.setRange(-150,150); s.setValue(int(self.view.scene.env_light[i]*100)); s.valueChanged.connect(self._on_env_changed)
            fe.addRow(f"Light {ch}", s); self._env_sliders.append(s)
        side.addWidget(gb_env)

        # --- Primitives + Editing ---
        gb_prims = QGroupBox("Primitives"); vp = QVBoxLayout(); gb_prims.setLayout(vp)
        self._prim_list_label = QLabel("(0)")
        self._prim_buttons_box = QVBoxLayout()
        btn_add_sphere = QPushButton("Add Sphere"); btn_add_sphere.clicked.connect(lambda: self._add_prim('sphere'))
        btn_add_box = QPushButton("Add Box"); btn_add_box.clicked.connect(lambda: self._add_prim('box'))
        btn_add_capsule = QPushButton("Add Capsule"); btn_add_capsule.clicked.connect(lambda: self._add_prim('capsule'))
        btn_add_torus = QPushButton("Add Torus"); btn_add_torus.clicked.connect(lambda: self._add_prim('torus'))
        btn_add_mobius = QPushButton("Add Mobius"); btn_add_mobius.clicked.connect(lambda: self._add_prim('mobius'))
        btn_add_superell = QPushButton("Add Superellipsoid"); btn_add_superell.clicked.connect(lambda: self._add_prim('superellipsoid'))
        btn_add_qc = QPushButton("Add QuasiCrystal"); btn_add_qc.clicked.connect(lambda: self._add_prim('quasicrystal'))
        btn_add_torus4d = QPushButton("Add 4D Torus"); btn_add_torus4d.clicked.connect(lambda: self._add_prim('torus4d'))
        btn_add_mandelbulb = QPushButton("Add Mandelbulb"); btn_add_mandelbulb.clicked.connect(lambda: self._add_prim('mandelbulb'))
        btn_add_klein = QPushButton("Add Klein Bottle"); btn_add_klein.clicked.connect(lambda: self._add_prim('klein'))
        btn_add_menger = QPushButton("Add Menger Sponge"); btn_add_menger.clicked.connect(lambda: self._add_prim('menger'))
        btn_add_hyperbolic = QPushButton("Add Hyperbolic Tiling"); btn_add_hyperbolic.clicked.connect(lambda: self._add_prim('hyperbolic'))
        btn_add_gyroid = QPushButton("Add Gyroid"); btn_add_gyroid.clicked.connect(lambda: self._add_prim('gyroid'))
        btn_add_trefoil = QPushButton("Add Trefoil Knot"); btn_add_trefoil.clicked.connect(lambda: self._add_prim('trefoil'))
        btn_del_last = QPushButton("Delete Last"); btn_del_last.clicked.connect(self._del_last)
        btn_dup_sel = QPushButton("Duplicate Selected"); btn_dup_sel.clicked.connect(self._duplicate_selected)
        btn_center_sel = QPushButton("Center On Selected"); btn_center_sel.clicked.connect(self._center_on_selected)
        btn_reset_cam = QPushButton("Reset Camera"); btn_reset_cam.clicked.connect(self._reset_camera)
        btn_export = QPushButton("Export Scene JSON"); btn_export.clicked.connect(self._export_scene_json)
        for b in (self._prim_list_label, btn_add_sphere, btn_add_box, btn_add_capsule, btn_add_torus, btn_add_mobius, btn_add_superell, btn_add_qc, btn_add_torus4d, btn_add_mandelbulb, btn_add_klein, btn_add_menger, btn_add_hyperbolic, btn_add_gyroid, btn_add_trefoil, btn_del_last):
            vp.addWidget(b)
        for b in (btn_dup_sel, btn_center_sel, btn_reset_cam, btn_export):
            vp.addWidget(b)
        vb_holder = QWidget(); vb_holder.setLayout(self._prim_buttons_box); vp.addWidget(vb_holder)
        # Edit group
        self._edit_group = QGroupBox("Edit Selected"); fe2 = QFormLayout(); self._edit_group.setLayout(fe2)
        from PySide6.QtWidgets import QComboBox as _QCB2, QColorDialog
        def dspin(r=(-10,10), step=0.01, val=0.0):
            sp = QDoubleSpinBox(); sp.setRange(r[0], r[1]); sp.setSingleStep(step); sp.setDecimals(4); sp.setValue(val); return sp
        self._sp_pos = [dspin() for _ in range(3)]
        self._sp_param = [dspin((0,10),0.01,0.5) for _ in range(2)]
        self._sp_beta = dspin((-1,1),0.01,0.0)
        self._btn_color = QPushButton("Color…")
        self._op_box = _QCB2(); self._op_box.addItems(["solid","subtract"])
        fe2.addRow("Pos X/Y/Z", self._make_row(self._sp_pos))
        fe2.addRow("Param A/B", self._make_row(self._sp_param))
        fe2.addRow("Beta", self._sp_beta)
        fe2.addRow("Color", self._btn_color)
        fe2.addRow("Op", self._op_box)
        # Rotation & Scale controls
        def rspin():
            s = QDoubleSpinBox(); s.setRange(-180.0,180.0); s.setSingleStep(1.0); s.setDecimals(2); s.setValue(0.0); return s
        def sspin():
            s = QDoubleSpinBox(); s.setRange(0.01,100.0); s.setSingleStep(0.05); s.setDecimals(3); s.setValue(1.0); return s
        self._sp_rot = [rspin() for _ in range(3)]
        self._sp_scl = [sspin() for _ in range(3)]
        # Uniform scale control: sets X/Y/Z together
        self._sp_scl_u = sspin()
        # Move (translate) controls
        def mspin():
            s = QDoubleSpinBox(); s.setRange(-1000.0,1000.0); s.setSingleStep(0.01); s.setDecimals(4); s.setValue(0.0); return s
        self._sp_move = [mspin() for _ in range(3)]
        fe2.addRow("Rot ° X/Y/Z", self._make_row(self._sp_rot))
        fe2.addRow("Scale X/Y/Z", self._make_row(self._sp_scl))
        def _set_uniform_scale(v: float):
            try:
                for sp in self._sp_scl:
                    sp.blockSignals(True)
                    sp.setValue(float(v))
                    sp.blockSignals(False)
                # Apply after setting all axes
                self._apply_edit()
            except Exception:
                pass
        self._sp_scl_u.valueChanged.connect(_set_uniform_scale)
        fe2.addRow("Scale (Uniform)", self._sp_scl_u)
        fe2.addRow("Move X/Y/Z", self._make_row(self._sp_move))
        self._nudge_step = QDoubleSpinBox(); self._nudge_step.setRange(0.001,100.0); self._nudge_step.setDecimals(3); self._nudge_step.setSingleStep(0.1); self._nudge_step.setValue(1.0)
        fe2.addRow("Nudge Step", self._nudge_step)
        self._edit_group.setEnabled(False); vp.addWidget(self._edit_group)
        for sp in self._sp_pos: sp.valueChanged.connect(self._apply_edit)
        for sp in self._sp_param: sp.valueChanged.connect(self._apply_edit)
        self._sp_beta.valueChanged.connect(self._apply_edit); self._op_box.currentIndexChanged.connect(self._apply_edit)
        for s in self._sp_rot + self._sp_scl + self._sp_move:
            s.valueChanged.connect(self._apply_edit)
        def pick_color():
            from PySide6.QtGui import QColor
            c = QColorDialog.getColor(QColor(200,180,160), self, "Primitive Color")
            if c.isValid():
                self._current_color = (c.red()/255.0, c.green()/255.0, c.blue()/255.0)
                self._apply_edit()
        self._btn_color.clicked.connect(pick_color)
        self._current_sel = -1; self._current_color = (0.8,0.7,0.6)
        side.addWidget(gb_prims)

        # --- Tesseract (4D hypercube) ---
        gb_tess = QGroupBox("Tesseract (4D Hypercube)"); ft = QFormLayout(); gb_tess.setLayout(ft)
        self._tess_edge = QDoubleSpinBox(); self._tess_edge.setRange(0.1, 10.0); self._tess_edge.setDecimals(3); self._tess_edge.setSingleStep(0.1); self._tess_edge.setValue(2.0)
        self._tess_edge_rad = QDoubleSpinBox(); self._tess_edge_rad.setRange(0.001, 1.0); self._tess_edge_rad.setDecimals(3); self._tess_edge_rad.setSingleStep(0.01); self._tess_edge_rad.setValue(0.03)
        self._tess_node_rad = QDoubleSpinBox(); self._tess_node_rad.setRange(0.001, 1.0); self._tess_node_rad.setDecimals(3); self._tess_node_rad.setSingleStep(0.01); self._tess_node_rad.setValue(0.04)
        self._tess_include_nodes = QCheckBox("Include Nodes (spheres)"); self._tess_include_nodes.setChecked(False)
        # Projection
        self._tess_proj = QComboBox(); self._tess_proj.addItems(["Orthographic","Perspective"])
        self._tess_fw = QDoubleSpinBox(); self._tess_fw.setRange(0.5, 10.0); self._tess_fw.setDecimals(2); self._tess_fw.setSingleStep(0.1); self._tess_fw.setValue(3.0)
        # 4D rotation angles
        def r4spin():
            s = QDoubleSpinBox(); s.setRange(-360.0, 360.0); s.setDecimals(2); s.setSingleStep(1.0); s.setValue(0.0); return s
        self._tess_ang_xy = r4spin(); self._tess_ang_xz = r4spin(); self._tess_ang_xw = r4spin();
        self._tess_ang_yz = r4spin(); self._tess_ang_yw = r4spin(); self._tess_ang_zw = r4spin();
        self._tess_autospin = QCheckBox("Auto-Spin"); self._tess_autospin.setChecked(True)
        self._tess_speed = QDoubleSpinBox(); self._tess_speed.setRange(0.0, 360.0); self._tess_speed.setDecimals(2); self._tess_speed.setSingleStep(5.0); self._tess_speed.setValue(30.0)
        btn_tess = QPushButton("Add Tesseract")
        btn_tess.clicked.connect(self._add_tesseract)
        # Layout
        ft.addRow("Edge Length", self._tess_edge)
        ft.addRow("Edge Radius", self._tess_edge_rad)
        ft.addRow("Node Radius", self._tess_node_rad)
        ft.addRow(self._tess_include_nodes)
        ft.addRow("Projection", self._tess_proj)
        ft.addRow("Focal (persp fw)", self._tess_fw)
        row_ang1 = self._make_row([QLabel("XY"), self._tess_ang_xy, QLabel("XZ"), self._tess_ang_xz, QLabel("XW"), self._tess_ang_xw])
        row_ang2 = self._make_row([QLabel("YZ"), self._tess_ang_yz, QLabel("YW"), self._tess_ang_yw, QLabel("ZW"), self._tess_ang_zw])
        ft.addRow("Angles 1", row_ang1)
        ft.addRow("Angles 2", row_ang2)
        row_spin = self._make_row([self._tess_autospin, QLabel("Speed (deg/s)"), self._tess_speed])
        ft.addRow("Spin", row_spin)
        ft.addRow(btn_tess)
        side.addWidget(gb_tess)

        # rigs & timer
        self._rigs = []
        self._rig_timer = QTimer(self); self._rig_timer.timeout.connect(self._update_rigs)
        self._rig_timer.start(33)

        # --- Scale Journey (Zoom) ---
        gb_zoom = QGroupBox("Scale Journey (Zoom)"); fz = QFormLayout(); gb_zoom.setLayout(fz)
        self._zoom_d0 = QDoubleSpinBox(); self._zoom_d0.setRange(0.1, 1000.0); self._zoom_d0.setDecimals(3)
        self._zoom_d0.setSingleStep(0.1); self._zoom_d0.setValue(float(getattr(self.view, 'distance', 6.0)))
        self._zoom_d1 = QDoubleSpinBox(); self._zoom_d1.setRange(0.01, 1000.0); self._zoom_d1.setDecimals(3)
        self._zoom_d1.setSingleStep(0.01); self._zoom_d1.setValue(0.6)
        self._zoom_dur = QDoubleSpinBox(); self._zoom_dur.setRange(0.2, 120.0); self._zoom_dur.setDecimals(1)
        self._zoom_dur.setSingleStep(0.5); self._zoom_dur.setValue(8.0)
        self._zoom_spawn = QCheckBox("Spawn molecule at end"); self._zoom_spawn.setChecked(True)
        from PySide6.QtWidgets import QComboBox as _QCB3
        self._zoom_mol_select = _QCB3(); self._zoom_mol_select.addItems(["Water", "Methane", "Choose File…", "None"]) ; self._zoom_mol_path = None
        def _choose_custom(idx:int):
            txt = self._zoom_mol_select.currentText()
            if txt == "Choose File…":
                try:
                    from PySide6.QtWidgets import QFileDialog
                    p, _ = QFileDialog.getOpenFileName(self, "Choose Molecule (XYZ)", filter="XYZ (*.xyz)")
                    if p:
                        self._zoom_mol_path = p
                        # keep selection as Choose File…; path stored
                except Exception:
                    pass
        self._zoom_mol_select.currentIndexChanged.connect(_choose_custom)
        btn_start_zoom = QPushButton("Start Zoom")
        btn_start_zoom.clicked.connect(self._start_zoom_journey)
        fz.addRow("Start Distance", self._zoom_d0)
        fz.addRow("End Distance", self._zoom_d1)
        fz.addRow("Duration (s)", self._zoom_dur)
        row_msel = self._make_row([self._zoom_spawn, QLabel("Molecule"), self._zoom_mol_select])
        fz.addRow("At End", row_msel)
        fz.addRow(btn_start_zoom)
        side.addWidget(gb_zoom)

        # --- Molecules ---
        gb_mol = QGroupBox("Molecules"); fmol = QFormLayout(); gb_mol.setLayout(fmol)
        self._mol_atom_scale = QDoubleSpinBox(); self._mol_atom_scale.setRange(0.1, 5.0); self._mol_atom_scale.setSingleStep(0.1); self._mol_atom_scale.setValue(0.6)
        self._mol_bond_r = QDoubleSpinBox(); self._mol_bond_r.setRange(0.005, 1.0); self._mol_bond_r.setDecimals(3); self._mol_bond_r.setSingleStep(0.005); self._mol_bond_r.setValue(0.035)
        self._mol_bond_thresh = QDoubleSpinBox(); self._mol_bond_thresh.setRange(0.0, 1.0); self._mol_bond_thresh.setDecimals(3); self._mol_bond_thresh.setSingleStep(0.02); self._mol_bond_thresh.setValue(0.2)
        self._mol_autospin = QCheckBox("Auto-Spin"); self._mol_autospin.setChecked(True)
        self._mol_speed = QDoubleSpinBox(); self._mol_speed.setRange(0.0, 360.0); self._mol_speed.setDecimals(1); self._mol_speed.setSingleStep(5.0); self._mol_speed.setValue(25.0)
        btn_imp_xyz = QPushButton("Import XYZ…")
        btn_imp_xyz.clicked.connect(self._import_xyz_molecule)
        fmol.addRow("Atom Scale", self._mol_atom_scale)
        fmol.addRow("Bond Radius", self._mol_bond_r)
        fmol.addRow("Bond Thresh", self._mol_bond_thresh)
        row_spinmol = self._make_row([self._mol_autospin, QLabel("Speed"), self._mol_speed])
        fmol.addRow("Spin", row_spinmol)
        fmol.addRow(btn_imp_xyz)
        side.addWidget(gb_mol)

        # finalize
        side.addStretch(1)
        # Wrap the side widget in a scroll area so overflowing controls
        # become scrollable instead of going off-screen.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(side_widget)
        # Compute a comfortable minimum width so horizontal scrolling is unnecessary
        try:
            content_w = int(side_widget.sizeHint().width()) + 28
        except Exception:
            content_w = 480
        # Widen clamp so horizontal scroll is unnecessary and all controls fit
        content_w = max(420, min(content_w, 860))
        scroll.setMinimumWidth(content_w)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        root.addWidget(scroll)
        self._refresh_prim_label()

    # --- Tesseract helpers ---
    def _add_tesseract(self):
        L = float(self._tess_edge.value())
        er = float(self._tess_edge_rad.value())
        nr = float(self._tess_node_rad.value())
        include_nodes = bool(self._tess_include_nodes.isChecked())
        proj = self._tess_proj.currentIndex()  # 0=ortho,1=persp
        fw = float(self._tess_fw.value())
        verts4 = self._tesseract_vertices4(L)
        edges = self._tesseract_edges()
        # Prim budget
        need = len(edges) + (len(verts4) if include_nodes else 0)
        if len(self.view.scene.prims) + need > MAX_PRIMS:
            log.info(f"[tesseract] Not enough prim budget: need {need}, have {MAX_PRIMS-len(self.view.scene.prims)}")
            return
        # Create prims
        color_edge = (0.9, 0.8, 0.35)
        color_node = (0.85, 0.55, 0.2)
        # Initial angles
        angs = { 'xy': float(self._tess_ang_xy.value()), 'xz': float(self._tess_ang_xz.value()), 'xw': float(self._tess_ang_xw.value()),
                 'yz': float(self._tess_ang_yz.value()), 'yw': float(self._tess_ang_yw.value()), 'zw': float(self._tess_ang_zw.value()) }
        pidx_edges = []
        for (i,j) in edges:
            pr = Prim(KIND_CAPSULE, [er, 1.0, 0.0, 0.0], beta=0.0, color=color_edge)
            self.view.scene.add(pr)
            pidx_edges.append(len(self.view.scene.prims)-1)
        pidx_nodes = []
        if include_nodes:
            for _ in verts4:
                pr = Prim(KIND_SPHERE, [nr, 0.0, 0.0, 0.0], beta=0.0, color=color_node)
                self.view.scene.add(pr)
                pidx_nodes.append(len(self.view.scene.prims)-1)
        rig = {
            'type':'tesseract', 'edges':edges, 'verts4':verts4,
            'pidx_edges':pidx_edges, 'pidx_nodes':pidx_nodes,
            'edge_r':er, 'node_r':nr, 'L':L,
            'proj':proj, 'fw':fw,
            'angs':angs,
            'autospin': bool(self._tess_autospin.isChecked()), 'speed': float(self._tess_speed.value())
        }
        self._rigs.append(rig)
        self._refresh_prim_label(); self.view.update()
        self._update_tesseract_rig(rig)

    def _update_rigs(self):
        if not self._rigs:
            return
        dt = 0.033
        for rig in self._rigs:
            try:
                if rig.get('type') == 'tesseract':
                    if rig.get('autospin', False):
                        sp = float(rig.get('speed', 0.0))
                        for k in ('xy','xz','xw','yz','yw','zw'):
                            rig['angs'][k] = float(rig['angs'][k]) + sp*dt
                    self._update_tesseract_rig(rig)
                elif rig.get('type') == 'molecule':
                    if rig.get('autospin', False):
                        rig['angle'] = float(rig.get('angle', 0.0)) + float(rig.get('speed', 0.0)) * dt
                    
                    # Handle molecule growth animation (for dramatic spawn)
                    if 'growth' in rig:
                        growth = rig['growth']
                        growth['start_time'] += dt
                        t = growth['start_time']
                        dur = growth['duration']
                        
                        if t < dur:
                            # Smooth growth curve
                            k = t / dur
                            kk = k * k * (3.0 - 2.0 * k)  # smoothstep
                            
                            # Scale atoms and bonds from tiny to full size
                            target_atom = growth['target_atom_scale']
                            target_bond = growth['target_bond_r']
                            target_speed = growth['target_speed']
                            
                            current_atom_scale = 0.1 * target_atom * (1.0 - kk) + target_atom * kk
                            current_bond_r = 0.1 * target_bond * (1.0 - kk) + target_bond * kk
                            current_speed = 60.0 * (1.0 - kk) + target_speed * kk
                            
                            # Update molecule size in real-time
                            for pidx in rig.get('pidx_atoms', []):
                                if pidx < len(self.view.scene.prims):
                                    pr = self.view.scene.prims[pidx]
                                    # Scale radius based on original atom properties
                                    base_r = pr.params[0] / (current_atom_scale / target_atom) if target_atom > 0 else 0.5
                                    pr.params[0] = float(base_r * current_atom_scale / target_atom)
                            
                            for pidx in rig.get('pidx_bonds', []):
                                if pidx < len(self.view.scene.prims):
                                    pr = self.view.scene.prims[pidx]
                                    pr.params[0] = float(current_bond_r)
                            
                            rig['speed'] = current_speed
                        else:
                            # Growth complete
                            del rig['growth']
                    
                    self._update_molecule_rig(rig)
                elif rig.get('type') == 'zoom':
                    t = rig.get('t', 0.0) + dt
                    dur = max(1e-3, float(rig.get('dur', 1.0)))
                    k = min(1.0, t/dur)
                    # smoothstep easing with slow finish
                    kk = k*k*k*(k*(k*6.0 - 15.0) + 10.0)  # smootherstep for dramatic pause
                    d = float(rig['d0']) * (1.0-kk) + float(rig['d1']) * kk
                    try:
                        self.view.distance = float(max(0.01, d))
                        self.view._update_camera()
                    except Exception:
                        self.view.update()
                    rig['t'] = t
                    
                    # Enhanced spawn moment - pause and reveal
                    if k >= 0.95 and not rig.get('spawn_triggered', False):
                        rig['spawn_triggered'] = True
                        if rig.get('spawn_molecule', False):
                            # Brief pause before spawn
                            rig['spawn_delay'] = 0.5  # half second pause
                            # Flash background to highlight the moment
                            try:
                                orig_bg = self.view.scene.bg_color.copy()
                                self.view.scene.bg_color[:] = [0.1, 0.1, 0.15]  # darker for contrast
                                rig['orig_bg'] = orig_bg
                            except Exception:
                                pass
                    
                    # Handle spawn delay and actual spawning
                    if rig.get('spawn_triggered', False):
                        delay = rig.get('spawn_delay', 0.0) - dt
                        rig['spawn_delay'] = max(0.0, delay)
                        if delay <= 0.0 and not rig.get('spawned', False):
                            rig['spawned'] = True
                            self._spawn_zoom_molecule(rig)
                            # Restore background gradually
                            if 'orig_bg' in rig:
                                rig['bg_restore_t'] = 0.0
                    
                    # Gradual background restore
                    if 'bg_restore_t' in rig:
                        restore_t = rig.get('bg_restore_t', 0.0) + dt
                        rig['bg_restore_t'] = restore_t
                        if restore_t < 1.0:
                            # Lerp back to original
                            orig = rig.get('orig_bg', [0.0, 0.0, 0.0])
                            current = [0.1, 0.1, 0.15]
                            for i in range(3):
                                self.view.scene.bg_color[i] = current[i] * (1.0 - restore_t) + orig[i] * restore_t
                        else:
                            # Fully restored
                            if 'orig_bg' in rig:
                                self.view.scene.bg_color[:] = rig['orig_bg']
                                del rig['orig_bg']
                                del rig['bg_restore_t']
                    
                    if k >= 1.0 and rig.get('spawned', False):
                        # mark for removal only after everything is complete
                        rig['done'] = True
            except Exception as e:
                log.debug(f"rig update failed: {e}")
        # remove finished rigs
        self._rigs = [r for r in self._rigs if not r.get('done')]

    def _update_tesseract_rig(self, rig):
        import math
        verts4 = rig['verts4']
        angs = rig['angs']
        proj = rig['proj']; fw = rig['fw']
        # Build 4D rotation matrix
        def R4_plane(i,j,deg):
            a = math.radians(float(deg)); c = math.cos(a); s = math.sin(a)
            R = np.eye(4, dtype=np.float32)
            R[i,i]=c; R[i,j]=-s; R[j,i]=s; R[j,j]=c
            return R
        R = np.eye(4, dtype=np.float32)
        R = R4_plane(0,1,angs['xy']) @ R
        R = R4_plane(0,2,angs['xz']) @ R
        R = R4_plane(0,3,angs['xw']) @ R
        R = R4_plane(1,2,angs['yz']) @ R
        R = R4_plane(1,3,angs['yw']) @ R
        R = R4_plane(2,3,angs['zw']) @ R
        # Transform and project
        v3 = []
        for v in verts4:
            vv = (R @ v.reshape(4,1)).reshape(4)
            x,y,z,w = float(vv[0]), float(vv[1]), float(vv[2]), float(vv[3])
            if proj == 0:
                p = np.array([x,y,z], np.float32)
            else:
                factor = fw / max(1e-3, (fw - w))
                p = np.array([x*factor, y*factor, z*factor], np.float32)
            v3.append(p)
        # Center offset
        center = np.array([0.0, 0.0, 0.0], np.float32)
        v3 = [p + center for p in v3]
        # Update edges
        for eidx, (i,j) in enumerate(rig['edges']):
            a = v3[i]; b = v3[j]
            mid = (a+b)*0.5
            dirv = b - a
            h = float(max(1e-6, np.linalg.norm(dirv)))
            y = dirv / max(1e-6, np.linalg.norm(dirv))
            # Build orthonormal basis (x,y,z) with y along dir
            tmp = np.array([1.0, 0.0, 0.0], np.float32) if abs(float(y[0])) < 0.9 else np.array([0.0, 0.0, 1.0], np.float32)
            x = np.cross(tmp, y); x /= max(1e-6, np.linalg.norm(x))
            z = np.cross(y, x)
            M = np.eye(4, dtype=np.float32)
            M[:3,0] = x; M[:3,1] = y; M[:3,2] = z; M[:3,3] = mid
            pr = self.view.scene.prims[rig['pidx_edges'][eidx]]
            pr.params[0] = float(rig['edge_r']); pr.params[1] = float(h)
            pr.xform.M = M
            try: pr.xform.M_inv = np.linalg.inv(M)
            except Exception: pr.xform.M_inv = None
        # Update nodes
        if rig['pidx_nodes']:
            for vidx, pidx in enumerate(rig['pidx_nodes']):
                pr = self.view.scene.prims[pidx]
                M = np.eye(4, dtype=np.float32); M[:3,3] = v3[vidx]
                pr.xform.M = M
                try: pr.xform.M_inv = np.linalg.inv(M)
                except Exception: pr.xform.M_inv = None
        try: self.view.scene._notify()
        except Exception: pass
        self.view.update()

    def _tesseract_vertices4(self, L):
        # 16 vertices with coords at +/- L/2 in 4D
        s = float(L) * 0.5
        verts = []
        for sx in (-s, s):
            for sy in (-s, s):
                for sz in (-s, s):
                    for sw in (-s, s):
                        verts.append(np.array([sx, sy, sz, sw], np.float32))
        return verts

    def _tesseract_edges(self):
        # Edges connect vertices differing in exactly one coordinate
        edges = []
        def idx_of(bits):
            # bits is tuple (bx,by,bz,bw) with 0 or 1
            return (bits[0]<<3) | (bits[1]<<2) | (bits[2]<<1) | bits[3]
        for bx in (0,1):
            for by in (0,1):
                for bz in (0,1):
                    for bw in (0,1):
                        b = (bx,by,bz,bw)
                        i = idx_of(b)
                        for dim in range(4):
                            bb = list(b); bb[dim] = 1 - bb[dim]
                            j = idx_of(tuple(bb))
                            if i < j:
                                edges.append((i,j))
        return edges

    # --- Zoom helpers ---
    def _start_zoom_journey(self):
        try:
            d0 = float(self._zoom_d0.value()); d1 = float(self._zoom_d1.value()); dur = float(self._zoom_dur.value())
            spawn = bool(self._zoom_spawn.isChecked())
            choice = self._zoom_mol_select.currentText()
            rig = {
                'type': 'zoom', 'd0': d0, 'd1': d1, 'dur': dur, 't': 0.0,
                'spawn_molecule': spawn,
                'choice': choice,
                'mol_path': self._zoom_mol_path
            }
            # replace existing zoom rigs
            self._rigs = [r for r in self._rigs if r.get('type') != 'zoom'] + [rig]
        except Exception as e:
            log.debug(f"start zoom failed: {e}")

    def _spawn_zoom_molecule(self, rig):
        try:
            choice = (rig.get('choice') or 'None').lower()
            path = None
            if choice.startswith('water'):
                path = os.path.join(os.getcwd(), 'examples', 'molecules', 'water.xyz')
            elif choice.startswith('methane'):
                path = os.path.join(os.getcwd(), 'examples', 'molecules', 'methane.xyz')
            elif 'choose' in choice and rig.get('mol_path'):
                path = rig.get('mol_path')
            if not path or not os.path.exists(path):
                return
            atoms = self._parse_xyz(path)
            if not atoms:
                return
            
            # Enhanced spawn with dramatic entrance
            mol_rig = self._build_molecule_rig(atoms,
                atom_scale=float(self._mol_atom_scale.value()) * 0.1,  # start tiny
                bond_r=float(self._mol_bond_r.value()) * 0.1,
                bond_thresh=float(self._mol_bond_thresh.value()),
                autospin=True,
                speed=60.0)  # faster spin initially
            
            if mol_rig:
                # Add growth animation to the molecule
                mol_rig['growth'] = {
                    'start_time': 0.0,
                    'duration': 2.0,
                    'target_atom_scale': float(self._mol_atom_scale.value()),
                    'target_bond_r': float(self._mol_bond_r.value()),
                    'target_speed': 30.0
                }
                self._rigs.append(mol_rig)
                log.info(f"Spawned molecule from zoom: {choice}")
                
        except Exception as e:
            log.debug(f"spawn molecule failed: {e}")

    # --- Molecule import ---
    def _import_xyz_molecule(self):
        try:
            from PySide6.QtWidgets import QFileDialog, QMessageBox
            path, _ = QFileDialog.getOpenFileName(self, "Import Molecule (XYZ)", filter="XYZ (*.xyz)")
            if not path:
                return
            atoms = self._parse_xyz(path)
            if not atoms:
                QMessageBox.information(self, "Import XYZ", "No atoms parsed.")
                return
            # Build rig with bonds and add to scene
            rig = self._build_molecule_rig(atoms,
                                           atom_scale=float(self._mol_atom_scale.value()),
                                           bond_r=float(self._mol_bond_r.value()),
                                           bond_thresh=float(self._mol_bond_thresh.value()),
                                           autospin=bool(self._mol_autospin.isChecked()),
                                           speed=float(self._mol_speed.value()))
            if rig is None:
                return
            self._rigs.append(rig)
            self._refresh_prim_label(); self.view.update()
        except Exception as e:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Import XYZ", f"Failed: {e}")
            except Exception:
                pass

    def _parse_xyz(self, path: str):
        atoms = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                return []
            try:
                n = int(lines[0].split()[0])
                start = 2 if len(lines) > 1 else 1
            except Exception:
                n = len(lines)
                start = 0
            for ln in lines[start:start+n]:
                parts = ln.split()
                if len(parts) < 4:
                    continue
                sym = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except Exception:
                    continue
                color, rcov = self._element_color_radius(sym)
                atoms.append({'sym': sym, 'pos': np.array([x, y, z], np.float32), 'color': color, 'rcov': rcov})
        except Exception:
            return []
        return atoms

    def _element_color_radius(self, sym: str):
        s = sym.capitalize()
        # Simple JMol-like colors and covalent radii (Å) for common elements; defaults otherwise
        colors = {
            'H': (1.0, 1.0, 1.0), 'C': (0.3, 0.3, 0.3), 'N': (0.1, 0.1, 0.9), 'O': (0.9, 0.1, 0.1),
            'S': (0.9, 0.9, 0.2), 'P': (1.0, 0.5, 0.0), 'F': (0.1, 0.9, 0.1), 'Cl': (0.1, 0.8, 0.1)
        }
        rcov = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05, 'P': 1.07, 'F': 0.57, 'Cl': 1.02
        }.get(s, 0.8)
        return colors.get(s, (0.7, 0.7, 0.7)), rcov

    def _build_molecule_rig(self, atoms, atom_scale=0.6, bond_r=0.035, bond_thresh=0.2, autospin=True, speed=25.0):
        try:
            pts = np.array([a['pos'] for a in atoms], dtype=np.float32)
            ctr = pts.mean(axis=0)
            pts -= ctr
            # Bond detection by covalent radii sum + threshold (Å)
            bonds = []
            N = len(atoms)
            for i in range(N):
                for j in range(i+1, N):
                    rij = np.linalg.norm(pts[i] - pts[j])
                    if rij <= (atoms[i]['rcov'] + atoms[j]['rcov'] + bond_thresh):
                        bonds.append((i, j))
            need = N + len(bonds)
            if len(self.view.scene.prims) + need > MAX_PRIMS:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Import XYZ", f"Not enough primitive budget for molecule: need {need}, have {MAX_PRIMS-len(self.view.scene.prims)}")
                return None
            pidx_atoms = []
            for i, a in enumerate(atoms):
                r = float(atom_scale * max(0.2, a.get('rcov', 0.8)) * 0.5)
                pr = Prim(KIND_SPHERE, [r, 0, 0, 0], beta=0.0, color=tuple(a.get('color', (0.7, 0.7, 0.7))))
                M = np.eye(4, dtype=np.float32); M[:3,3] = pts[i]
                pr.xform.M = M
                try: pr.xform.M_inv = np.linalg.inv(M)
                except Exception: pr.xform.M_inv = None
                self.view.scene.add(pr)
                pidx_atoms.append(len(self.view.scene.prims)-1)
            pidx_bonds = []
            for (i, j) in bonds:
                pr = Prim(KIND_CAPSULE, [float(bond_r), 1.0, 0.0, 0.0], beta=0.0, color=(0.85, 0.85, 0.85))
                self.view.scene.add(pr)
                pidx_bonds.append(len(self.view.scene.prims)-1)
            rig = {
                'type': 'molecule',
                'pts': pts,  # Nx3 np.array
                'bonds': bonds,
                'pidx_atoms': pidx_atoms,
                'pidx_bonds': pidx_bonds,
                'bond_r': float(bond_r),
                'autospin': bool(autospin),
                'speed': float(speed),
                'angle': 0.0
            }
            # Apply initial placement
            self._update_molecule_rig(rig)
            return rig
        except Exception as e:
            log.debug(f"molecule build failed: {e}")
            return None

    def _update_molecule_rig(self, rig):
        try:
            # Simple rotation about Y axis by angle (degrees)
            ang = float(rig.get('angle', 0.0))
            ca = np.cos(np.deg2rad(ang)); sa = np.sin(np.deg2rad(ang))
            R = np.array([[ ca, 0.0,  sa],
                          [0.0, 1.0, 0.0],
                          [-sa, 0.0,  ca]], dtype=np.float32)
            pts_rot = (R @ rig['pts'].T).T
            # Update atoms
            for idx, pidx in enumerate(rig['pidx_atoms']):
                pr = self.view.scene.prims[pidx]
                M = np.eye(4, dtype=np.float32); M[:3,3] = pts_rot[idx]
                pr.xform.M = M
                try: pr.xform.M_inv = np.linalg.inv(M)
                except Exception: pr.xform.M_inv = None
            # Update bonds
            for bidx, (i, j) in enumerate(rig['bonds']):
                a = pts_rot[i]; b = pts_rot[j]
                mid = (a + b) * 0.5
                dirv = b - a
                h = float(max(1e-6, np.linalg.norm(dirv)))
                y = dirv / max(1e-6, np.linalg.norm(dirv))
                tmp = np.array([1.0, 0.0, 0.0], np.float32) if abs(float(y[0])) < 0.9 else np.array([0.0, 0.0, 1.0], np.float32)
                x = np.cross(tmp, y); x /= max(1e-6, np.linalg.norm(x))
                z = np.cross(y, x)
                M = np.eye(4, dtype=np.float32)
                M[:3,0] = x; M[:3,1] = y; M[:3,2] = z; M[:3,3] = mid
                pr = self.view.scene.prims[rig['pidx_bonds'][bidx]]
                pr.params[0] = float(rig['bond_r']); pr.params[1] = float(h)
                pr.xform.M = M
                try: pr.xform.M_inv = np.linalg.inv(M)
                except Exception: pr.xform.M_inv = None
            try: self.view.scene._notify()
            except Exception: pass
            self.view.update()
        except Exception as e:
            log.debug(f"molecule update failed: {e}")

    def _update_sketch_info(self, sel_key):
        try:
            if sel_key is None:
                self._sketch_info_label.setText("No selection"); return
            if isinstance(sel_key, tuple) and len(sel_key)==2 and isinstance(sel_key[0], Polyline2D):
                ent, idx = sel_key
                self._sketch_info_label.setText(f"Polyline: {len(getattr(ent,'pts',[]))} pts, vertex #{idx}")
                return
            if isinstance(sel_key, tuple) and sel_key[0]=='circle' and isinstance(sel_key[1], Circle2D):
                ent = sel_key[1]
                self._sketch_info_label.setText(f"Circle: r={getattr(ent,'radius',0.0):.3f}")
                return
            self._sketch_info_label.setText("Selection: unknown")
        except Exception:
            try:
                self._sketch_info_label.setText("Selection: error")
            except Exception:
                pass

    # --- UI callbacks ---

    def _sync_wheel_controls(self) -> None:
        if not _HAVE_WHEEL:
            return
        if self._cb_wheel_enabled is not None:
            self._cb_wheel_enabled.blockSignals(True)
            self._cb_wheel_enabled.setChecked(bool(self._wheel_enabled))
            self._cb_wheel_enabled.blockSignals(False)
        if self._cb_wheel_edge_fit is not None:
            self._cb_wheel_edge_fit.blockSignals(True)
            self._cb_wheel_edge_fit.setChecked(bool(self._wheel_edge_fit))
            self._cb_wheel_edge_fit.blockSignals(False)
        if self._spin_wheel_overshoot is not None:
            self._spin_wheel_overshoot.blockSignals(True)
            self._spin_wheel_overshoot.setValue(int(self._wheel_edge_overshoot))
            self._spin_wheel_overshoot.setEnabled(bool(self._wheel_edge_fit))
            self._spin_wheel_overshoot.blockSignals(False)
        if self._slider_wheel_thickness is not None:
            val = int(round(self._wheel_thickness * 100))
            self._slider_wheel_thickness.blockSignals(True)
            self._slider_wheel_thickness.setValue(val)
            self._slider_wheel_thickness.blockSignals(False)
            if self._label_wheel_thickness is not None:
                self._label_wheel_thickness.setText(f"{val}%")
        if self._slider_wheel_opacity is not None:
            val = int(round(self._wheel_opacity * 100))
            self._slider_wheel_opacity.blockSignals(True)
            self._slider_wheel_opacity.setValue(val)
            self._slider_wheel_opacity.blockSignals(False)
            if self._label_wheel_opacity is not None:
                self._label_wheel_opacity.setText(f"{val}%")
        if self._slider_wheel_text is not None:
            val = int(round(self._wheel_text_scale * 100))
            self._slider_wheel_text.blockSignals(True)
            self._slider_wheel_text.setValue(val)
            self._slider_wheel_text.blockSignals(False)
            if self._label_wheel_text is not None:
                self._label_wheel_text.setText(f"{val/100:.2f}x")
        if self._cb_wheel_autospin is not None:
            self._cb_wheel_autospin.blockSignals(True)
            self._cb_wheel_autospin.setChecked(bool(self._wheel_auto_spin))
            self._cb_wheel_autospin.blockSignals(False)
        if self._slider_wheel_spin is not None:
            val = int(round(self._wheel_auto_spin_speed))
            self._slider_wheel_spin.blockSignals(True)
            self._slider_wheel_spin.setValue(val)
            self._slider_wheel_spin.blockSignals(False)
            self._slider_wheel_spin.setEnabled(bool(self._wheel_auto_spin))
            if self._label_wheel_spin is not None:
                self._label_wheel_spin.setText(f"{val} deg/s")

    def _apply_wheel_settings(self, persist: bool = False) -> None:
        if not _HAVE_WHEEL:
            return
        wheel = getattr(self, '_radial_wheel', None)
        if wheel is None:
            return
        wheel.setVisible(bool(self._wheel_enabled))
        wheel.set_fit_to_viewport_edge(bool(self._wheel_edge_fit))
        wheel.set_edge_overshoot(int(self._wheel_edge_overshoot))
        wheel.set_thickness_ratio(float(self._wheel_thickness))
        wheel.set_opacity(float(self._wheel_opacity))
        wheel.set_text_scale(float(self._wheel_text_scale))
        spin_speed = float(self._wheel_auto_spin_speed) if self._wheel_auto_spin else 0.0
        wheel.set_idle_rotation(spin_speed)
        if persist:
            self._persist_wheel_settings()

    def _persist_wheel_settings(self) -> None:
        try:
            self.view._save_settings()
        except Exception:
            pass

    def _on_wheel_enabled(self, checked: bool) -> None:
        self._wheel_enabled = bool(checked)
        self._apply_wheel_settings()
        self._persist_wheel_settings()

    def _on_wheel_edge_fit(self, checked: bool) -> None:
        self._wheel_edge_fit = bool(checked)
        if self._spin_wheel_overshoot is not None:
            self._spin_wheel_overshoot.setEnabled(self._wheel_edge_fit)
        self._apply_wheel_settings()
        self._persist_wheel_settings()

    def _on_wheel_overshoot(self, value: int) -> None:
        self._wheel_edge_overshoot = int(value)
        self._apply_wheel_settings()
        self._persist_wheel_settings()

    def _on_wheel_thickness(self, value: int) -> None:
        self._wheel_thickness = max(0.02, min(0.40, value / 100.0))
        self._apply_wheel_settings()
        self._persist_wheel_settings()

    def _on_wheel_opacity(self, value: int) -> None:
        self._wheel_opacity = max(0.0, min(1.0, value / 100.0))
        self._apply_wheel_settings()
        self._persist_wheel_settings()

    def _on_wheel_text_scale(self, value: int) -> None:
        self._wheel_text_scale = max(0.5, min(2.5, value / 100.0))
        self._apply_wheel_settings()
        self._persist_wheel_settings()

    def _on_wheel_autospin(self, checked: bool) -> None:
        self._wheel_auto_spin = bool(checked)
        if self._slider_wheel_spin is not None:
            self._slider_wheel_spin.setEnabled(self._wheel_auto_spin)
        self._apply_wheel_settings()
        self._persist_wheel_settings()

    def _on_wheel_spin_speed(self, value: int) -> None:
        self._wheel_auto_spin_speed = float(max(0, value))
        self._apply_wheel_settings()
        self._persist_wheel_settings()

    def _on_mode(self, idx:int):
        self.view.debug_mode = idx
        self.view._update_title(); self.view.update()

    def _on_toggle(self):
        self.view.use_analytic_aa = 1 if self.cb_aa.isChecked() else 0
        self.view.use_foveated = 1 if self.cb_fov.isChecked() else 0
        self.view.use_toon = 1 if self.cb_toon.isChecked() else 0
        self.view.show_beta_overlay = 1 if self.cb_beta.isChecked() else 0
        self.view._update_title(); self.view.update(); self.view._save_settings()

    def _on_toon_levels(self, v:int):
        self.view.toon_levels = float(v)
        if self.view.use_toon:
            self.view.update(); self.view._save_settings()

    def _on_curv(self, v:int):
        self.view.curv_strength = v/100.0
        self.view.update(); self.view._save_settings()

    def _on_beta_int(self, v:int):
        self.view.beta_overlay_intensity = v/100.0
        if self.view.show_beta_overlay:
            self.view.update(); self.view._save_settings()

    def _on_beta_scale(self, v:int):
        self.view.beta_scale = v/100.0
        if self.view.show_beta_overlay:
            self.view.update(); self.view._save_settings()

    def _on_cmap(self, idx:int):
        self.view.beta_cmap = idx
        if self.view.show_beta_overlay:
            self.view.update()
        self.view._save_settings(); self.view._update_title()

    # --- Environment callbacks ---
    def _on_bg_changed(self, _v):
        for i,s in enumerate(self._bg_sliders):
            self.view.scene.bg_color[i] = s.value()/255.0
        self.view.update()

    def _on_env_changed(self, _v):
        for i,s in enumerate(self._env_sliders):
            self.view.scene.env_light[i] = s.value()/100.0
        self.view.update()

    # --- Radial wheel handler ---
    def _delete_last_sketch_item(self) -> None:
        layer = getattr(self, '_sketch', None)
        if layer is None:
            return
        removed = False
        active_poly = getattr(layer, 'active_poly', None)
        if active_poly is not None and getattr(active_poly, 'pts', None):
            if active_poly.pts:
                active_poly.pts.pop()
                if not active_poly.pts:
                    layer.active_poly = None
                removed = True
        elif getattr(layer, 'dimensions', None):
            if layer.dimensions:
                layer.dimensions.pop()
                removed = True
        if not removed and getattr(layer, 'entities', None):
            if layer.entities:
                layer.entities.pop()
                removed = True
        if removed:
            try:
                self.view.update()
            except Exception:
                pass

    def _on_wheel_tool(self, tool_id: str, _label: str):
        try:
            # Ensure sketch overlay is enabled for sketch tools
            def enable_sketch():
                if self._cb_sketch is not None and not self._cb_sketch.isChecked():
                    self._cb_sketch.setChecked(True)
                    # stateChanged handler will update view
            if tool_id in ("polyline","line","arc3","circle","rect","insert_vertex","dimension"):
                enable_sketch()
                self._sketch_mode = tool_id
                # reset active shape/poly for a clean start
                if tool_id == 'polyline':
                    self._sketch.active_shape = None
                    self._sketch.active_poly = None
                else:
                    self._sketch.active_poly = None
                    self._sketch.active_shape = None
                # lock to top if desired
                if getattr(self, '_sketch_lock_top', False):
                    try:
                        self.view.yaw = 0.0; self.view.pitch = 0.0
                        self.view._update_camera()
                    except Exception:
                        pass
                self.view.update()
                return
            # Gizmo modes
            if tool_id in ("move","rotate","scale"):
                self._gizmo_mode = tool_id
                self.view.update(); return
            # Actions
            if tool_id == 'showcase' and self._btn_showcase is not None:
                self._btn_showcase.click(); return
            if tool_id == 'clear' and self._btn_clear_scene is not None:
                self._btn_clear_scene.click(); return
            if tool_id == 'center_selected':
                self._center_on_selected(); return
            if tool_id == 'reset_camera':
                self._reset_camera(); return
            if tool_id == 'delete_last':
                self._delete_last_sketch_item(); return
        except Exception as e:
            try:
                log.debug(f"wheel action failed: {e}")
            except Exception:
                pass

    # --- Primitive management ---
    def _add_prim(self, kind:str):
        if len(self.view.scene.prims) >= MAX_PRIMS:
            print(f"[prims] Max {MAX_PRIMS} reached")
            return
        if kind=='sphere':
            self.view.scene.add(Prim(KIND_SPHERE, [0.6,0,0,0], beta=0.05, color=(0.9,0.5,0.4)))
        elif kind=='box':
            self.view.scene.add(Prim(KIND_BOX, [0.5,0.5,0.5,0], beta=0.0, color=(0.4,0.7,0.9)))
        elif kind=='capsule':
            self.view.scene.add(Prim(KIND_CAPSULE, [0.3,1.5,0,0], beta=0.02, color=(0.6,0.9,0.5)))
        elif kind=='torus':
            self.view.scene.add(Prim(KIND_TORUS, [0.9,0.25,0,0], beta=0.03, color=(0.8,0.6,0.4)))
        elif kind=='mobius':
            # params: [R, width]
            self.view.scene.add(Prim(KIND_MOBIUS, [1.2, 0.25, 0, 0], beta=0.02, color=(0.7,0.5,0.9)))
        elif kind=='superellipsoid':
            # params: [radius, power]
            self.view.scene.add(Prim(KIND_SUPERELLIPSOID, [1.2, 4.0, 0, 0], beta=0.0, color=(0.95,0.9,0.85)))
        elif kind=='quasicrystal':
            # params: [scale, iso, thickness]
            self.view.scene.add(Prim(KIND_QUASICRYSTAL, [3.0, 1.0, 0.02, 0], beta=0.0, color=(0.2,0.9,0.9)))
        elif kind=='torus4d':
            # params: [R1, R2, r, w_slice] - two major radii, minor radius, 4D slice position
            self.view.scene.add(Prim(KIND_TORUS4D, [1.0, 0.8, 0.3, 0.0], beta=0.0, color=(0.9,0.3,0.9)))
        elif kind=='mandelbulb':
            # params: [power, bailout, max_iter, scale] - fractal parameters (better visibility)
            self.view.scene.add(Prim(KIND_MANDELBULB, [8.0, 2.0, 24, 0.8], beta=0.0, color=(0.9,0.2,0.1)))
        elif kind=='klein':
            # params: [scale, n, t_offset, thickness] - Klein bottle parameters (75% smaller)
            self.view.scene.add(Prim(KIND_KLEIN, [0.25, 2.0, 0.0, 0.025], beta=0.0, color=(0.1,0.7,0.9)))
        elif kind=='menger':
            # params: [iterations, size, 0, 0] - Menger sponge fractal
            self.view.scene.add(Prim(KIND_MENGER, [3, 1.0, 0, 0], beta=0.0, color=(0.8,0.6,0.2)))
        elif kind=='hyperbolic':
            # params: [scale, order, symmetry, 0] - Hyperbolic {order,symmetry} tiling
            self.view.scene.add(Prim(KIND_HYPERBOLIC, [2.0, 7, 3, 0], beta=0.0, color=(0.9,0.3,0.7)))
        elif kind=='gyroid':
            # params: [scale, tau, thickness, 0]
            self.view.scene.add(Prim(KIND_GYROID, [2.5, 0.0, 0.04, 0], beta=0.0, color=(0.7,0.9,0.3)))
        elif kind=='trefoil':
            # params: [scale, tube, samples, 0]
            self.view.scene.add(Prim(KIND_TREFOIL, [0.8, 0.08, 96, 0], beta=0.0, color=(0.9,0.5,0.2)))
        self._refresh_prim_label(); self.view.update()

    def _del_last(self):
        if self.view.scene.prims:
            self.view.scene.remove_index(len(self.view.scene.prims)-1)
            self._refresh_prim_label(); self.view.update()

    def _duplicate_selected(self):
        if not (0 <= self._current_sel < len(self.view.scene.prims)):
            return
        try:
            src = self.view.scene.prims[self._current_sel]
            # Shallow copy primitive fields
            from adaptivecad.aacore.sdf import Prim as _Prim
            new_pr = _Prim(src.kind, list(src.params), beta=float(src.beta), color=tuple(src.color[:3]), op=src.op)
            # Offset position slightly so it's visible; rely on set_transform if available
            M = src.xform.M.copy()
            M[:3,3] += np.array([0.25,0.25,0.0], np.float32)
            if hasattr(new_pr, 'xform'):
                new_pr.xform.M = M
            self.view.scene.add(new_pr)
            self._refresh_prim_label(); self.view.update()
        except Exception as e:
            log.debug(f"duplicate failed: {e}")

    def _center_on_selected(self):
        if not (0 <= self._current_sel < len(self.view.scene.prims)):
            return
        pr = self.view.scene.prims[self._current_sel]
        try:
            pos = pr.xform.M[:3,3]
            self.view.cam_target = pos.astype(np.float32)
            self.view._update_camera()
        except Exception:
            pass

    def _reset_camera(self):
        self.view.yaw = 0.0; self.view.pitch = 0.0
        self.view.distance = 6.0
        self.view.cam_target = np.array([0.0,0.0,0.0], np.float32)
        self.view._update_camera()

    def _export_scene_json(self):
        try:
            data = []
            for pr in self.view.scene.prims:
                entry = {
                    'kind': self._kind_name(pr.kind),
                    'params': [float(x) for x in pr.params],
                    'beta': float(pr.beta),
                    'color': [float(c) for c in pr.color[:3]],
                    'op': pr.op,
                    'pos': [float(v) for v in pr.xform.M[:3,3]]
                }
                data.append(entry)
            ts = time.strftime('%Y%m%d_%H%M%S')
            out_path = os.path.join(os.getcwd(), f'analytic_scene_{ts}.json')
            with open(out_path,'w',encoding='utf-8') as f:
                json.dump(data,f,indent=2)
            log.info(f"Exported analytic scene -> {out_path}")
        except Exception as e:
            log.debug(f"export scene failed: {e}")

    def _refresh_prim_label(self):
        self._prim_list_label.setText(f"({len(self.view.scene.prims)})")
        # rebuild primitive buttons
        while self._prim_buttons_box.count():
            it = self._prim_buttons_box.takeAt(0)
            w = it.widget()
            if w: w.deleteLater()
        for idx, pr in enumerate(self.view.scene.prims):
            b = QPushButton(f"#{idx} {self._kind_name(pr.kind)}")
            b.setCheckable(True)
            if idx == self._current_sel: b.setChecked(True)
            def make_cb(i):
                return lambda _=False: self._select_prim(i)
            b.clicked.connect(make_cb(idx))
            self._prim_buttons_box.addWidget(b)
        self._prim_buttons_box.addStretch(1)

    def _kind_name(self, k):
        return {KIND_SPHERE:"Sphere",KIND_BOX:"Box",KIND_CAPSULE:"Capsule",KIND_TORUS:"Torus", KIND_MOBIUS: "Mobius", KIND_SUPERELLIPSOID:"Superellipsoid", KIND_QUASICRYSTAL:"QuasiCrystal", KIND_TORUS4D:"4D Torus", KIND_MANDELBULB:"Mandelbulb", KIND_KLEIN:"Klein Bottle", KIND_MENGER:"Menger Sponge", KIND_HYPERBOLIC:"Hyperbolic Tiling", KIND_GYROID:"Gyroid", KIND_TREFOIL:"Trefoil Knot"}.get(k,str(k))

    def _make_row(self, widgets):
        box = QWidget(); hl = QHBoxLayout(box); hl.setContentsMargins(0,0,0,0)
        for w in widgets: hl.addWidget(w)
        return box

    def _ensure_on_screen(self):
        try:
            top = self.window()
            if top is None:
                return
            # get window frame geometry and check against available screens
            fg = top.frameGeometry()
            screens = QGuiApplication.screens()
            intersects = False
            for s in screens:
                ag = s.availableGeometry()
                if fg.intersected(ag).isValid():
                    intersects = True
                    break
            if not intersects and screens:
                # move to center of primary screen
                primary = QGuiApplication.primaryScreen() or screens[0]
                ag = primary.availableGeometry()
                new_x = ag.x() + max(0, (ag.width() - fg.width()) // 2)
                new_y = ag.y() + max(0, (ag.height() - fg.height()) // 2)
                try:
                    top.move(new_x, new_y)
                except Exception:
                    try:
                        top.setGeometry(new_x, new_y, fg.width(), fg.height())
                    except Exception:
                        pass
        except Exception:
            pass

    def _select_prim(self, idx:int):
        if not (0 <= idx < len(self.view.scene.prims)):
            self._current_sel = -1; self._edit_group.setEnabled(False); return
        self._current_sel = idx
        pr = self.view.scene.prims[idx]
        # position from transform matrix (assume affine last column xyz)
        pos = pr.xform.M[:3,3]
        for i in range(3): self._sp_pos[i].blockSignals(True); self._sp_pos[i].setValue(float(pos[i])); self._sp_pos[i].blockSignals(False)
        # populate move spinners (authoritative for translation edits)
        if hasattr(self, '_sp_move'):
            for i in range(3):
                self._sp_move[i].blockSignals(True)
                self._sp_move[i].setValue(float(pos[i]))
                self._sp_move[i].blockSignals(False)
        # params
        self._sp_param[0].blockSignals(True); self._sp_param[0].setValue(float(pr.params[0])); self._sp_param[0].blockSignals(False)
        self._sp_param[1].blockSignals(True); self._sp_param[1].setValue(float(pr.params[1] if pr.kind!=KIND_SPHERE else 0.0)); self._sp_param[1].blockSignals(False)
        self._sp_beta.blockSignals(True); self._sp_beta.setValue(pr.beta); self._sp_beta.blockSignals(False)
        self._op_box.blockSignals(True); self._op_box.setCurrentIndex(0 if pr.op=='solid' else 1); self._op_box.blockSignals(False)
        self._current_color = tuple(pr.color[:3])
        # rotation / scale sync
        if hasattr(pr, 'euler'):
            for i in range(3):
                self._sp_rot[i].blockSignals(True); self._sp_rot[i].setValue(float(pr.euler[i])); self._sp_rot[i].blockSignals(False)
        if hasattr(pr, 'scale'):
            for i in range(3):
                self._sp_scl[i].blockSignals(True); self._sp_scl[i].setValue(float(pr.scale[i])); self._sp_scl[i].blockSignals(False)
            try:
                # Reflect current uniform value as X scale (assumes uniform when set via this control)
                self._sp_scl_u.blockSignals(True)
                self._sp_scl_u.setValue(float(pr.scale[0]))
                self._sp_scl_u.blockSignals(False)
            except Exception:
                pass
        self._edit_group.setEnabled(True)
        # Update parameter tooltips / semantic hints
        try:
            kind_name = self._kind_name(pr.kind)
            if kind_name == 'Sphere':
                a_tt = 'Radius'; b_tt = 'Unused'
            elif kind_name == 'Capsule':
                a_tt = 'Radius'; b_tt = 'Half-Length'
            elif kind_name == 'Torus':
                a_tt = 'Major Radius'; b_tt = 'Minor Radius'
            elif kind_name == 'Mobius':
                a_tt = 'Ring Radius R'; b_tt = 'Strip Half-Width'
            elif kind_name == 'Box':
                a_tt = 'Half-Extent X'; b_tt = 'Half-Extent Y (Z via scale)'
            elif kind_name == 'Superellipsoid':
                a_tt = 'Radius (overall)'; b_tt = 'Power p (boxier > 2)'
            elif kind_name == 'QuasiCrystal':
                a_tt = 'Scale (frequency)'; b_tt = 'Iso (level)'
            elif kind_name == '4D Torus':
                a_tt = 'Major Radius 1 (XY plane)'; b_tt = 'Major Radius 2 (ZW plane)'
            elif kind_name == 'Mandelbulb':
                a_tt = 'Power (8.0 = classic)'; b_tt = 'Bailout Radius (escape)'
            elif kind_name == 'Klein Bottle':
                a_tt = 'Scale (overall size)'; b_tt = 'Shape Parameter'
            elif kind_name == 'Menger Sponge':
                a_tt = 'Iterations (detail level)'; b_tt = 'Size (overall scale)'
            elif kind_name == 'Hyperbolic Tiling':
                a_tt = 'Scale (overall size)'; b_tt = 'Order (polygon sides)'
            elif kind_name == 'Gyroid':
                a_tt = 'Scale (frequency)'; b_tt = 'Tau (iso-level)'
            elif kind_name == 'Trefoil Knot':
                a_tt = 'Scale (curve size)'; b_tt = 'Tube Radius'
            else:
                a_tt = 'Param A'; b_tt = 'Param B'
            self._sp_param[0].setToolTip(a_tt)
            self._sp_param[1].setToolTip(b_tt)
        except Exception:
            pass

    def _apply_edit(self):
        if self._current_sel < 0 or self._current_sel >= len(self.view.scene.prims):
            return
        pr = self.view.scene.prims[self._current_sel]
        # existing matrix before rebuilding
        M = pr.xform.M.copy()
        # apply params
        pr.params[0] = self._sp_param[0].value()
        if pr.kind != KIND_SPHERE:
            pr.params[1] = self._sp_param[1].value()
        pr.beta = self._sp_beta.value()
        pr.op = 'solid' if self._op_box.currentIndex()==0 else 'subtract'
        pr.color[:3] = np.array(self._current_color[:3])
        # translation driven by move spinners
        if hasattr(self, '_sp_move'):
            pos = [self._sp_move[i].value() for i in range(3)]
        else:
            pos = [self._sp_pos[i].value() for i in range(3)]
        rx, ry, rz = [self._sp_rot[i].value() for i in range(3)]
        sx, sy, sz = [self._sp_scl[i].value() for i in range(3)]
        if hasattr(pr, 'set_transform'):
            pr.set_transform(pos=pos, euler=[rx,ry,rz], scale=[sx,sy,sz])
        else:
            # fallback: just translate
            for i in range(3): M[i,3] = pos[i]
            pr.xform.M = M
        # notify & refresh
        try:
            self.view.scene._notify()
        except Exception:
            pass
        self.view.update()

    # --- Sketch save/load ---
    def _save_sketch_json(self):
        try:
            from PySide6.QtWidgets import QFileDialog, QMessageBox
            path, _ = QFileDialog.getSaveFileName(self, "Save Sketch JSON", "sketch.json", "JSON (*.json)")
            if not path:
                return
            # Build doc from overlay entities (minimal polyline/circle export)
            doc = SketchDocument(units=Units.MM)
            pid = 0
            def new_pid():
                nonlocal pid; pid += 1; return f"p{pid}"
            extra = { 'lines': [], 'rects': [], 'arcs3': [], 'dimensions': [], 'dim_precision': int(getattr(self,'_dim_precision',2)) }
            for ent in self._sketch.entities:
                if isinstance(ent, Polyline2D):
                    ids = []
                    for pt in ent.pts:
                        idp = new_pid(); ids.append(idp)
                        doc.points.append(type('P',(),{})())  # placeholder; will patch below
                        doc.points[-1].__dict__.update({"id": idp, "x": float(pt[0]), "y": float(pt[1]), "construction": False})
                    doc.polylines.append(type('PL',(),{})())
                    doc.polylines[-1].__dict__.update({"id": f"pl{pid}", "verts": ids, "closed": False, "construction": False})
                elif isinstance(ent, Circle2D):
                    idc = new_pid()
                    c = ent.center
                    doc.points.append(type('P',(),{})())
                    doc.points[-1].__dict__.update({"id": idc, "x": float(c[0]), "y": float(c[1]), "construction": False})
                    doc.circles.append(type('C',(),{})())
                    doc.circles[-1].__dict__.update({"id": f"c{pid}", "center": idc, "radius": float(ent.radius), "construction": False})
                elif isinstance(ent, Line2D):
                    if getattr(ent, 'p1', None) is not None:
                        extra['lines'].append({'p0':[float(ent.p0[0]), float(ent.p0[1])], 'p1':[float(ent.p1[0]), float(ent.p1[1])]})
                elif isinstance(ent, Rect2D):
                    p0 = [float(ent.p0[0]), float(ent.p0[1])]
                    p1 = [float(ent.p1[0]), float(ent.p1[1])] if getattr(ent,'p1', None) is not None else p0
                    extra['rects'].append({'p0': p0, 'p1': p1})
                elif isinstance(ent, Arc2D):
                    if getattr(ent,'p0',None) is not None and getattr(ent,'p1',None) is not None and getattr(ent,'p2',None) is not None:
                        extra['arcs3'].append({'p0':[float(ent.p0[0]), float(ent.p0[1])], 'p1':[float(ent.p1[0]), float(ent.p1[1])], 'p2':[float(ent.p2[0]), float(ent.p2[1])]} )
            # dimensions
            for d in getattr(self._sketch, 'dimensions', []) or []:
                try:
                    if isinstance(d, DimDiameter2D) and isinstance(d.circle_ref, tuple) and d.circle_ref[0]=='circle' and isinstance(d.circle_ref[1], Circle2D):
                        c = d.circle_ref[1]
                        extra['dimensions'].append({'type':'diameter', 'center':[float(c.center[0]), float(c.center[1])]})
                    elif isinstance(d, DimLinear2D):
                        a = d._xy(d.ref_a); b = d._xy(d.ref_b)
                        if a is not None and b is not None:
                            # type discrimination for H/V vs general
                            t = 'linear'
                            if isinstance(d, DimHorizontal2D): t='horizontal'
                            elif isinstance(d, DimVertical2D): t='vertical'
                            extra['dimensions'].append({'type': t, 'a':[float(a[0]), float(a[1])], 'b':[float(b[0]), float(b[1])], 'offset': float(getattr(d,'offset',0.0))})
                except Exception:
                    pass
            # Merge extra into doc JSON
            try:
                base = json.loads(doc.to_json())
            except Exception:
                base = {}
            base.update(extra)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(base, f, indent=2)
            QMessageBox.information(self, "Sketch", f"Saved sketch to {path}")
        except Exception as e:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Sketch Save", f"Failed: {e}")
            except Exception:
                pass

    def _load_sketch_json(self):
        try:
            from PySide6.QtWidgets import QFileDialog, QMessageBox
            path, _ = QFileDialog.getOpenFileName(self, "Load Sketch JSON", filter="JSON (*.json)")
            if not path:
                return
            with open(path, 'r', encoding='utf-8') as f:
                s = f.read()
            # Parse raw JSON alongside SketchDocument for compatibility
            try:
                raw = json.loads(s)
            except Exception:
                raw = {}
            doc = SketchDocument.from_json(s)
            # Replace overlay entities and include extended shapes
            self._sketch.entities.clear()
            # Build point map
            pm = {p.id: np.array([p.x, p.y], np.float32) for p in getattr(doc, 'points', [])}
            for pl in getattr(doc, 'polylines', []):
                pts = [ pm[v] for v in getattr(pl, 'verts', []) if v in pm ]
                self._sketch.entities.append(Polyline2D(pts))
            for c in getattr(doc, 'circles', []):
                ctr = pm.get(getattr(c, 'center', ''), np.array([0,0], np.float32))
                self._sketch.entities.append(Circle2D(ctr, float(getattr(c, 'radius', 0.0))))
            # Extended entities (raw JSON)
            for li in raw.get('lines', []) or []:
                p0 = np.array([float(li.get('p0',[0,0])[0]), float(li.get('p0',[0,0])[1])], np.float32)
                p1 = np.array([float(li.get('p1',[0,0])[0]), float(li.get('p1',[0,0])[1])], np.float32)
                self._sketch.entities.append(Line2D(p0, p1))
            for r in raw.get('rects', []) or []:
                p0 = np.array([float(r.get('p0',[0,0])[0]), float(r.get('p0',[0,0])[1])], np.float32)
                p1 = np.array([float(r.get('p1',[0,0])[0]), float(r.get('p1',[0,0])[1])], np.float32)
                self._sketch.entities.append(Rect2D(p0, p1))
            for a in raw.get('arcs3', []) or []:
                p0 = np.array([float(a.get('p0',[0,0])[0]), float(a.get('p0',[0,0])[1])], np.float32)
                p1 = np.array([float(a.get('p1',[0,0])[0]), float(a.get('p1',[0,0])[1])], np.float32)
                p2 = np.array([float(a.get('p2',[0,0])[0]), float(a.get('p2',[0,0])[1])], np.float32)
                self._sketch.entities.append(Arc2D(p0, p1, p2))
            # Dimensions
            self._sketch.dimensions = []
            for d in raw.get('dimensions', []) or []:
                t = d.get('type','linear')
                if t == 'diameter':
                    cx,cy = d.get('center',[0,0])
                    circ = None
                    for ent in self._sketch.entities:
                        if isinstance(ent, Circle2D) and abs(ent.center[0]-cx)<1e-6 and abs(ent.center[1]-cy)<1e-6:
                            circ = ent; break
                    if circ is not None:
                        self._sketch.dimensions.append(DimDiameter2D(('circle', circ), self))
                else:
                    a = d.get('a'); b = d.get('b')
                    if a is not None and b is not None:
                        pa = np.array([float(a[0]), float(a[1])], np.float32)
                        pb = np.array([float(b[0]), float(b[1])], np.float32)
                        off = float(d.get('offset', 0.0))
                        if t=='horizontal':
                            self._sketch.dimensions.append(DimHorizontal2D(pa, pb, self, offset=off))
                        elif t=='vertical':
                            self._sketch.dimensions.append(DimVertical2D(pa, pb, self, offset=off))
                        else:
                            self._sketch.dimensions.append(DimLinear2D(pa, pb, self, offset=off))
            # restore precision if present
            try:
                self._dim_precision = int(raw.get('dim_precision', getattr(self,'_dim_precision',2)))
            except Exception:
                pass
            self.view.update()
            QMessageBox.information(self, "Sketch", f"Loaded sketch from {path}")
        except Exception as e:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Sketch Load", f"Failed: {e}")
            except Exception:
                pass

    # Convenience passthroughs
    def viewport(self) -> AnalyticViewport: return self.view

def create_analytic_viewport_with_panel(parent=None, aacore_scene: AACoreScene | None = None):
    return AnalyticViewportPanel(parent, aacore_scene=aacore_scene)

# --- persistence helpers on AnalyticViewport ---
def _analytic_settings_defaults():
    return {
        'debug_mode':0,
        'use_analytic_aa':1,
        'use_toon':0,
        'toon_levels':4.0,
        'curv_strength':1.0,
        'use_foveated':1,
        'show_beta_overlay':0,
        'beta_overlay_intensity':0.65,
        'beta_scale':1.0,
        'beta_cmap':0,
        'wheel_enabled': True,
        'wheel_edge_fit': True,
        'wheel_overshoot_px': 14,
        'wheel_thickness_ratio': 0.10,
        'wheel_opacity': 1.0,
        'wheel_text_scale': 1.0,
        'wheel_auto_spin': False,
        'wheel_auto_spin_speed': 0.0
    }

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _apply_settings(self, data):
    self.debug_mode = int(data.get('debug_mode',0))
    self.use_analytic_aa = int(data.get('use_analytic_aa',1))
    self.use_toon = int(data.get('use_toon',0))
    self.toon_levels = float(_clamp(data.get('toon_levels',4.0),1.0,16.0))
    self.curv_strength = float(_clamp(data.get('curv_strength',1.0),0.0,2.0))
    self.use_foveated = int(data.get('use_foveated',1))
    self.show_beta_overlay = int(data.get('show_beta_overlay',0))
    self.beta_overlay_intensity = float(_clamp(data.get('beta_overlay_intensity',0.65),0.0,1.0))
    self.beta_scale = float(_clamp(data.get('beta_scale',1.0),0.01,32.0))
    self.beta_cmap = int(_clamp(data.get('beta_cmap',0),0,3))
    # dim settings
    try:
        panel = self.parent() if hasattr(self,'parent') else None
        if panel is not None:
            setattr(panel, '_dim_precision', int(data.get('dim_precision', getattr(panel,'_dim_precision',2))))
            setattr(panel, '_dim_type', str(data.get('dim_type', getattr(panel,'_dim_type','linear'))))
            if hasattr(panel, '_wheel_enabled'):
                panel._wheel_enabled = bool(data.get('wheel_enabled', panel._wheel_enabled))
                panel._wheel_edge_fit = bool(data.get('wheel_edge_fit', panel._wheel_edge_fit))
                panel._wheel_edge_overshoot = int(data.get('wheel_overshoot_px', panel._wheel_edge_overshoot))
                panel._wheel_thickness = float(_clamp(data.get('wheel_thickness_ratio', panel._wheel_thickness), 0.02, 0.40))
                panel._wheel_opacity = float(_clamp(data.get('wheel_opacity', panel._wheel_opacity), 0.0, 1.0))
                panel._wheel_text_scale = float(_clamp(data.get('wheel_text_scale', panel._wheel_text_scale), 0.5, 2.5))
                panel._wheel_auto_spin = bool(data.get('wheel_auto_spin', panel._wheel_auto_spin))
                panel._wheel_auto_spin_speed = float(_clamp(data.get('wheel_auto_spin_speed', panel._wheel_auto_spin_speed), 0.0, 360.0))
                if hasattr(panel, '_sync_wheel_controls'):
                    panel._sync_wheel_controls()
                if hasattr(panel, '_apply_wheel_settings'):
                    panel._apply_wheel_settings()
    except Exception:
        pass

def _load_settings(self):
    try:
        if os.path.exists(self._settings_path):
            with open(self._settings_path,'r',encoding='utf-8') as f:
                data=json.load(f)
        else:
            data=_analytic_settings_defaults()
        _apply_settings(self, data)
        # Restore sketch overlay (if any)
        try:
            panel = self.parent() if hasattr(self, 'parent') else None
            sk = data.get('sketch_overlay', None)
            if panel is not None and sk is not None and hasattr(panel, '_sketch'):
                panel._sketch.entities.clear()
                for pl in sk.get('polylines', []) or []:
                    pts = [np.array([float(p[0]), float(p[1])], np.float32) for p in pl]
                    panel._sketch.entities.append(Polyline2D(pts))
                for c in sk.get('circles', []) or []:
                    ctr = np.array([float(c.get('center',[0,0])[0]), float(c.get('center',[0,0])[1])], np.float32)
                    rad = float(c.get('radius', 0.0))
                    panel._sketch.entities.append(Circle2D(ctr, rad))
                for li in sk.get('lines', []) or []:
                    p0 = np.array([float(li.get('p0',[0,0])[0]), float(li.get('p0',[0,0])[1])], np.float32)
                    p1 = np.array([float(li.get('p1',[0,0])[0]), float(li.get('p1',[0,0])[1])], np.float32)
                    panel._sketch.entities.append(Line2D(p0, p1))
                for r in sk.get('rects', []) or []:
                    p0 = np.array([float(r.get('p0',[0,0])[0]), float(r.get('p0',[0,0])[1])], np.float32)
                    p1 = np.array([float(r.get('p1',[0,0])[0]), float(r.get('p1',[0,0])[1])], np.float32)
                    panel._sketch.entities.append(Rect2D(p0, p1))
                for a in sk.get('arcs3', []) or []:
                    p0 = np.array([float(a.get('p0',[0,0])[0]), float(a.get('p0',[0,0])[1])], np.float32)
                    p1 = np.array([float(a.get('p1',[0,0])[0]), float(a.get('p1',[0,0])[1])], np.float32)
                    p2 = np.array([float(a.get('p2',[0,0])[0]), float(a.get('p2',[0,0])[1])], np.float32)
                    panel._sketch.entities.append(Arc2D(p0, p1, p2))
                # dimensions
                try:
                    panel._sketch.dimensions = []
                    for d in sk.get('dimensions', []) or []:
                        t = d.get('type','linear')
                        if t == 'diameter':
                            # find circle by matching center exactly
                            cx,cy = d.get('center',[0,0])
                            circ = None
                            for ent in panel._sketch.entities:
                                if isinstance(ent, Circle2D) and abs(ent.center[0]-cx)<1e-6 and abs(ent.center[1]-cy)<1e-6:
                                    circ = ent; break
                            if circ is not None:
                                panel._sketch.dimensions.append(DimDiameter2D(('circle', circ), panel))
                        else:
                            a = d.get('a'); b = d.get('b')
                            if a is not None and b is not None:
                                pa = np.array([float(a[0]), float(a[1])], np.float32)
                                pb = np.array([float(b[0]), float(b[1])], np.float32)
                                off = float(d.get('offset', 0.0))
                                if t=='horizontal':
                                    panel._sketch.dimensions.append(DimHorizontal2D(pa, pb, panel, offset=off))
                                elif t=='vertical':
                                    panel._sketch.dimensions.append(DimVertical2D(pa, pb, panel, offset=off))
                                else:
                                    panel._sketch.dimensions.append(DimLinear2D(pa, pb, panel, offset=off))
                except Exception:
                    pass
                # grid settings
                try:
                    panel._sketch_grid_on = bool(sk.get('grid_on', panel._sketch_grid_on))
                    panel._sketch_snap_grid = bool(sk.get('snap_grid', panel._sketch_snap_grid))
                    panel._sketch_grid_step = float(sk.get('grid_step', panel._sketch_grid_step))
                    panel._sketch_grid_screen_lock = bool(sk.get('grid_screen_lock', getattr(panel,'_sketch_grid_screen_lock', True)))
                except Exception:
                    pass
        except Exception:
            pass
    except Exception as e:
        print(f"(analytic settings) load failed: {e}")

def _save_settings(self):
    try:
        data_update={
            'debug_mode': self.debug_mode,
            'use_analytic_aa': self.use_analytic_aa,
            'use_toon': self.use_toon,
            'toon_levels': self.toon_levels,
            'curv_strength': self.curv_strength,
            'use_foveated': self.use_foveated,
            'show_beta_overlay': self.show_beta_overlay,
            'beta_overlay_intensity': self.beta_overlay_intensity,
            'beta_scale': self.beta_scale,
            'beta_cmap': self.beta_cmap
        }
        panel = self.parent() if hasattr(self, 'parent') else None
        if panel is not None and hasattr(panel, '_wheel_enabled'):
            data_update.update({
                'wheel_enabled': bool(panel._wheel_enabled),
                'wheel_edge_fit': bool(panel._wheel_edge_fit),
                'wheel_overshoot_px': int(panel._wheel_edge_overshoot),
                'wheel_thickness_ratio': float(panel._wheel_thickness),
                'wheel_opacity': float(panel._wheel_opacity),
                'wheel_text_scale': float(panel._wheel_text_scale),
                'wheel_auto_spin': bool(panel._wheel_auto_spin),
                'wheel_auto_spin_speed': float(panel._wheel_auto_spin_speed),
            })
        # Preserve unrelated preference keys (e.g., 'analytic_as_main') by merging
        existing = {}
        if os.path.exists(self._settings_path):
            try:
                with open(self._settings_path,'r',encoding='utf-8') as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}
        # Serialize sketch overlay
        try:
            panel = self.parent() if hasattr(self, 'parent') else None
            if panel is not None and hasattr(panel, '_sketch'):
                sk = {'polylines': [], 'circles': [], 'lines': [], 'rects': [], 'arcs3': [], 'dimensions': [], 'grid_on': bool(getattr(panel,'_sketch_grid_on', False)), 'snap_grid': bool(getattr(panel,'_sketch_snap_grid', False)), 'grid_step': float(getattr(panel,'_sketch_grid_step', 10.0)), 'grid_screen_lock': bool(getattr(panel,'_sketch_grid_screen_lock', True))}
                for ent in getattr(panel, '_sketch').entities:
                    if isinstance(ent, Polyline2D):
                        sk['polylines'].append([[float(p[0]), float(p[1])] for p in ent.pts])
                    elif isinstance(ent, Circle2D):
                        sk['circles'].append({'center':[float(ent.center[0]), float(ent.center[1])], 'radius': float(ent.radius)})
                    elif isinstance(ent, Line2D):
                        if ent.p1 is not None:
                            sk['lines'].append({'p0':[float(ent.p0[0]), float(ent.p0[1])], 'p1':[float(ent.p1[0]), float(ent.p1[1])]} )
                    elif isinstance(ent, Rect2D):
                        p0 = [float(ent.p0[0]), float(ent.p0[1])]
                        p1 = [float(ent.p1[0]), float(ent.p1[1])] if getattr(ent,'p1', None) is not None else p0
                        sk['rects'].append({'p0': p0, 'p1': p1})
                    elif isinstance(ent, Arc2D):
                        if ent.p0 is not None and ent.p1 is not None and ent.p2 is not None:
                            sk['arcs3'].append({'p0':[float(ent.p0[0]), float(ent.p0[1])], 'p1':[float(ent.p1[0]), float(ent.p1[1])], 'p2':[float(ent.p2[0]), float(ent.p2[1])]} )
                # dimensions (store basic types we support now)
                for d in getattr(panel._sketch, 'dimensions', []) or []:
                    try:
                        if isinstance(d, DimDiameter2D) and isinstance(d.circle_ref, tuple) and d.circle_ref[0]=='circle' and isinstance(d.circle_ref[1], Circle2D):
                            c = d.circle_ref[1]
                            sk['dimensions'].append({'type':'diameter', 'center':[float(c.center[0]), float(c.center[1])]})
                        elif isinstance(d, DimLinear2D):
                            a = d._xy(d.ref_a); b = d._xy(d.ref_b)
                            if a is not None and b is not None:
                                t = 'linear'
                                if isinstance(d, DimHorizontal2D): t='horizontal'
                                elif isinstance(d, DimVertical2D): t='vertical'
                                sk['dimensions'].append({'type': t, 'a':[float(a[0]), float(a[1])], 'b':[float(b[0]), float(b[1])], 'offset': float(getattr(d,'offset',0.0))})
                    except Exception:
                        pass
                data_update['sketch_overlay'] = sk
                # dimension settings
                try:
                    data_update['dim_precision'] = int(getattr(panel,'_dim_precision',2))
                    data_update['dim_type'] = str(getattr(panel,'_dim_type','linear'))
                except Exception:
                    pass
        except Exception:
            pass
        existing.update(data_update)
        with open(self._settings_path,'w',encoding='utf-8') as f:
            json.dump(existing,f,indent=2)
    except Exception as e:
        print(f"(analytic settings) save failed: {e}")

# Monkey-patch methods onto AnalyticViewport class after definition (simple, avoids editing earlier class body heavily)
AnalyticViewport._load_settings = _load_settings
AnalyticViewport._save_settings = _save_settings
