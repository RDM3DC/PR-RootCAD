import os, sys, time, math
import argparse
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PySide6.QtCore import QTimer
from OpenGL.GL import *

# Import project modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptivecad.gui.analytic_viewport import AnalyticViewport
from adaptivecad.aacore.sdf import Scene, Prim, \
    KIND_SUPERELLIPSOID, KIND_QUASICRYSTAL, KIND_SPHERE, OP_SUBTRACT

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 4*t*t*t if t < 0.5 else 1 - pow(-2*t+2, 3)/2.0


def save_frame_rgba(view: AnalyticViewport, path: str):
    """Capture current framebuffer via QOpenGLWidget.grabFramebuffer."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    qimg = view.grabFramebuffer()  # QImage in RGBA8888 premultiplied
    try:
        qimg.save(path)
    except Exception:
        # Fallback path using numpy if needed
        ptr = qimg.bits()
        ptr.setsize(qimg.sizeInBytes())
        arr = np.array(ptr, dtype=np.uint8).reshape(qimg.height(), qimg.width(), 4)
        if _HAVE_PIL:
            Image.fromarray(arr, 'RGBA').save(path)
        else:
            with open(path + '.raw', 'wb') as f:
                f.write(arr.tobytes())


def build_scene() -> Scene:
    scene = Scene()
    # Superellipsoid shell via solid minus inner subtract
    # Outer: radius R, power p (boxier with higher p); anisotropy handled via transform scale
    R = 1.7; pwr = 4.0; thick = 0.10
    outer = Prim(KIND_SUPERELLIPSOID, [R, pwr, 0, 0], beta=0.0, color=(0.98, 0.94, 0.88))
    outer.set_transform(pos=[0,0,0], euler=[0,0,0], scale=[1.25, 1.25, 1.0])
    inner = Prim(KIND_SUPERELLIPSOID, [max(0.05, R - thick), pwr, 0, 0], beta=0.0, color=(0.0,0.0,0.0), op='subtract')
    inner.set_transform(pos=[0,0,0], euler=[0,0,0], scale=[1.25, 1.25, 1.0])
    scene.add(outer); scene.add(inner)

    # Quasi-crystal field (union): params [scale, iso, thickness]
    qc = Prim(KIND_QUASICRYSTAL, [3.0, 1.0, 0.02, 0], beta=0.0, color=(0.2,0.9,0.9))
    scene.add(qc)

    # Environment colors
    scene.bg_color[:] = np.array([0.04, 0.045, 0.06], np.float32)
    scene.env_light[:] = np.array([0.9, 0.9, 1.2], np.float32)
    return scene


class ShotRunner:
    def __init__(self, args):
        self.args = args
        self.nframes = int(args.fps * args.seconds)
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.host = QWidget(); self.layout = QVBoxLayout(self.host); self.layout.setContentsMargins(0,0,0,0)
        self.view = AnalyticViewport(self.host, aacore_scene=build_scene())
        self.view.resize(args.w, args.h)
        self.layout.addWidget(self.view)
        self.host.resize(args.w, args.h)
        self.host.show()

        # Camera setup
        self.view.distance = 8.0
        self.view.cam_target = np.array([0.0, 0.0, 0.0], np.float32)
        self.view.yaw = 0.0; self.view.pitch = 0.2
        self.view._update_camera()

        # QC prim index (last added)
        self.scene = self.view.scene
        self.qc_idx = len(self.scene.prims) - 1
        self.out_dir = Path(self.args.out); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.i = 0

        # Timer for frame stepping
        self.timer = QTimer()
        self.timer.setInterval(0)  # as fast as possible
        self.timer.timeout.connect(self.step)

    def start(self):
        # Poll until GL is initialized (initializeGL has run)
        self.readyTimer = QTimer()
        self.readyTimer.setInterval(16)
        def _check_ready():
            # Trigger paints
            self.view.update()
            # Consider ready when shader program compiled and VAO created
            if getattr(self.view, 'prog', None) is not None and getattr(self.view, '_vao', None) is not None:
                self.readyTimer.stop()
                self.timer.start()
        self.readyTimer.timeout.connect(_check_ready)
        self.readyTimer.start()
        rc = self.app.exec()
        return rc

    def step(self):
        if self.i >= self.nframes:
            self.timer.stop()
            print(f"Done. Frames in {self.out_dir}")
            self.app.quit();
            return
        t = self.i / max(1, self.nframes - 1)
        e = ease_in_out_cubic(t)
        # Camera orbit + dolly
        self.view.yaw = 0.0 + 2.0*math.pi*(0.25*e)
        self.view.pitch = 0.18 + 0.06*math.sin(2*math.pi*e)
        self.view.distance = 8.0 - 0.6*e
        self.view._update_camera()
        # QC params
        qc = self.scene.prims[self.qc_idx]
        sc0, sc1 = 2.0, 6.0
        qc.params[0] = sc0 + (sc1 - sc0) * e
        qc.params[1] = 1.0 + 0.05*math.sin(2*math.pi*1.5*e)
        qc.params[2] = 0.02
        try:
            self.scene._notify()
        except Exception:
            pass
        # Render and capture
        try:
            self.view.makeCurrent()
        except Exception:
            pass
        self.view._draw_frame(debug_override=0)
        frame_path = self.out_dir / f"frame_{self.i:05d}.png"
        save_frame_rgba(self.view, str(frame_path))
        if (self.i % 10) == 0:
            print(f"Rendered {self.i+1}/{self.nframes}")
        self.i += 1


def main():
    ap = argparse.ArgumentParser(description='Render quasi-crystal hero take to PNG frames (no tessellation).')
    ap.add_argument('--w', type=int, default=1920)
    ap.add_argument('--h', type=int, default=1080)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--out', type=str, default=str(Path('renders') / 'hero_qc'))
    args = ap.parse_args()

    runner = ShotRunner(args)
    return runner.start()

if __name__ == '__main__':
    raise SystemExit(main())
