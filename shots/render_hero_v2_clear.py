import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from OpenGL.GL import *
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

# Import project modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptivecad.aacore.sdf import (
    KIND_QUASICRYSTAL,
    KIND_SUPERELLIPSOID,
    Prim,
    Scene,
)
from adaptivecad.gui.analytic_viewport import AnalyticViewport

try:
    from PIL import Image

    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2.0


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
            Image.fromarray(arr, "RGBA").save(path)
        else:
            with open(path + ".raw", "wb") as f:
                f.write(arr.tobytes())


def build_scene() -> Scene:
    scene = Scene()
    # Superellipsoid shell via solid minus inner subtract
    # Outer: radius R, power p (boxier with higher p); anisotropy handled via transform scale
    R = 1.8
    pwr = 3.5
    thick = 0.12
    outer = Prim(KIND_SUPERELLIPSOID, [R, pwr, 0, 0], beta=0.0, color=(0.98, 0.94, 0.88))
    outer.set_transform(pos=[0, 0, 0], euler=[0, 0, 0], scale=[1.3, 1.3, 1.0])
    inner = Prim(
        KIND_SUPERELLIPSOID,
        [max(0.05, R - thick), pwr, 0, 0],
        beta=0.0,
        color=(0.0, 0.0, 0.0),
        op="subtract",
    )
    inner.set_transform(pos=[0, 0, 0], euler=[0, 0, 0], scale=[1.3, 1.3, 1.0])
    scene.add(outer)
    scene.add(inner)

    # Quasi-crystal field (union): params [scale, iso, thickness]
    # Start with very fine scale so it's barely visible initially
    qc = Prim(KIND_QUASICRYSTAL, [8.0, 1.0, 0.008, 0], beta=0.0, color=(0.15, 0.85, 0.9))
    scene.add(qc)

    # Environment colors (darker, more dramatic)
    scene.bg_color[:] = np.array([0.02, 0.025, 0.04], np.float32)
    scene.env_light[:] = np.array([1.1, 1.0, 1.3], np.float32)
    return scene


class ShotRunner:
    def __init__(self, args):
        self.args = args
        self.nframes = int(args.fps * args.seconds)
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.host = QWidget()
        self.layout = QVBoxLayout(self.host)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.view = AnalyticViewport(self.host, aacore_scene=build_scene())
        self.view.resize(args.w, args.h)
        self.layout.addWidget(self.view)
        self.host.resize(args.w, args.h)
        self.host.show()

        # Camera setup - start closer for better detail
        self.view.distance = 6.0
        self.view.cam_target = np.array([0.0, 0.0, 0.0], np.float32)
        self.view.yaw = 0.2
        self.view.pitch = 0.15
        self.view._update_camera()

        # QC prim index (last added)
        self.scene = self.view.scene
        self.qc_idx = len(self.scene.prims) - 1
        self.out_dir = Path(self.args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
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
            if (
                getattr(self.view, "prog", None) is not None
                and getattr(self.view, "_vao", None) is not None
            ):
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
            self.app.quit()
            return

        t = self.i / max(1, self.nframes - 1)  # 0..1 over full duration

        # Timing breakdown for clearer phases:
        # 0-25%: Shell alone, slow rotation
        # 25-40%: QC starts growing (pause camera)
        # 40-80%: Main growth with gentle orbit
        # 80-100%: Final reveal, pull back slightly

        if t < 0.25:
            # Phase 1: Shell reveal - very slow rotation
            phase_t = t / 0.25
            self.view.yaw = 0.2 + 0.8 * phase_t
            self.view.pitch = 0.15
            self.view.distance = 6.0
            # QC invisible (very fine)
            qc = self.scene.prims[self.qc_idx]
            qc.params[0] = 12.0  # very fine
            qc.params[2] = 0.003  # very thin
        elif t < 0.4:
            # Phase 2: QC emergence - camera pauses
            phase_t = (t - 0.25) / 0.15
            e = ease_in_out_cubic(phase_t)
            self.view.yaw = 1.0  # hold position
            self.view.pitch = 0.15
            self.view.distance = 6.0
            # QC becomes visible
            qc = self.scene.prims[self.qc_idx]
            qc.params[0] = 12.0 - 6.0 * e  # 12 -> 6
            qc.params[2] = 0.003 + 0.012 * e  # get thicker
        elif t < 0.8:
            # Phase 3: Main growth with gentle orbit
            phase_t = (t - 0.4) / 0.4
            e = ease_in_out_cubic(phase_t)
            self.view.yaw = 1.0 + 1.5 * phase_t  # gentle continued rotation
            self.view.pitch = 0.15 + 0.08 * math.sin(2 * math.pi * phase_t * 0.5)
            self.view.distance = 6.0 - 0.3 * e  # slight zoom in
            # QC main growth
            qc = self.scene.prims[self.qc_idx]
            qc.params[0] = 6.0 - 2.5 * e  # 6 -> 3.5 (features get bigger)
            qc.params[1] = 1.0 + 0.03 * math.sin(2 * math.pi * 1.2 * phase_t)  # subtle iso drift
            qc.params[2] = 0.015  # stable thickness
        else:
            # Phase 4: Final reveal - pull back slightly
            phase_t = (t - 0.8) / 0.2
            e = ease_in_out_cubic(phase_t)
            self.view.yaw = 2.5 + 0.3 * phase_t
            self.view.pitch = 0.23
            self.view.distance = 5.7 + 0.8 * e  # pull back for full view
            # QC final state
            qc = self.scene.prims[self.qc_idx]
            qc.params[0] = 3.5  # hold final size
            qc.params[1] = 1.0
            qc.params[2] = 0.015

        self.view._update_camera()

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
    ap = argparse.ArgumentParser(description="Render clear, slower quasi-crystal hero take.")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--out", type=str, default=str(Path("renders") / "hero_v2_clear"))
    args = ap.parse_args()

    runner = ShotRunner(args)
    return runner.start()


if __name__ == "__main__":
    raise SystemExit(main())
