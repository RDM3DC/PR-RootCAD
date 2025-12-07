#!/usr/bin/env python3
"""Test script for Menger sponge fractal primitive."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PySide6.QtWidgets import QApplication

from adaptivecad.aacore.math import Xform
from adaptivecad.aacore.sdf import KIND_MENGER, Prim, Scene
from adaptivecad.gui.analytic_viewport import AnalyticViewport


def main():
    app = QApplication.instance() or QApplication(sys.argv)

    # Create a scene with Menger sponges at different detail levels
    scene = Scene()

    # Level 2 Menger sponge
    menger2 = Prim(KIND_MENGER, [2, 1.0, 0, 0], beta=0.0, color=(0.8, 0.6, 0.2))
    menger2.xform = menger2.xform @ Xform.translate(-3.0, 0, 0)
    scene.add(menger2)

    # Level 3 Menger sponge (classic)
    menger3 = Prim(KIND_MENGER, [3, 1.0, 0, 0], beta=0.0, color=(0.9, 0.7, 0.3))
    scene.add(menger3)

    # Level 4 Menger sponge (high detail)
    menger4 = Prim(KIND_MENGER, [4, 1.0, 0, 0], beta=0.0, color=(0.7, 0.5, 0.1))
    menger4.xform = menger4.xform @ Xform.translate(3.0, 0, 0)
    scene.add(menger4)

    # Set nice lighting for fractal details
    scene.bg_color[:] = np.array([0.02, 0.03, 0.06], np.float32)
    scene.env_light[:] = np.array([1.4, 1.2, 1.0], np.float32)

    print("Menger Sponge fractal test scene created!")
    print("- Left: Level 2 (27 holes)")
    print("- Center: Level 3 (classic, 160 holes)")
    print("- Right: Level 4 (high detail, 1000+ holes)")
    print("\nControls:")
    print("- Mouse: Orbit camera")
    print("- Numbers 0-5: Debug modes")
    print("- Add Menger Sponge button: Create new fractals")
    print("- Perfect CSG operations - no tessellation artifacts!")

    # Launch in viewer
    viewport = AnalyticViewport(None, aacore_scene=scene)
    viewport.distance = 8.0
    viewport.show()

    return app.exec()


if __name__ == "__main__":
    exit(main())
