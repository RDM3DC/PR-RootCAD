#!/usr/bin/env python3
"""Test script for Mandelbulb and Klein bottle primitives."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PySide6.QtWidgets import QApplication

from adaptivecad.aacore.math import Xform
from adaptivecad.aacore.sdf import KIND_KLEIN, KIND_MANDELBULB, Prim, Scene
from adaptivecad.gui.analytic_viewport import AnalyticViewport


def main():
    app = QApplication.instance() or QApplication(sys.argv)

    # Create a scene with Mandelbulb and Klein bottle
    scene = Scene()

    # Classic Mandelbulb (power 8) - better parameters
    mandelbulb = Prim(KIND_MANDELBULB, [8.0, 2.0, 24, 0.8], beta=0.0, color=(0.9, 0.2, 0.1))
    mandelbulb.xform = mandelbulb.xform @ Xform.translate(-2.0, 0, 0)
    scene.add(mandelbulb)

    # Klein bottle - 75% smaller
    klein = Prim(KIND_KLEIN, [0.25, 2.0, 0.0, 0.025], beta=0.0, color=(0.1, 0.7, 0.9))
    klein.xform = klein.xform @ Xform.translate(2.0, 0, 0)
    scene.add(klein)

    # Power 4 Mandelbulb (different shape) - better parameters
    mandelbulb2 = Prim(KIND_MANDELBULB, [4.0, 2.0, 20, 0.6], beta=0.0, color=(0.2, 0.9, 0.1))
    mandelbulb2.xform = mandelbulb2.xform @ Xform.translate(0, 2.5, 0)
    scene.add(mandelbulb2)

    # Set nice lighting
    scene.bg_color[:] = np.array([0.01, 0.02, 0.05], np.float32)
    scene.env_light[:] = np.array([1.2, 1.0, 1.3], np.float32)

    print("Mandelbulb & Klein Bottle test scene created!")
    print("- Red: Classic Mandelbulb (power=8)")
    print("- Green: Power-4 Mandelbulb")
    print("- Blue: Klein Bottle (4D->3D projection)")
    print("\nControls:")
    print("- Mouse: Orbit camera")
    print("- Numbers 0-5: Debug modes")
    print("- Add Mandelbulb/Klein Bottle buttons: Create new fractals")

    # Launch in viewer
    viewport = AnalyticViewport(None, aacore_scene=scene)
    viewport.distance = 8.0
    viewport.show()

    return app.exec()


if __name__ == "__main__":
    exit(main())
