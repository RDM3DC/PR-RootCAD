#!/usr/bin/env python3
"""Test script for 4D torus primitive."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptivecad.aacore.sdf import Scene, Prim, KIND_TORUS4D
from adaptivecad.aacore.math import Xform
from adaptivecad.gui.analytic_viewport import AnalyticViewport
from PySide6.QtWidgets import QApplication
import numpy as np

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Create a scene with multiple 4D toruses
    scene = Scene()
    
    # Classic 4D torus (duocylinder)
    torus4d_1 = Prim(KIND_TORUS4D, [1.0, 0.8, 0.2, 0.0], beta=0.0, color=(0.9, 0.3, 0.9))
    scene.add(torus4d_1)
    
    # Different 4D slice 
    torus4d_2 = Prim(KIND_TORUS4D, [0.8, 1.0, 0.15, 0.5], beta=0.0, color=(0.3, 0.9, 0.9))
    torus4d_2.xform = torus4d_2.xform @ Xform.translate(2.5, 0, 0)
    scene.add(torus4d_2)
    
    # Animated 4D slice
    torus4d_3 = Prim(KIND_TORUS4D, [1.2, 0.6, 0.25, 0.0], beta=0.0, color=(0.9, 0.9, 0.3))
    torus4d_3.xform = torus4d_3.xform @ Xform.translate(-2.5, 0, 0)
    scene.add(torus4d_3)
    
    # Set nice lighting
    scene.bg_color[:] = np.array([0.02, 0.02, 0.08], np.float32)
    scene.env_light[:] = np.array([1.0, 0.95, 1.1], np.float32)
    
    print("4D Torus test scene created!")
    print("- Purple: Classic duocylinder (R1=1.0, R2=0.8, r=0.2, w=0.0)")
    print("- Cyan: Different proportions (R1=0.8, R2=1.0, r=0.15, w=0.5)")  
    print("- Yellow: Larger torus (R1=1.2, R2=0.6, r=0.25, w=0.0)")
    print("\nControls:")
    print("- Mouse: Orbit camera")
    print("- Numbers 0-5: Debug modes (try 5 for heatmap)")
    print("- Add 4D Torus button: Create new 4D tori")
    
    # Launch in viewer
    viewport = AnalyticViewport(None, aacore_scene=scene)
    viewport.distance = 6.0
    viewport.show()
    
    return app.exec()

if __name__ == '__main__':
    exit(main())