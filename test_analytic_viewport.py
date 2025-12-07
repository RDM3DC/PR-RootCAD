"""Test script for the analytic viewport.

This standalone script runs the analytic viewport using PySide6.
It creates a simple scene with various primitive shapes to test the rendering.
"""

import os
import sys
import traceback

# Set up PyOpenGL debug output
os.environ["PYOPENGL_DEBUG"] = "debug"

print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

try:
    print("Importing PySide6...")
    from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

    print("Importing OpenGL...")
    from OpenGL.GL import *

    print("Importing GLRenderWidget...")
    from adaptivecad.analytic.viewport import GLRenderWidget

    print("Importing Scene...")
    from adaptivecad.analytic.scene import Scene

    print("Importing primitives...")
    from adaptivecad.analytic.primitives import (
        AnalyticBox,
        AnalyticCapsule,
        AnalyticCylinder,
        AnalyticSphere,
        AnalyticTorus,
    )
except Exception as e:
    print(f"Error during imports: {e}")
    traceback.print_exc()
    sys.exit(1)


def create_test_scene():
    """Create a test scene with various primitives"""
    scene = Scene()

    # Add primitives to the scene
    # Sphere at (0,0,0) with radius 1.0
    scene.add_primitive(AnalyticSphere(radius=1.0, position=(0, 0, 0), color=(1.0, 0.0, 0.0)))

    # Box at (-2, 0, 0) with size 1.5x1.0x1.0
    scene.add_primitive(
        AnalyticBox(size=(1.5, 1.0, 1.0), position=(-2, 0, 0), color=(0.0, 1.0, 0.0))
    )

    # Cylinder at (2, 0, 0) with radius 0.5 and height 2.0
    scene.add_primitive(
        AnalyticCylinder(radius=0.5, height=2.0, position=(2, 0, 0), color=(0.0, 0.0, 1.0))
    )

    # Capsule at (0, 2, 0) with radius 0.4 and height 1.5
    scene.add_primitive(
        AnalyticCapsule(radius=0.4, height=1.5, position=(0, 2, 0), color=(1.0, 1.0, 0.0))
    )

    # Torus at (0, -2, 0) with major radius 1.0 and minor radius 0.3
    scene.add_primitive(
        AnalyticTorus(
            major_radius=1.0, minor_radius=0.3, position=(0, -2, 0), color=(1.0, 0.0, 1.0)
        )
    )

    print("Created test scene with primitives:")
    print("- Red sphere at (0,0,0)")
    print("- Green box at (-2,0,0)")
    print("- Blue cylinder at (2,0,0)")
    print("- Yellow capsule at (0,2,0)")
    print("- Purple torus at (0,-2,0)")

    return scene


def main():
    print("Creating application...")
    app = QApplication(sys.argv)
    print("Creating main window...")
    win = QMainWindow()
    win.setWindowTitle("AdaptiveCAD Analytic Viewport Test")

    print("Creating central widget...")
    central = QWidget()
    win.setCentralWidget(central)

    print("Setting up layout...")
    layout = QVBoxLayout(central)

    print("Creating test scene...")
    scene = create_test_scene()

    print("Creating GL render widget...")
    gl_widget = GLRenderWidget(central)
    gl_widget.scene = scene  # Set scene after creating widget

    # Set initial camera distance for better view
    gl_widget.camera_distance = 8.0

    # Disable shaders if they cause problems
    try:
        # This is only for testing - we'll try using our fallback renderer
        print("Using simple drawing mode (fallback to non-shader rendering)")
        gl_widget.use_shaders = False
    except Exception as e:
        print(f"Note: Failed to disable shaders: {e}")

    layout.addWidget(gl_widget)

    print("Showing window...")
    win.resize(800, 600)
    win.show()

    print("Starting application event loop...")
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running analytic viewport test: {e}")
        traceback.print_exc()
        sys.exit(1)
