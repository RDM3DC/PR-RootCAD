"""Test script for the analytic viewport.

This standalone script runs the analytic viewport in a separate PyQt6 process.
It can be launched independently or from within the AdaptiveCAD GUI.
"""
import sys
import os
import traceback
from pathlib import Path

# Set up PyOpenGL debug output
os.environ['PYOPENGL_DEBUG'] = 'debug'

print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

try:
    print("Importing PyQt6...")
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    print("Importing QOpenGLWidget...")
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    print("Importing OpenGL...")
    from OpenGL.GL import *
    print("Importing AnalyticViewport...")
    from adaptivecad.gui.analytic_viewport import AnalyticViewport
except Exception as e:
    print(f"Error during imports: {e}")
    traceback.print_exc()
    sys.exit(1)

def main():
    print("Creating application...")
    app = QApplication(sys.argv)
    print("Creating main window...")
    win = QMainWindow()
    win.setWindowTitle("AdaptiveCAD Analytic Viewport (No Triangles)")
    
    print("Creating central widget...")
    central = QWidget()
    win.setCentralWidget(central)
    
    print("Setting up layout...")
    layout = QVBoxLayout(central)
    
    print("Creating analytic viewport...")
    viewport = AnalyticViewport(central)
    layout.addWidget(viewport)
    
    print("Showing window...")
    win.resize(960, 720)
    win.show()
    
    print("Starting application event loop...")
    sys.exit(app.exec())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running analytic viewport: {e}")
        traceback.print_exc()
        sys.exit(1)

def main():
    print("Creating application...")
    app = QApplication(sys.argv)
    print("Creating main window...")
    win = QMainWindow()
    win.setWindowTitle("Analytic Viewport Test")
    
    print("Creating central widget...")
    central = QWidget()
    win.setCentralWidget(central)
    
    print("Setting up layout...")
    layout = QVBoxLayout(central)
    
    print("Creating analytic viewport...")
    viewport = AnalyticViewport(central)
    layout.addWidget(viewport)
    
    print("Showing window...")
    win.resize(800, 600)
    win.show()
    
    print("Starting application event loop...")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()