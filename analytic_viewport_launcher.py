#!/usr/bin/env python
"""
Minimal launcher for the Analytic Viewport Panel (PySide6).
This script is intended to be used by the VS Code task "Launch Analytic Viewport".
"""
import os
import sys

# Ensure project root is importable
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from PySide6.QtWidgets import QApplication, QMainWindow
except Exception as e:
    print("PySide6 is not available:", e)
    sys.exit(1)

try:
    from adaptivecad.gui.analytic_viewport import AnalyticViewportPanel
except Exception as e:
    print("Failed to import AnalyticViewportPanel:", e)
    sys.exit(1)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("AdaptiveCAD – Analytic Viewport Panel")
    panel = AnalyticViewportPanel(win)
    win.setCentralWidget(panel)
    win.resize(1100, 750)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
