"""Standalone demo showcasing the radial tool wheel overlay."""

import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from adaptivecad.ui.radial_tool_wheel import RadialToolWheelOverlay, ToolSpec


class DummyViewport(QWidget):
    """Simple placeholder viewport with a dark background."""

    def __init__(self) -> None:
        super().__init__()
        palette = QPalette(self.palette())
        palette.setColor(QPalette.Window, QColor(10, 12, 16))
        self.setAutoFillBackground(True)
        self.setPalette(palette)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    central = QWidget()
    layout = QVBoxLayout(central)
    viewport = DummyViewport()
    layout.addWidget(viewport)
    window.setCentralWidget(central)
    window.resize(1280, 800)
    window.show()

    wheel = RadialToolWheelOverlay(
        viewport,
        tools=[
            ToolSpec("Select", "select"),
            ToolSpec("Line", "line"),
            ToolSpec("Arc3", "arc3"),
            ToolSpec("Circle", "circle"),
            ToolSpec("Move", "move"),
            ToolSpec("Rotate", "rotate"),
            ToolSpec("Measure", "measure"),
            ToolSpec("Extrude", "extrude"),
        ],
        idle_rotate_deg_per_sec=10.0,
        enable_twist_shading=True,
    )
    wheel.toolActivated.connect(lambda tid, label: print("TOOL:", tid, label))

    sys.exit(app.exec())
