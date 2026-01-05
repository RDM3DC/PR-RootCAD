from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow

app = QApplication.instance() or QApplication([])
win = QMainWindow()
win.setWindowTitle("AdaptiveCAD Qt Smoke Test")
win.resize(320, 200)
win.show()
print("WINDOW_SHOWN")
# Quit after 1 second
QTimer.singleShot(1000, app.quit)
ret = app.exec()
print("APP_EXITED", ret)
