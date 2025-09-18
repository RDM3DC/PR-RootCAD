import os, sys, traceback
print('PYTHON', sys.executable)
print('PLATFORM', sys.platform)
print('ENV QT_QPA_PLATFORM=', os.environ.get('QT_QPA_PLATFORM'))
print('QT_DEBUG_PLUGINS=', os.environ.get('QT_DEBUG_PLUGINS'))
try:
    import PySide6
    from PySide6 import QtCore, QtWidgets, QtGui
    print('PySide6 version', PySide6.__version__)
except Exception as e:
    print('PySide6 import failed:', e)
    traceback.print_exc()
    sys.exit(1)

# Force Qt plugin debug
os.environ['QT_DEBUG_PLUGINS'] = '1'
print('Set QT_DEBUG_PLUGINS=1')

try:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    print('QApplication created')
    win = QtWidgets.QMainWindow()
    win.setWindowTitle('Debug Qt Launch')
    win.resize(320,200)
    win.show()
    print('Window shown; entering event loop for 1s')
    QtCore.QTimer.singleShot(1000, app.quit)
    ret = app.exec()
    print('Event loop exited:', ret)
except Exception as e:
    print('ERROR during Qt startup:', e)
    traceback.print_exc()
    sys.exit(2)

print('DONE')
