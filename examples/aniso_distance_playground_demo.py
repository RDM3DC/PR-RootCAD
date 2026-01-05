"""
examples/aniso_distance_playground_demo.py

Tiny demo: tries PySide6/PyQt5 to expose two actions (compute / trace).
Falls back to console if Qt isn't available.
"""

import sys

try:
    pass
except Exception:
    print("NumPy required for this demo.")
    sys.exit(1)

from adaptive_pi.aniso_fmm import anisotropic_fmm, metric_const_aniso, trace_geodesic


def console_demo():
    nx, ny, a, b = 129, 129, 1.3, 1.0
    G = metric_const_aniso(nx, ny, a=a, b=b)
    src = (nx // 2, ny // 2)
    T = anisotropic_fmm(G, src, use_diagonals=True)
    mid = src[1]
    xs = [0, 5, 10, 20, 30, 40]
    row_x = [round(float(T[src[0] + k, mid]), 4) for k in xs]
    print("Sample T(+x):", row_x)
    start = (nx - 5, ny // 2)
    path = trace_geodesic(T, G, start, src, step=0.8)
    print("Trace length:", len(path), "end≈", path[-1])


def qt_demo():
    try:
        from PySide6 import QtWidgets
    except Exception:
        try:
            from PyQt5 import QtWidgets  # type: ignore
        except Exception:
            return False
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QWidget()
    w.setWindowTitle("Anisotropic Distance (FMM-lite)")
    layout = QtWidgets.QFormLayout(w)
    a_sp = QtWidgets.QDoubleSpinBox()
    a_sp.setRange(0.01, 100.0)
    a_sp.setValue(1.3)
    b_sp = QtWidgets.QDoubleSpinBox()
    b_sp.setRange(0.01, 100.0)
    b_sp.setValue(1.0)
    out = QtWidgets.QPlainTextEdit()
    out.setReadOnly(True)
    out.setFixedHeight(140)

    def run_demo():
        nx, ny = 129, 129
        a, b = a_sp.value(), b_sp.value()
        G = metric_const_aniso(nx, ny, a=a, b=b)
        src = (nx // 2, ny // 2)
        T = anisotropic_fmm(G, src, use_diagonals=True)
        mid = src[1]
        xs = [0, 5, 10, 20, 30, 40]
        row_x = [round(float(T[src[0] + k, mid]), 4) for k in xs]
        start = (nx - 5, ny // 2)
        path = trace_geodesic(T, G, start, src, step=0.8)
        out.setPlainText(f"T(+x)={row_x}\nGeodesic len={len(path)} end≈{path[-1]}")

    btn = QtWidgets.QPushButton("Run")
    btn.clicked.connect(run_demo)
    layout.addRow("G=diag(a,b) a:", a_sp)
    layout.addRow("b:", b_sp)
    layout.addRow(btn)
    layout.addRow("Output:", out)
    w.setLayout(layout)
    w.resize(360, 240)
    w.show()
    app.exec()
    return True


if __name__ == "__main__":
    if not qt_demo():
        console_demo()
