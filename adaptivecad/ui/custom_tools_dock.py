from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from adaptivecad.plugins.tool_registry import ToolRegistry


class CustomToolsDock(QDockWidget):
    def __init__(
        self, registry: ToolRegistry, *, parent: Optional[QWidget] = None, run_cb=None, pin_cb=None
    ):
        super().__init__("Custom Tools", parent)
        self.registry = registry
        self.run_cb = run_cb
        self.pin_cb = pin_cb

        body = QWidget(self)
        lay = QVBoxLayout(body)

        self.listw = QListWidget(self)
        btnrow = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.del_btn = QPushButton("Delete")
        self.pin_chk = QCheckBox("Pin to Wheel")

        btnrow.addWidget(self.pin_chk)
        btnrow.addStretch(1)
        btnrow.addWidget(self.run_btn)
        btnrow.addWidget(self.del_btn)

        lay.addWidget(self.listw)
        lay.addLayout(btnrow)
        self.setWidget(body)

        self.run_btn.clicked.connect(self._on_run)
        self.del_btn.clicked.connect(self._on_del)
        self.pin_chk.toggled.connect(self._on_pin)
        self.listw.itemSelectionChanged.connect(self._on_sel)
        self.listw.itemDoubleClicked.connect(lambda _: self._on_run())

        self.registry.changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        self.listw.clear()
        for m in sorted(self.registry.list(), key=lambda x: x.name.lower()):
            it = QListWidgetItem(f"{m.name}  ({m.id})")
            it.setData(Qt.UserRole, m.id)
            self.listw.addItem(it)
        self._sync_pin_chk()

    def _current_id(self) -> Optional[str]:
        it = self.listw.currentItem()
        return it.data(Qt.UserRole) if it else None

    def _sync_pin_chk(self):
        tid = self._current_id()
        if not tid:
            self.pin_chk.setChecked(False)
            self.pin_chk.setEnabled(False)
            return
        self.pin_chk.setEnabled(True)
        m = self.registry.get(tid)
        self.pin_chk.setChecked(bool(m and m.ui.get("pinned", False)))

    def _on_sel(self):
        self._sync_pin_chk()

    def _on_run(self):
        tid = self._current_id()
        if not tid:
            return
        if self.run_cb:
            self.run_cb(tid)
        else:
            try:
                self.registry.run(tid, parent_widget=self)
            except Exception as e:
                QMessageBox.warning(self, "Run failed", str(e))

    def _on_del(self):
        tid = self._current_id()
        if not tid:
            return
        if (
            QMessageBox.question(self, "Delete Tool", f"Delete custom tool '{tid}'?")
            == QMessageBox.Yes
        ):
            self.registry.remove(tid)

    def _on_pin(self, checked: bool):
        tid = self._current_id()
        if not tid:
            return
        self.registry.set_pinned(tid, checked)
        if self.pin_cb:
            self.pin_cb(tid, checked)
