from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from adaptivecad.ai.intent_router import CADActionBus, chat_with_tools
from adaptivecad.plugins.tool_registry import ToolRegistry


class _LLMWorker(QThread):
    finished_text = Signal(str)

    def __init__(
        self,
        text: str,
        tools: List[str],
        bus: CADActionBus,
        model: str,
        prior: List[Dict[str, str]] | None = None,
        registry: Optional[ToolRegistry] = None,
    ):
        super().__init__()
        self.text = text
        self.tools = tools
        self.bus = bus
        self.model = model
        self.prior = prior or []
        self.registry = registry

    def run(self) -> None:
        try:
            output = chat_with_tools(
                self.text,
                available_tools=self.tools,
                bus=self.bus,
                model=self.model,
                prior_messages=self.prior,
                registry=self.registry,
            )
        except Exception as exc:  # pragma: no cover - network/runtime errors
            output = f"LLM error: {exc}"
        self.finished_text.emit(output)


class AICopilotDock(QDockWidget):
    """Dockable chat/command panel that proxies CAD actions via the LLM."""

    def __init__(
        self,
        parent=None,
        *,
        available_tools: List[str],
        bus: CADActionBus,
        model: str = "gpt-4o-mini",
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        super().__init__("AI Copilot", parent)
        self.setObjectName("AICopilotDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._bus = bus
        self._tools = available_tools
        self._model = model
        self._registry = registry
        # simple in-memory chat history for this dock
        self._history = []  # type: List[Dict[str, str]]

        body = QWidget(self)
        layout = QVBoxLayout(body)

        self.view = QTextEdit(self)
        self.view.setReadOnly(True)

        self.input = QLineEdit(self)
        self.input.setPlaceholderText(
            "Describe what you want... e.g., 'make a 100 mm circle and extrude it 5 mm'"
        )

        button_row = QHBoxLayout()
        self.send_btn = QPushButton("Send")
        self.model_label = QLabel(self._model)
        self.model_label.setToolTip("Model used for requests")

        button_row.addWidget(self.model_label)
        button_row.addStretch(1)
        button_row.addWidget(self.send_btn)

        layout.addWidget(self.view)
        layout.addLayout(button_row)
        layout.addWidget(self.input)

        self.setWidget(body)

        self.send_btn.clicked.connect(self._on_send)
        self.input.returnPressed.connect(self._on_send)

        self._worker = None

    def _append(self, who: str, text: str) -> None:
        self.view.append(f"<b>{who}:</b> {text}")

    def _on_send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return

        self._append("You", text)
        # record user turn
        self._history.append({"role": "user", "content": text})
        self.input.clear()
        self.send_btn.setEnabled(False)
        self._worker = _LLMWorker(
            text, self._tools, self._bus, self._model, prior=self._history, registry=self._registry
        )
        self._worker.finished_text.connect(self._on_done)
        self._worker.start()

    def _on_done(self, text: str) -> None:
        self._append("Copilot", text)
        # record assistant turn
        self._history.append({"role": "assistant", "content": text})
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        self.send_btn.setEnabled(True)
