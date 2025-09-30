from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List, Optional
from pathlib import Path
import json

from PySide6.QtCore import QObject, Signal

from .macro_engine import MacroDef, MacroParam, MacroStep, MacroEngine

# Store user tools under home directory
_DEF_DIR = Path.home() / ".adaptivecad" / "tools"
_DEF_DIR.mkdir(parents=True, exist_ok=True)

def _from_json(d: dict) -> MacroDef:
    params = [MacroParam(**p) for p in d.get("params", [])]
    steps = [MacroStep(**s) for s in d.get("steps", [])]
    return MacroDef(
        id=d["id"], name=d["name"], version=d.get("version","1.0"),
        author=d.get("author","user"), description=d.get("description",""),
        params=params, steps=steps, ui=d.get("ui", {})
    )

class ToolRegistry(QObject):
    changed = Signal()  # emitted on add/remove/update

    def __init__(self, bus):
        super().__init__()
        self._macros: Dict[str, MacroDef] = {}
        self._engine = MacroEngine(bus)
        self.dir = _DEF_DIR
        self.load_all()

    # ---------- Persistence ----------
    def _path(self, tool_id: str) -> Path:
        return self.dir / f"{tool_id}.ac_tool.json"

    def load_all(self):
        self._macros.clear()
        for p in self.dir.glob("*.ac_tool.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                m = _from_json(d)
                self._macros[m.id] = m
            except Exception:
                continue

    def save(self, macro: MacroDef):
        p = self._path(macro.id)
        dd = asdict(macro)
        p.write_text(json.dumps(dd, indent=2), encoding="utf-8")

    # ---------- CRUD ----------
    def list(self) -> List[MacroDef]:
        return list(self._macros.values())

    def get(self, tool_id: str) -> Optional[MacroDef]:
        return self._macros.get(tool_id)

    def add_or_update(self, macro: MacroDef, *, persist: bool = True):
        self._macros[macro.id] = macro
        if persist:
            self.save(macro)
        self.changed.emit()

    def remove(self, tool_id: str):
        if tool_id in self._macros:
            del self._macros[tool_id]
            p = self._path(tool_id)
            if p.exists():
                try: p.unlink()
                except Exception: pass
            self.changed.emit()

    # ---------- Execute ----------
    def run(self, tool_id: str, *, parent_widget=None, params_values=None):
        m = self.get(tool_id)
        if not m:
            raise RuntimeError(f"Unknown custom tool: {tool_id}")
        return self._engine.run(m, parent_widget=parent_widget, params_values=params_values)

    # ---------- Pinning ----------
    def pinned(self) -> List[MacroDef]:
        return [m for m in self._macros.values() if m.ui.get("pinned")]

    def set_pinned(self, tool_id: str, pinned: bool):
        m = self.get(tool_id)
        if not m:
            return
        m.ui["pinned"] = bool(pinned)
        self.add_or_update(m, persist=True)
