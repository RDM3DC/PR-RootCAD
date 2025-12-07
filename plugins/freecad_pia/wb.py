import json
import pathlib

import FreeCADGui

from .commands import register_commands

APP_DIR = pathlib.Path.home() / ".adaptivecad"
APP_DIR.mkdir(exist_ok=True)
SETTINGS = APP_DIR / "pia_settings.json"

DEFAULTS = {"beta": 0.2, "s0": 1.0, "clamp": 0.3, "segments": 128}


def load_settings():
    if SETTINGS.exists():
        try:
            return json.loads(SETTINGS.read_text())
        except Exception:
            pass
    SETTINGS.write_text(json.dumps(DEFAULTS, indent=2))
    return DEFAULTS.copy()


def save_settings(cfg):
    SETTINGS.write_text(json.dumps(cfg, indent=2))


def init_workbench():
    class PiAWB(FreeCADGui.Workbench):
        MenuText = "PiA"
        ToolTip = "Adaptive π (πₐ) tools"

        def Initialize(self):
            register_commands(load_settings, save_settings)
            self.appendToolbar("PiA", ["PiA_AdaptiveCircle", "PiA_Settings"])

        def GetClassName(self):
            return "Gui::PythonWorkbench"

    FreeCADGui.addWorkbench(PiAWB())
