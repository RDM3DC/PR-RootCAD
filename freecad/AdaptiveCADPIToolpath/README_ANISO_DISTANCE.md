
# Anisotropic Distance — FreeCAD Workbench Hook

Add this command to `freecad/AdaptiveCADPIToolpath/InitGui.py`:

```python
from .commands.CommandAnisoDistance import GuiCommand as _CmdAnisoDistance
Gui.addCommand("Adaptive_Aniso_Distance", _CmdAnisoDistance())
```
Restart FreeCAD; the command appears under the workbench menu.
