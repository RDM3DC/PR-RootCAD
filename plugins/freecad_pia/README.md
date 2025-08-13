# FreeCAD Workbench: Adaptive-π (πₐ) Tools

## Install
1. Ensure your AdaptiveCAD repo is on `sys.path` when FreeCAD starts (e.g., symlink into `~/.FreeCAD/Mod/AdaptiveCAD`, or add path in `InitGui.py`).
2. Copy `plugins/freecad_pia` into `~/.FreeCAD/Mod/freecad_pia` (Linux/macOS) or `%APPDATA%\FreeCAD\Mod\freecad_pia` (Windows).
3. Start FreeCAD → select the **PiA** workbench.

## Tools (v1)
- **Adaptive Circle**: builds a sketch wire polyline using `adaptivecad.pi.kernel.make_adaptive_circle()`.
- Settings dialog exposes β, s₀, clamp; persisted in `~/.adaptivecad/pia_settings.json`.
