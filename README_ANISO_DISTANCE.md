
# Anisotropic Distance (FMM-lite) — Integration Bundle

This adds a compact **anisotropic fast-marching** baseline to AdaptiveCAD and exposes it in:
- **Playground/CLI**: `plugins/aniso_distance_plugin.py`
- **FreeCAD**: `freecad/AdaptiveCADPIToolpath/commands/CommandAnisoDistance.py`
- **Blender**: `blender_addons/adaptivecad_pia_aniso.py`

## Files

- `adaptive_pi/aniso_fmm.py` — library: ordered solver + geodesic backtracer
- `plugins/aniso_distance_plugin.py` — CLI / plugin hook
- `examples/aniso_distance_playground_demo.py` — Qt/console demo
- `freecad/AdaptiveCADPIToolpath/commands/CommandAnisoDistance.py` — FreeCAD command
- `blender_addons/adaptivecad_pia_aniso.py` — Blender add-on panel

## Quick start

```bash
# From repo root
python plugins/aniso_distance_plugin.py --mode demo --a 1.3 --b 1.0
python plugins/aniso_distance_plugin.py --mode trace
python examples/aniso_distance_playground_demo.py
```

## FreeCAD

Edit `freecad/AdaptiveCADPIToolpath/InitGui.py` and add:

```python
from .commands.CommandAnisoDistance import GuiCommand as _CmdAnisoDistance
Gui.addCommand("Adaptive_Aniso_Distance", _CmdAnisoDistance())
```

## Blender

Install `blender_addons/adaptivecad_pia_aniso.py` via **Edit → Preferences → Add-ons → Install...**

## Notes

- Uses NumPy only; no SciPy required.
- Solver uses a two-neighbor PDE update + one-sided fallback; soft-lands to isotropic FMM.
- Best with moderate anisotropy (κ ≲ 10–20); for extreme ratios consider rotated stencils/FM-LBR.
- Geodesic backtracer follows \(-G^-1\nabla T\) with bilinear sampling, good for visuals.
