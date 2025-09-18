# AdaptiveCAD – AI Assistant Working Notes

This repo is a Windows-first Python project with an optional OCC path and a fast analytic (SDF) renderer + PySide6 GUI. Use these notes to be productive quickly and avoid common pitfalls.

## Architecture
Layers: `gui/` (PySide6 UI) → `aacore/` (SDF scene, transforms, packing) → `analytic/shaders/` (GLSL raymarcher). OCC-based tools live in `playground.py` and `command_defs.py`.
Analytic renderer: `adaptivecad/aacore/sdf.py` defines `Scene`, `Prim`, kind enums, transform (`xform.M` with translation in `M[:3,3]`), and GPU packing (`Scene.to_gpu_structs`). Shader side: `adaptivecad/analytic/shaders/sdf.vert|frag` implements the same kinds.
Main GUI: `adaptivecad/gui/playground.py` (`MainWindow`) with menus and commands; analytic viewport panel: `adaptivecad/gui/analytic_viewport.py` (OpenGL QOpenGLWidget + control panel).
Optional integrations: FreeCAD (`freecad/AdaptiveCADPIToolpath`), Blender (`blender_addons/...`), examples in `examples/`.

## Dev Workflows (Windows PowerShell)
- Activate dev env and run GUI Playground:
  - VS Code task: “Run AdaptiveCAD GUI Playground” or
  - `python -m adaptivecad.gui.playground`
- Quick env diagnostics: `check_environment.ps1` (PySide6, OCC). Qt only smoke test: `check_qt.py` or `qt_smoke_test.py`.
- Tests: activate the correct conda/venv, then `python -m pytest -q`. Some tests require OCC.
- Analytic viewport (standalone): from GUI menus “View → Show Analytic Viewport (Panel)”. Settings persist to `~/.adaptivecad_analytic.json`.

## Project Conventions
- GUI: PySide6 only (no PyQt mix). Use `QOpenGLWidget` for the analytic view. Keep UI actions in `MainWindow._create_menus_and_toolbar`.
- SDF math: numpy `float32` arrays; transforms are affine 4×4 with translation in last column; packing to shaders is column-major. Keep CPU and GLSL implementations in sync.
- Boolean op strings: `op='solid'|'subtract'` (see `Prim.op`). Respect `MAX_PRIMS` guard in panels.
- Persistence: analytic panel/user prefs merge into `~/.adaptivecad_analytic.json`. Don’t overwrite unknown keys; merge updates.
- Coding style: minimal, targeted edits; avoid broad refactors. No license headers. Prefer small helpers over large rewrites.

## Extending SDF Primitives (end-to-end)
1) Add CPU SDF + kind in `adaptivecad/aacore/sdf.py`:
   - Define `KIND_*` enum, implement distance/params, update `Scene.sdf()` dispatch if applicable, and `to_gpu_structs()` packing.
2) Add GLSL in `adaptivecad/analytic/shaders/sdf.frag`:
   - Mirror `KIND_*`, implement `sd_*` function, add to `map_scene` dispatch; keep params order aligned with CPU.
3) Expose in GUI: `adaptivecad/gui/analytic_viewport.py`:
   - Add button in “Primitives” group mapping to the new kind; set sensible default `params`, `beta`, and `color`.
4) Verify: run the Playground, open the Analytic Viewport Panel, add the new primitive, toggle debug modes (0..4) to inspect ID/depth/thickness.

## Sketch (2D) Overlay (MVP)
- Overlay tools live in `analytic_viewport.py`: Polyline/Circle/Rectangle with grid + grid snapping. Ctrl+Click selects vertices; drag to move; Delete removes selection. Save/Load Sketch JSON via the panel. Overlay auto-persists in `~/.adaptivecad_analytic.json`.
- Data model: `adaptivecad/sketch/model.py` (`SketchDocument`, entities, JSON) and `sketch/units.py`. Keep UI overlay exports compatible with `SketchDocument` for future OCC adapters.

## OCC Path (when available)
- `playground.py` conditionally enables OCC (pythonocc-core). Keep UI responsive when OCC is missing (fallback stubs already present). Commands add features and display via `qtViewer3d`.

## Common Gotchas
- Transforms: translation must end up in `xform.M[:3,3]` and pack correctly (column-major) for shaders and picking to work.
- Don’t import nonexistent legacy modules (e.g., `adaptivecad.core.backends`); the analytic path is under `adaptivecad/aacore`.
- Limit primitive count (`MAX_PRIMS`) and keep `params` vector length ≤ 4; use transform and color for everything else.

## Testing & Tasks
- Env: prefer the `adaptivecad` conda/venv (see `check_environment.ps1` output). Verify Qt with `check_qt.py`.
- Run tests: `python -m pytest -q` (some tests require OCC). Target specific files to iterate faster, e.g., `pytest tests/test_gui_startup.py -q`.
- GUI tasks (VS Code):
   - `Run AdaptiveCAD GUI Playground` → starts `python -m adaptivecad.gui.playground` in the right env.
   - Optional demo task: `Run AdaptiveCAD with 4D Chess` wires a plugin panel for ND chess.
- Useful scripts (repo root): `run_full_playground.py`, `run_enhanced_playground.py`, `run_analytic_viewport.py`, `debug_qt_launch.py`.
- Diagnostics: `check_environment.ps1` (PySide6/OCC), `check_occ.py`, `qt_smoke_test.py`.

## When Unsure
- Prefer reading the corresponding CPU+GLSL pair and the GUI panel button that uses it. Search for the kind name in `sdf.py`, `sdf.frag`, and `analytic_viewport.py`.
- Need library specifics? Check latest docs before coding. If your agent supports web/context tools, use them to fetch up-to-date PySide6/OpenGL/pythonocc idioms.