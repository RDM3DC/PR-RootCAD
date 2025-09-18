# AdaptiveCAD 2D Sketch Suite Roadmap

This plan outlines a pragmatic, incremental path to a full 2D CAD sketch environment with units, dimensions, constraints, and solid features (extrude/revolve), integrated with AdaptiveCAD’s existing OCC and Analytic (SDF) pipelines.

## Goals
- Fast, useful MVP: persistent 2D sketches with grid/snaps, units, basic shapes, driven dimensions, and OCC-based extrude.
- Extensible core: constraint graph and solver that can grow from simple H/V/coincident toward a robust set.
- Dual backends: OCC solids for production; SDF fallbacks for analytic previews/exports.

## Guiding Principles
- UX-first: minimal clicks, intuitive snapping, live feedback, clean dimension editing.
- Project-centric: units are per-project; sketches serialize into project files.
- Decoupled layers: UI → Sketch model → Geometry kernel adapters (OCC / SDF) via a clean API.

---

## Phase 0 — Foundations (Day 0-1)
- Project Units
  - Add project-wide units setting (mm/in) with conversion utils.
  - Display/input normalization (store metric internally, convert for UI).
- Sketch Data Model
  - `SketchDocument` (units, entities, dimensions, constraints, metadata)
  - Entities: `Point`, `Line`, `Arc`, `Circle`, `Bezier`, `Polyline` (closed flag)
  - Dimensions: `DimLinear`, `DimAligned`, `DimHorizontal`, `DimVertical`, `DimRadius`, `DimDiameter` (driven)
  - Constraints (subset): `Coincident`, `Horizontal`, `Vertical`, `Parallel`, `Perpendicular`, `Equal`, `Tangent` (store-only initially)
- Serialization
  - JSON save/load (embedded in project file); versioned schema.

Deliverables:
- `adaptivecad/sketch/model.py`: dataclasses for entities/dimensions/constraints.
- `adaptivecad/sketch/units.py`: unit handling and conversion helpers.
- Basic tests for model serialization.

---

## Phase 1 — Sketcher MVP (Days 2-4)
- UI & Tools (Analytic Viewport Panel integration)
  - Dedicated Sketch mode UI: toolbar and property palette.
  - Drawing tools: Line, Rectangle, Circle, Arc (3-pt), Polyline; point snapping.
  - Grid + snaps: grid toggle, snap-to-grid, endpoint/midpoint/center snaps.
  - Selection + edit: move vertices, delete, close polyline, toggle construction.
- Dimensions (Driven)
  - Place dimensions; label in current units; live updates on geometry edits.
  - Edit dimensions as annotations (no solver yet): update only the label.
- Persistence
  - Sketch stored in project; load/save roundtrip.

Deliverables:
- `adaptivecad/sketch/ui.py`: sketch canvas additions reused by `AnalyticViewportPanel` overlay.
- Buttons in `AnalyticViewportPanel`: New Sketch, Line, Arc, Circle, Rect, Polyline, Dimension, Grid, Snap.
- JSON persistence wired to project save/open.

---

## Phase 2 — Solid Features (Days 5-7)
- OCC Feature: Extrude
  - Closed profiles → wire → face → prism; options: height, direction, add/cut/intersect.
  - Multi-contour (outer loop + holes) from nested curves; automatic orientation.
- OCC Feature: Revolve
  - Axis pick; angle; add/cut/intersect.
- Feature Management
  - Link feature to sketch reference (non-destructive); regenerate on sketch edits.
- Fallback (Analytic/SDF Preview)
  - Extruded polygon SDF primitive for preview when OCC not available.

Deliverables:
- `adaptivecad/sketch/occ_adapter.py`: build wires/faces from sketch; extrude/revolve.
- UI: “Extrude Sketch…”, “Revolve Sketch…” dialogs.
- Tests for profile to wire/face conversion (simple cases).

---

## Phase 3 — Constraint Solving (Days 8-12)
- Constraint Graph
  - Build graph over entities/points with param variables (x,y,r,angle,length).
  - Error functions per constraint; objective = sum of squared errors.
- Solver
  - Gauss-Newton/Levenberg-Marquardt iterative solve; simple damping and bounds.
  - Driving dimensions: treat as hard constraints with target values.
- UX
  - Apply constraints via toolbar; badges on geometry; hover tips.
  - Solve-on-edit with guards (timeout, fallback, rollback on divergence).

Deliverables:
- `adaptivecad/sketch/solver.py`: minimal numeric solver (LM) with Jacobians.
- Constraint set: Coincident, H/V, Parallel, Perpendicular, Equal Length, Concentric, Tangent (arc/line, circle/line).
- Unit tests per constraint and composite solve scenarios.

---

## Phase 4 — Power Tools (Days 13-16)
- Patterns: Rectangular, Circular; array counts and spacing/angle.
- Trim/Extend, Fillet/Chamfer (2D), Offset (single/bi-direction with caps).
- Construction geometry toggle; hide/show; layer-like grouping.
- Sketch Inspect: open loop detection, self-intersection, tolerance checks.

Deliverables:
- `adaptivecad/sketch/ops.py`: curve ops (offset, fillet, trim), using robust libs or custom geometry.
- Additional UI actions in Sketch toolbar.

---

## Phase 5 — IO & Interop (Days 17-20)
- DXF Import/Export (R12 or R2000)
  - Entities: LINE, ARC, CIRCLE, LWPOLYLINE; dimension export optional (MText).
- SVG Export (for quick sharing)
- Image Underlay: attach raster as reference (scale/origin/rotate).

Deliverables:
- `adaptivecad/sketch/dxf.py`, `adaptivecad/sketch/svg.py`.
- Simple import/export dialogs.

---

## Phase 6 — Polish & Integration (Days 21-24)
- Dimension styles (font/size/arrowheads); annotation layers.
- Better selection/filtering (by type, by construction).
- Performance: partial redraw, spatial index for snaps/hit-tests.
- Docs & Demos: tutorial sketches, sample projects.

Deliverables:
- Styling options; config in settings.
- Demo scripts in `shots/` to render sketch-to-solid workflows.

---

## API & Data Structures
- `SketchDocument`
  - `units: str` ("mm"|"in")
  - `entities: List[Entity]` (Point, Line, Arc, Circle, Bezier, Polyline)
  - `dimensions: List[Dimension]`
  - `constraints: List[Constraint]`
  - `meta: dict` (name, author, timestamps, version)
- `Entity` fields
  - IDs, construction flag, style, reference geometry (e.g., center/radius/angles for circle/arc)
- `Constraint`
  - Type, refs (entity IDs / point refs), parameters (e.g., target angle)
- `Dimension`
  - Type, refs, value (driving), style; measured distance/radius calculation

---

## UX Sketch (Initial)
- Sketch Mode Toolbar
  - New, Select, Line, Arc, Circle, Rect, Polyline, Trim/Extend, Fillet, Offset, Dimension, Constraints, Grid, Snaps, Units.
- Property Panel
  - Entity parameters; dimension value editing; constraint toggles.
- Context Menu
  - Convert to Construction, Close Polyline, Make Parallel/Perp, Equal, Fix, Delete.

---

## Risks & Mitigations
- Constraint complexity: start with small set and numeric solver; avoid full-blown symbolic from day one.
- Robust offsets/fillets: lean on proven geometric routines; clamp edge cases.
- OCC tolerance quirks: normalize/skew inputs; add small epsilons; validate wires.

---

## Success Criteria
- Create a sketch with dimensions in mm/in, extrude to a valid solid, modify a dimension, regenerate successfully.
- Constraint demo: rectangle with H/V/Equal, tangent circle; solve reliably.
- Export/import: roundtrip a simple DXF profile.

---

## Stretch (Post-v1)
- Loft/Sweep from multiple profiles; guide curves.
- Parametric templates (bolt circles, gears, slots).
- Constraint presets & dimension-driven configs.
- Collaborative sketch editing (future).

---

## Implementation Steps (10-Step Plan)
Below is a concrete 10-step execution plan that maps the roadmap to actionable, verifiable milestones.

1) Initialize Sketch Core and Units
- Create `adaptivecad/sketch/` package with `model.py` and `units.py`.
- Implement dataclasses: `SketchDocument`, `Point`, `Line`, `Arc`, `Circle`, `Bezier`, `Polyline`, `Dimension*` types, `Constraint`.
- Add unit helpers (mm/in) and project setting integration. Write serialization tests (roundtrip JSON).
- Acceptance: New/empty sketch saved/loaded; unit conversion helpers verified.

2) Wire Sketch Mode into Analytic Viewport Panel
- Add “Sketch Mode” toggle to `AnalyticViewportPanel` with grid display and snap toggles.
- Implement selection/move/delete for points/segments; property panel shows entity parameters.
- Persist the sketch within project save/open.
- Acceptance: Draw points, see grid, select/move items; sketch survives save+reload.

3) Core Drawing Tools and Snapping
- Tools: Line, Rectangle, Circle, Arc (3-point), Polyline with “Close” action and construction flag.
- Snaps: grid, endpoint, midpoint, center (with hover hints); hit-test tolerances.
- Acceptance: User can draft a closed profile with reliable snaps and toggle construction geometry.

4) Dimensions (Driven annotations first)
- Add placement of linear/aligned/horizontal/vertical/radius/diameter dimensions (labels reflect units).
- Dimensions update when referenced geometry moves; values editable as annotations (no geometry solve yet).
- Acceptance: Place/edit dimension labels; labels update live on geometry edits.

5) OCC Profile Conversion and Extrude (Add/Cut/Intersect)
- Implement `occ_adapter.py`: build OCC wires/faces from closed profiles (handle numeric tolerances).
- “Extrude Sketch…” dialog: height, direction, operation (add/cut/intersect).
- Link result to source sketch for regeneration on edit.
- Acceptance: Extrude rectangle to solid; modify width dimension → update solid.

6) Revolve and Multi-Loop Profiles
- Add revolve (axis pick, angle, operation). Support nested loops (outer + holes) with proper orientation.
- Tests for donut/slot-like profiles; UI affordance for axis selection.
- Acceptance: Revolve circle profile into a torus-like solid; multi-loop face yields holes when extruded.

7) Constraint Graph and Minimal Solver
- Introduce constraint graph (variables: x,y,r,angle,length). Implement LM solver with damping and bounds.
- Enable constraints: Coincident, Horizontal, Vertical, Parallel, Perpendicular, Equal Length.
- Solve-on-edit with guardrails (iteration/time caps, rollback on divergence).
- Acceptance: Rectangle with H/V + Equal locks to square; coincident vertices merge reliably.

8) Tangency, Concentric, and Driving Dimensions
- Add Tangent (line/arc, circle/line) and Concentric constraints; promote dimensions to driving (set value → solve).
- Improve stability (scaling, parameterization) and UX (badges, hover tips, constraint list editing).
- Acceptance: Tangent circle to line stays tangent after moves; changing a driving dimension repositions geometry.

9) Power Tools and Patterns
- Implement Trim/Extend, Fillet/Chamfer (2D), Offset (single/bi-direction with caps), Rect/Circ patterns.
- Add construction visibility toggle, selection filters (by type, construction), “Inspect Sketch” checks (open loops, intersections).
- Acceptance: Offset closed profile with caps, pattern holes around circle, fillet corners with set radius.

10) IO, Underlays, and Polish
- DXF import/export (LINE, ARC, CIRCLE, LWPOLYLINE), SVG export, image underlay with scale/origin.
- Dimension styles (font/size/arrowheads), performance passes (partial redraw, spatial index), user docs and demos.
- Acceptance: Roundtrip simple DXF; export SVG preview; demo scripts show sketch→solid workflows end-to-end.

