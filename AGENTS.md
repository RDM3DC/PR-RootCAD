# 2D CAD SYSTEM COMPLETION ROADMAP (200 TASKS)

## FOUNDATIONS
1. Define 2D geometry kernel interfaces (entities, constraints, dimensions).
2. Choose/implement robust numeric types (double-precision, tolerances, epsilons).
3. Implement 2D vector, matrix, transform utilities with stable predicates.
4. Add geometric predicates (orientation, colinearity, intersection, distance).
5. Implement robust line/segment/arc/circle intersection functions.
6. Add bounding box (AABB) and spatial indexing (R-tree/Quadtree) for entities.
7. Establish document model: `SketchDocument` with layers, blocks, styles, units.
8. Add undo/redo command stack with batched/atomic transactions.
9. Implement constraint solver core scaffolding (variables, constraints, solver loop).
10. Wire save/load of sketches into project files independent of 3D workspace.

## ENTITY TYPES
11. Implement `Line2D` entity (two points; param, length, midpoint).
12. Implement `Polyline2D` (open) with vertex-level editing and bulge support.
13. Implement `Polygon2D` (closed polyline; area/orientation; hole flags).
14. Implement `Arc2D` (center, radius, start angle, end angle, direction).
15. Implement `Circle2D` (center, radius) with derived quadrants and tan points.
16. Implement `Ellipse2D` (center, rx, ry, rotation) and `EllipticArc2D`.
17. Implement `Bezier2D` (quadratic and cubic) with evaluation and splitting.
18. Implement `Spline2D` (NURBS-like) with knots, weights, evaluation, tessellation.
19. Implement `Rectangle2D` (by two points; by center/size; by 3-point).
20. Implement `RoundedRectangle2D` with uniform and per-corner radii.
21. Implement `RegularPolygon2D` (center, radius, sides, rotation).
22. Implement `Star2D` (inner/outer radius, points, rotation).
23. Implement `OffsetCurve2D` for lines, polylines, arcs, and splines.
24. Implement `ConstructionLine2D` (infinite) and `Ray2D`.
25. Implement `HatchRegion2D` (boundary + pattern + scale/angle).
26. Implement `FillRegion2D` (solid fill with winding/even-odd rules).
27. Implement `Text2D` (single-line) with font, size, alignment, rotation.
28. Implement `MText2D` (multi-line rich text) with wrapping and styles.
29. Implement `Leader2D` (arrow, landing, content) including multiline content.
30. Implement `Image2D` (raster reference) with scale, rotation, transparency.
31. Implement `Point2D` (markers) with symbol types and sizes.
32. Implement `BlockRef2D` (Insert) with scale/rotation/mirror and attributes.
33. Implement `DimensionBase2D` class (anchor points, style, overrides).
34. Implement `ParamCurve2D` interface for evaluation, tangent, normal.
35. Implement entity style resolution (by-layer, by-block, by-entity overrides).
36. Implement entity metadata (name, id, tags, custom properties).
37. Add entity selection handles and grips (hover/selection feedback).
38. Implement z-order and draw order management within a layer.
39. Define pickable bounds and hit-testing for each entity type.
40. Implement entity serialization (JSON) with versioning and migration.

## CONSTRAINTS
41. Add coincident constraint (point-on-point; point-on-curve).
42. Add colinear constraint (line with line/segment).
43. Add parallel constraint (line-to-line).
44. Add perpendicular constraint (line-to-line; line-to-axis).
45. Add horizontal constraint (line/segment orientation = 0°).
46. Add vertical constraint (line/segment orientation = 90°).
47. Add fixed constraint (fix point/vertex in model coordinates).
48. Add equal length constraint (segments equal).
49. Add equal radius constraint (circles/arcs equal).
50. Add concentric constraint (circles/arcs share center).
51. Add tangent constraint (line–circle/arc; arc–arc; spline–circle).
52. Add curvature continuity (G2) between curves.
53. Add distance constraint (point–point, point–line offset).
54. Add angle constraint (between lines/vectors; absolute and relative).
55. Add midpoint constraint (point locked to segment midpoint).
56. Add symmetric constraint (points/segments across axis or line).
57. Add mirror constraint reference (dynamic symmetry about construction line).
58. Add lock x/y constraint (point’s coordinate axis lock).
59. Add radius/diameter constraint (arc/circle explicit value).
60. Add length constraint (line/segment explicit value).
61. Add slope constraint (line angle relative to X).
62. Add curvature/radius-of-curvature at spline point.
63. Add equal slope constraint between two lines/curves at points.
64. Add coincident with grid node constraint (grid snapping as constraint).
65. Add point-on-curve parameter constraint (lock parameter t).
66. Add equal parameter constraint between two curves (matched t).
67. Add offset distance constraint for parallel curves.
68. Add arc length constraint between two points along a curve.
69. Add perpendicular at point (tangent normal) constraint on curve.
70. Add 2-point circle constraint (circle through two fixed points + radius).
71. Add 3-point circle constraint (circle through 3 points).
72. Add 2-point tangent circle constraint (tangent to two entities).
73. Add equal area constraint for closed regions.
74. Add lock orientation constraint for block refs.
75. Add equal scale constraint for block refs (uniform scaling).
76. Add anchored leader constraint (leader follows target).
77. Add associative hatch boundary linkage (updates with edges).
78. Add constraint groups with on/off toggles and priorities.
79. Add DOF (degrees of freedom) analyzer visualization.
80. Implement robust conflict detection and explainable resolution hints.

## DIMENSIONS
81. Implement linear dimension (horizontal/vertical/aligned).
82. Implement ordinate dimensions (X/Y baseline references).
83. Implement angular dimension (line–line, arc center-based).
84. Implement radial dimension (circle/arc radius).
85. Implement diameter dimension (circle/arc diameter).
86. Implement arc length dimension with chord/arc options.
87. Implement baseline/continued dimension chains.
88. Implement jogged dimension for large radii.
89. Implement center mark/centerline annotations.
90. Add tolerance (±, limits, fit) and precision controls per dimension.
91. Add arrowhead styles, sizes, and placement rules.
92. Add dimension text overrides with fields and expressions.
93. Add associative dimensioning (updates with geometry changes).
94. Add dim style manager (text, arrows, extension, tolerances, units).
95. Add inspection dimensions (flagged notes).
96. Add fractional/architectural unit formatting.
97. Add scale-aware dimensioning (annotation scale).
98. Add dimension break/avoidance for overlapping dims.
99. Add automatic placement assist and collision avoidance.
100. Add export-ready dimension rendering to PDF/SVG.

## SNAPPING & INPUT
101. Implement grid snap with configurable spacing and sub-divisions.
102. Implement object snaps: endpoint, midpoint, center, quadrant.
103. Implement object snaps: tangent, perpendicular, nearest, intersection.
104. Implement apparent intersection (visual) and extension snaps.
105. Implement node snap to spline/curve parameter (t snap with markers).
106. Implement angle snapping (polar tracking) with custom increments.
107. Implement distance/length snapping with incremental steps.
108. Implement dynamic input heads-up display near cursor.
109. Implement ortho mode (lock to 0/90/180/270° directions).
110. Implement temporary tracking lines (X/Y from picked points).
111. Implement relative/absolute coordinate input (e.g., @10,5).
112. Implement typed constraints in command bar (e.g., L=25, A=30°).
113. Implement snap priority and cycling (TAB to cycle snap points).
114. Implement snap filters (enable/disable specific snaps quickly).
115. Implement inferred constraints while drawing (auto horizontal, vertical, tangent).
116. Implement visual snap indicators with tooltips.
117. Implement magnet radius settings and adaptive snapping by zoom.
118. Implement gesture-based quick constraints (e.g., H/V stroke).
119. Implement coordinate readout and delta readout in status bar.
120. Implement snap to grid/non-printing construction geometry.

## EDITING TOOLS
121. Implement move tool with basepoint and orthogonal options.
122. Implement copy tool (single/multiple) with basepoint and array options.
123. Implement rotate tool with basepoint and angle/3-point rotate.
124. Implement scale tool (uniform, non-uniform) with reference length.
125. Implement mirror tool with line of symmetry (keep/delete source).
126. Implement offset tool for lines, polylines, arcs, and splines.
127. Implement fillet tool (radius, trim/extend options) across entities.
128. Implement chamfer tool (distance–distance and distance–angle).
129. Implement trim tool (single, power trim) for curve intersections.
130. Implement extend tool to nearest boundary with preview.
131. Implement break at point and break between two points.
132. Implement join tool to merge contiguous segments into polylines.
133. Implement explode tool to break blocks or polylines to primitives.
134. Implement stretch tool using crossing window selection.
135. Implement align tool (point/line; 2D affine fit) for blocks and groups.
136. Implement array tools: rectangular, polar, and path arrays.
137. Implement measure tools: distance, angle, radius, area.
138. Implement area calculation with island/hole detection.
139. Implement boundary detection (create region from selection).
140. Implement curve smoothing and simplify (Douglas–Peucker, curvature).
141. Implement convert between entity types (arc↔polyline bulge, spline↔polyline).
142. Implement curve projection to construction lines and axes.
143. Implement grips editing for all entities (move, rotate, scale, stretch).
144. Implement multi-entity property editing (style, layer, linetype).
145. Implement block creation from selection with attribute definitions.
146. Implement block editor (edit block definition in-place).
147. Implement reference editing with propagate changes toggle.
148. Implement path editing: add/remove/move vertices with constraints.
149. Implement segment length equalization on polylines.
150. Implement curvature comb visualization for splines.
151. Implement text editing (in-place), find/replace, and spellcheck hooks.
152. Implement hatch editing: pattern, scale, angle, boundary reassociate.
153. Implement image reference editing (clip, transparency, brightness).
154. Implement isolate/lock entities for focused edits.
155. Implement selection filters (by type, by layer, by property).
156. Implement quick select queries (property-based selection).
157. Implement linetype/lineweight management and preview.
158. Implement color by layer/entity and true-color picker.
159. Implement printing lineweight preview toggle.
160. Implement parametric arrays linked to constraints/expressions.

## LAYERS, BLOCKS, ANNOTATION
161. Implement layer manager (create, rename, delete, freeze, lock, color).
162. Implement layer states (save/restore sets of layer properties).
163. Implement layer filters and search.
164. Implement xref-like external block libraries.
165. Implement block attribute definitions (tags, prompts, defaults).
166. Implement attribute editing and synchronize across instances.
167. Implement annotation scaling (model space scale factor management).
168. Implement text styles (font, width factor, oblique angle).
169. Implement dimension styles (overrides, annotative support).
170. Implement multileader styles (arrow, landing, content style).
171. Implement hatch patterns library (ANSI, ISO, custom PAT support).
172. Implement centerline and symmetry line styles.
173. Implement title blocks and drawing templates as blocks.
174. Implement page setup manager (paper size, margins, plot styles).
175. Implement viewport frames for layouts (scale, lock, visual settings).
176. Implement printable vs non-printing layers and entities.
177. Implement annotation reorder and collision resolution.
178. Implement BOM/notes tables with CSV import/export.
179. Implement callouts and detail bubbles linked to views.
180. Implement sheet set basics (multi-sheet linking and renumbering).

## FILE I/O & PRINT
181. Implement SVG export with layers, styles, text, and dimensions.
182. Implement SVG import (paths, groups, transforms; map to entities).
183. Implement DXF export (R2018) for entities, layers, linetypes, dims.
184. Implement DXF import (common entities, blocks, dimensions, hatches).
185. Implement PDF export (vector) with page setups, lineweights, fonts.
186. Implement image exports (PNG/SVG/PDF) with background and DPI settings.
187. Implement JSON sketch export/import with robust schema versioning.
188. Implement clipboard interoperability (SVG/DXF fragments) copy/paste.
189. Implement printing pipeline (printer selection, preview, scaling).
190. Implement plot styles (monochrome, grayscale, custom CTB-like mapping).

## UX, SHORTCUTS, SETTINGS, QA
191. Implement command bar with history, autocomplete, and variables.
192. Implement customizable keyboard shortcuts and mouse gestures.
193. Implement customizable toolbars/ribbons with presets.
194. Implement status bar with toggles (SNAP, ORTHO, POLAR, GRID, OSNAP).
195. Implement settings dialog (units, formats, snaps, performance, UI).
196. Implement tutorial hints and first-run sample drawing.
197. Implement automated tests for geometry, constraints, snaps, and IO.
198. Implement performance profiling and large-drawing stress tests.
199. Implement accessibility (colors/contrast, keyboard-only navigation).
200. Implement documentation pages and short videos for 2D CAD workflows.

---

USAGE NOTE FOR AGENTS: START WITH FOUNDATIONS AND ENTITY TYPES, THEN ENABLE CORE CONSTRAINTS AND DIMENSIONS, FOLLOWED BY SNAPPING AND KEY EDIT TOOLS. SHIP IO/PRINT AND UX REFINEMENTS IN PARALLEL WITH TESTS AND DOCS. EACH ITEM SHOULD BECOME A TRACKABLE ISSUE WITH ACCEPTANCE CRITERIA AND TESTS.