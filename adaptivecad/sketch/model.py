from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import json

from .units import Units

# --- Entities ---
@dataclass
class Point:
    id: str
    x: float
    y: float
    construction: bool = False

@dataclass
class Line:
    id: str
    a: str  # point id
    b: str  # point id
    construction: bool = False

@dataclass
class Arc:
    id: str
    center: str  # point id
    start: str   # point id
    end: str     # point id
    construction: bool = False

@dataclass
class Circle:
    id: str
    center: str  # point id
    radius: float
    construction: bool = False

@dataclass
class Bezier:
    id: str
    p0: str
    p1: str
    p2: str
    p3: Optional[str] = None  # optional for quadratic (p3 None)
    construction: bool = False

@dataclass
class Polyline:
    id: str
    verts: List[str]  # point ids in order
    closed: bool = False
    construction: bool = False

# --- Dimensions ---
class DimensionKind(str, Enum):
    Linear = "linear"
    Aligned = "aligned"
    Horizontal = "horizontal"
    Vertical = "vertical"
    Radius = "radius"
    Diameter = "diameter"

@dataclass
class Dimension:
    id: str
    kind: DimensionKind
    refs: List[str]  # point ids and/or entity ids (depending on kind)
    value: Optional[float] = None  # driving value (None = driven/annotation)
    style: Dict[str, Any] = field(default_factory=dict)

# --- Constraints ---
class ConstraintKind(str, Enum):
    Coincident = "coincident"
    Horizontal = "horizontal"
    Vertical = "vertical"
    Parallel = "parallel"
    Perpendicular = "perpendicular"
    Equal = "equal"
    Tangent = "tangent"
    Concentric = "concentric"

@dataclass
class Constraint:
    id: str
    kind: ConstraintKind
    refs: List[str]
    params: Dict[str, Any] = field(default_factory=dict)

# --- Document ---
@dataclass
class SketchDocument:
    units: Units = Units.MM
    points: List[Point] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    arcs: List[Arc] = field(default_factory=list)
    circles: List[Circle] = field(default_factory=list)
    beziers: List[Bezier] = field(default_factory=list)
    polylines: List[Polyline] = field(default_factory=list)
    dimensions: List[Dimension] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=lambda: {"version": 1})

    def to_json(self) -> str:
        d = asdict(self)
        # Enums -> value
        d["units"] = self.units.value
        for dim in d.get("dimensions", []):
            if isinstance(dim.get("kind"), Enum):
                dim["kind"] = dim["kind"].value
        for con in d.get("constraints", []):
            if isinstance(con.get("kind"), Enum):
                con["kind"] = con["kind"].value
        return json.dumps(d, separators=(",", ":"))

    @staticmethod
    def from_json(s: str) -> "SketchDocument":
        o = json.loads(s)
        doc = SketchDocument(units=Units(o.get("units", "mm")))
        def add_points(lst):
            for it in lst:
                doc.points.append(Point(**it))
        def add_lines(lst):
            for it in lst:
                doc.lines.append(Line(**it))
        def add_arcs(lst):
            for it in lst:
                doc.arcs.append(Arc(**it))
        def add_circles(lst):
            for it in lst:
                doc.circles.append(Circle(**it))
        def add_beziers(lst):
            for it in lst:
                doc.beziers.append(Bezier(**it))
        def add_polylines(lst):
            for it in lst:
                doc.polylines.append(Polyline(**it))
        def add_dims(lst):
            for it in lst:
                kind = DimensionKind(it["kind"]) if isinstance(it.get("kind"), str) else it.get("kind")
                it2 = dict(it); it2["kind"] = kind
                doc.dimensions.append(Dimension(**it2))
        def add_cons(lst):
            for it in lst:
                kind = ConstraintKind(it["kind"]) if isinstance(it.get("kind"), str) else it.get("kind")
                it2 = dict(it); it2["kind"] = kind
                doc.constraints.append(Constraint(**it2))
        add_points(o.get("points", []))
        add_lines(o.get("lines", []))
        add_arcs(o.get("arcs", []))
        add_circles(o.get("circles", []))
        add_beziers(o.get("beziers", []))
        add_polylines(o.get("polylines", []))
        add_dims(o.get("dimensions", []))
        add_cons(o.get("constraints", []))
        doc.meta = o.get("meta", {"version": 1})
        return doc
