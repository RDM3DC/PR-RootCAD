import json
from adaptivecad.sketch.units import Units, to_internal, from_internal
from adaptivecad.sketch.model import SketchDocument, Point, Line, Dimension, DimensionKind


def test_unit_conversions():
    assert to_internal(25.4, Units.IN) == 25.4 * 25.4 / 25.4  # 1 inch -> 25.4 mm
    assert abs(to_internal(1.0, Units.IN) - 25.4) < 1e-9
    assert abs(from_internal(25.4, Units.IN) - 1.0) < 1e-9
    assert abs(from_internal(10.0, Units.MM) - 10.0) < 1e-9


def test_roundtrip_json():
    doc = SketchDocument(units=Units.MM)
    doc.points.append(Point(id="p1", x=0.0, y=0.0))
    doc.points.append(Point(id="p2", x=10.0, y=0.0))
    doc.lines.append(Line(id="l1", a="p1", b="p2"))
    doc.dimensions.append(Dimension(id="d1", kind=DimensionKind.Horizontal, refs=["p1","p2"], value=None))

    s = doc.to_json()
    doc2 = SketchDocument.from_json(s)

    assert doc2.units == Units.MM
    assert len(doc2.points) == 2
    assert len(doc2.lines) == 1
    assert doc2.dimensions[0].kind == DimensionKind.Horizontal
    # Ensure JSON parse is valid
    json.loads(s)
