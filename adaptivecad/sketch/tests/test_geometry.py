import math

import pytest

from adaptivecad.sketch.geometry import (
    Intersection,
    Transform2D,
    Vec2,
    closest_point_on_segment,
    distance_point_to_line,
    is_colinear,
    line_circle_intersections,
    orientation,
    polyline_length,
    segment_circle_intersections,
    segment_intersection,
)


def test_vec2_arithmetic():
    a = Vec2(1.0, 2.0)
    b = Vec2(3.0, -1.0)
    assert a + b == Vec2(4.0, 1.0)
    assert a - b == Vec2(-2.0, 3.0)
    assert a * 2 == Vec2(2.0, 4.0)
    assert 2 * a == Vec2(2.0, 4.0)
    assert pytest.approx(a.length()) == math.hypot(1.0, 2.0)


def test_transform2d_roundtrip_translation_and_rotation():
    p = Vec2(1.0, 0.0)
    t = Transform2D.translation(2.0, 3.0)
    r = Transform2D.rotation(math.pi / 2)
    combined = t.combine(r)
    result = combined.apply(p)
    # Rotate (1,0) by 90deg -> (0,1); then translate -> (2,4)
    assert result.almost_equals(Vec2(2.0, 4.0))


def test_orientation_and_colinear():
    a = Vec2(0.0, 0.0)
    b = Vec2(1.0, 0.0)
    c = Vec2(0.0, 1.0)
    assert orientation(a, b, c) > 0  # CCW
    assert orientation(a, c, b) < 0  # CW
    assert is_colinear(a, b, Vec2(2.0, 0.0))


def test_distance_point_to_line():
    p = Vec2(1.0, 1.0)
    a = Vec2(0.0, 0.0)
    b = Vec2(2.0, 0.0)
    assert pytest.approx(distance_point_to_line(p, a, b)) == 1.0


def test_segment_intersection_crossing():
    a0 = Vec2(0.0, 0.0)
    a1 = Vec2(2.0, 2.0)
    b0 = Vec2(0.0, 2.0)
    b1 = Vec2(2.0, 0.0)
    inter = segment_intersection(a0, a1, b0, b1)
    assert inter.kind == "point"
    assert inter.point.almost_equals(Vec2(1.0, 1.0))


def test_segment_intersection_overlapping_segment():
    a0 = Vec2(0.0, 0.0)
    a1 = Vec2(2.0, 0.0)
    b0 = Vec2(1.0, 0.0)
    b1 = Vec2(3.0, 0.0)
    inter = segment_intersection(a0, a1, b0, b1)
    assert inter.kind == "segment"
    seg = inter.segment
    assert seg[0].almost_equals(Vec2(1.0, 0.0))
    assert seg[1].almost_equals(Vec2(2.0, 0.0))


def test_line_circle_intersections_tangent():
    pts = line_circle_intersections(Vec2(0.0, 1.0), Vec2(1.0, 0.0), Vec2(0.0, 0.0), 1.0)
    assert len(pts) == 1
    assert pts[0].almost_equals(Vec2(0.0, 1.0))


def test_segment_circle_intersections_two_points():
    a = Vec2(-2.0, 0.0)
    b = Vec2(2.0, 0.0)
    pts = segment_circle_intersections(a, b, Vec2(0.0, 0.0), 1.0)
    assert len(pts) == 2
    assert pts[0].almost_equals(Vec2(-1.0, 0.0))
    assert pts[1].almost_equals(Vec2(1.0, 0.0))


def test_closest_point_on_segment_and_polyline_length():
    seg = (Vec2(0.0, 0.0), Vec2(2.0, 0.0))
    pt = Vec2(1.0, 1.0)
    closest = closest_point_on_segment(pt, *seg)
    assert closest.almost_equals(Vec2(1.0, 0.0))

    polyline = [Vec2(0.0, 0.0), Vec2(0.0, 1.0), Vec2(1.0, 1.0)]
    assert pytest.approx(polyline_length(polyline)) == 2.0


def test_intersection_none_when_segments_apart():
    inter = segment_intersection(Vec2(0, 0), Vec2(1, 0), Vec2(0, 1), Vec2(1, 1))
    assert inter == Intersection.none()
