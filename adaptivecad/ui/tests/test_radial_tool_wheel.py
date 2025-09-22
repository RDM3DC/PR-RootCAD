import math
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF

from adaptivecad.ui.radial_tool_wheel import RadialToolWheel, ToolSpec


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_wheel(tools=None, **kwargs):
    tools = tools or [ToolSpec("A", "a"), ToolSpec("B", "b"), ToolSpec("C", "c"), ToolSpec("D", "d")]
    wheel = RadialToolWheel(None, tools=tools, **kwargs)
    wheel.resize(400, 400)
    return wheel


def test_radii_fit_to_edge(qapp):
    wheel = _make_wheel(fit_to_viewport_edge=True, edge_overshoot_px=18)
    r_outer, thickness, r_inner = wheel._radii(400, 300)
    half_min = 0.5 * min(400, 300)
    assert pytest.approx(r_outer) == half_min + 18
    wheel.set_fit_to_viewport_edge(False)
    wheel.set_outer_radius_ratio(0.5)
    r_outer2, _, _ = wheel._radii(400, 300)
    assert pytest.approx(r_outer2) == 0.5 * half_min
    wheel.deleteLater()


def test_hit_test_cardinal_segments(qapp):
    tools = [ToolSpec(str(i), str(i)) for i in range(4)]
    wheel = _make_wheel(tools=tools)
    cx, cy = wheel.width() * 0.5, wheel.height() * 0.5
    r_outer, thickness, r_inner = wheel._radii(wheel.width(), wheel.height())
    sample_radius = (r_outer + r_inner) * 0.5
    top = QPointF(cx, cy - sample_radius)
    right = QPointF(cx + sample_radius, cy)
    over_top, idx_top = wheel._hit_test(top)
    over_right, idx_right = wheel._hit_test(right)
    assert over_top and idx_top == 0
    assert over_right and idx_right == 1
    wheel.deleteLater()


def test_mutators_update_state(qapp):
    wheel = _make_wheel()
    wheel.set_opacity(0.4)
    wheel.set_thickness_ratio(0.2)
    wheel.set_edge_overshoot(8)
    wheel.set_text_scale(1.5)
    wheel.set_idle_rotation(12.0)
    assert wheel._opacity == pytest.approx(0.4)
    assert wheel._thickness_ratio == pytest.approx(0.2)
    assert wheel._edge_overshoot_px == 8
    assert wheel._text_scale == pytest.approx(1.5)
    assert wheel._idle_rotate_speed == pytest.approx(12.0)
    wheel.deleteLater()


def test_compute_alpha_smoothstep(qapp):
    wheel = _make_wheel()
    centre_alpha = wheel._compute_alpha(QPointF(wheel.width() * 0.5, wheel.height() * 0.5))
    edge_alpha = wheel._compute_alpha(QPointF(wheel.width() * 0.5, 0))
    assert centre_alpha == pytest.approx(0.0)
    assert edge_alpha > centre_alpha
    wheel.deleteLater()


def test_idle_timer_gates_when_inactive(qapp):
    wheel = _make_wheel()
    wheel.set_idle_rotation(0.0)
    wheel._alpha = 0.0
    wheel._hover_index = -1
    wheel._on_tick()
    assert not wheel._timer.isActive()
    wheel.set_idle_rotation(10.0)
    assert wheel._timer.isActive()
    wheel.deleteLater()
