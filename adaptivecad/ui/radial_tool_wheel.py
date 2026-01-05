"""PySide6 radial "holographic" tool wheel overlay for AdaptiveCAD.

- Rotates slowly by default; can also spin with mouse wheel or drag.
- Clickable segments emit toolActivated(tool_id, label).
- Fades out as cursor approaches viewport center.

Drop-in: parent it to your viewport widget; it auto-resizes/overlays and
passes events through unless the pointer is over the ring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PySide6.QtCore import QElapsedTimer, QEvent, QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QRadialGradient,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget


@dataclass
class ToolSpec:
    """Simple tool descriptor used to populate the wheel."""

    label: str
    tool_id: Optional[str] = None  # your internal name, e.g., "line", "arc3", etc.
    children: List["ToolSpec"] = field(default_factory=list)

    def clone(self) -> "ToolSpec":
        return ToolSpec(self.label, self.tool_id, [child.clone() for child in self.children])

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RadialToolWheel(QWidget):
    """
    Translucent overlay widget that draws a rotating, clickable ring of tool
    segments around the viewport edges. It fades out as the mouse approaches
    the viewport centre so normal modelling workflows stay unobstructed.
    """

    toolActivated = Signal(str, str)  # (tool_id, label)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        tools: Optional[List[ToolSpec]] = None,
        *,
        outer_radius_ratio: float = 0.46,  # fraction of min(width, height)/2 when not fitting to edge
        thickness_ratio: float = 0.09,  # ring thickness fraction of min(wh)/2
        fade_full_at_center_ratio: float = 0.05,  # <= this distance => fully transparent
        fade_full_at_edge_ratio: float = 0.45,  # >= this distance => fully opaque
        idle_rotate_deg_per_sec: float = 0.0,
        hover_glow: QColor = QColor(0, 255, 255, 180),  # cyan-ish glow
        base_ring_color: QColor = QColor(0, 200, 255, 95),
        segment_line_color: QColor = QColor(0, 255, 255, 140),
        text_color: QColor = QColor(200, 255, 255, 220),
        font_point_size: int = 9,
        enable_twist_shading: bool = True,  # faux Mobius vibe
        # When True, the wheel auto-sizes so its outer edge extends slightly
        # beyond the viewport borders (most of the ring is inside the view,
        # but the outside edge is clipped). If enabled, outer_radius_ratio is
        # ignored for the outer radius computation.
        fit_to_viewport_edge: bool = False,
        # Extra pixels to push the outer edge past the viewport when fitting
        # to edges; small values like 8-16px work well.
        edge_overshoot_px: int = 12,
        min_visible_alpha: float = 0.18,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # toggled dynamically
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

        self._tools: List[ToolSpec] = tools or []
        self._outer_ratio = outer_radius_ratio
        self._thickness_ratio = thickness_ratio
        self._fade_center_ratio = fade_full_at_center_ratio
        self._fade_edge_ratio = fade_full_at_edge_ratio

        self._hover_glow = hover_glow
        self._base_color = base_ring_color
        self._line_color = segment_line_color
        self._text_color = text_color
        self._font_pt = font_point_size
        self._twist = enable_twist_shading
        self._fit_edge = bool(fit_to_viewport_edge)
        self._edge_overshoot_px = int(edge_overshoot_px)
        self._min_alpha = max(0.0, min(1.0, float(min_visible_alpha)))

        self._opacity: float = 1.0
        self._text_scale: float = 1.0
        self._needs_repaint: bool = True
        self._inactive_interval_ms = 160

        self._rotation_deg: float = 0.0
        self._dragging: bool = False
        self._drag_last_angle: Optional[float] = None
        self._hover_index: int = -1
        self._hover_sub_index: int = -1
        self._submenu_open: bool = False
        self._active_group: int = -1
        self._submenu_tools: List[ToolSpec] = []
        self._press_ring: Optional[str] = None
        self._press_index: int = -1
        self._alpha: float = 1.0

        # Animation / update timer (~60 Hz)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(16)

        self._idle_rotate_speed = idle_rotate_deg_per_sec  # deg/s
        self._time_ref = QElapsedTimer()
        self._time_ref.start()

        # Track parent resize/move to stay full-coverage overlay
        if parent is not None:
            parent.installEventFilter(self)

    # ------------ Public API ------------

    def _open_submenu(self, index: int) -> None:
        if index < 0 or index >= len(self._tools):
            return
        group = self._tools[index]
        if group.is_leaf():
            self._close_submenu()
            return
        self._active_group = index
        self._submenu_tools = [child.clone() for child in group.children]
        self._submenu_open = True
        self._hover_sub_index = -1
        self._mark_dirty()

    def _close_submenu(self) -> None:
        if not self._submenu_open and self._active_group == -1:
            return
        self._submenu_open = False
        self._active_group = -1
        self._submenu_tools.clear()
        self._hover_sub_index = -1
        self._mark_dirty()

    def _activate_outer(self, index: int) -> None:
        if index < 0 or index >= len(self._tools):
            return
        spec = self._tools[index]
        if not spec.is_leaf():
            if self._submenu_open and self._active_group == index:
                self._close_submenu()
            else:
                self._open_submenu(index)
            return
        if spec.tool_id:
            self.toolActivated.emit(spec.tool_id, spec.label)
            self._close_submenu()

    def _activate_inner(self, index: int) -> None:
        if not (self._submenu_open and 0 <= index < len(self._submenu_tools)):
            return
        spec = self._submenu_tools[index]
        if not spec.is_leaf():
            self._submenu_tools = [child.clone() for child in spec.children]
            self._hover_sub_index = -1
            self._mark_dirty()
            return
        if spec.tool_id:
            self.toolActivated.emit(spec.tool_id, spec.label)
            self._close_submenu()

    def set_tools(self, tools: List[Tuple[str, str]] | List[ToolSpec]) -> None:
        """Replace the wheel contents with hierarchical specs or legacy tuples."""

        def coerce(item) -> ToolSpec:
            if isinstance(item, ToolSpec):
                return (
                    item.clone()
                    if hasattr(item, "clone")
                    else ToolSpec(item.label, item.tool_id, list(item.children))
                )
            if isinstance(item, tuple):
                label = item[0]
                tool_id = None
                children_data = None
                if len(item) > 1:
                    if isinstance(item[1], str):
                        tool_id = item[1]
                        if len(item) > 2:
                            children_data = item[2]
                    else:
                        children_data = item[1]
                spec_children = []
                if isinstance(children_data, (list, tuple)):
                    spec_children = [coerce(child) for child in children_data]
                return ToolSpec(label=label, tool_id=tool_id, children=spec_children)
            raise TypeError(f"Unsupported tool descriptor: {item!r}")

        self._close_submenu()
        if not tools:
            self._tools = []
            self._mark_dirty()
            return
        coerced = [coerce(item) for item in tools]
        self._tools = coerced
        self._mark_dirty()

    def set_idle_rotation(self, deg_per_sec: float) -> None:
        """Update the gentle ambient spin speed."""

        self._idle_rotate_speed = float(deg_per_sec)
        self._time_ref.restart()
        if abs(self._idle_rotate_speed) > 1e-6:
            self._ensure_timer()

    def _ensure_timer(self, interval: int = 16) -> None:
        if self._timer.isActive():
            if self._timer.interval() != interval:
                self._timer.setInterval(interval)
        else:
            self._time_ref.restart()
            self._timer.start(interval)

    def _mark_dirty(self) -> None:
        self._needs_repaint = True
        self._ensure_timer()

    def set_fit_to_viewport_edge(self, enabled: bool) -> None:
        self._fit_edge = bool(enabled)
        self._mark_dirty()

    def set_edge_overshoot(self, pixels: int) -> None:
        self._edge_overshoot_px = max(0, int(pixels))
        self._mark_dirty()

    def set_thickness_ratio(self, ratio: float) -> None:
        self._thickness_ratio = max(0.02, min(0.40, float(ratio)))
        self._mark_dirty()

    def set_outer_radius_ratio(self, ratio: float) -> None:
        self._outer_ratio = max(0.10, min(0.90, float(ratio)))
        self._mark_dirty()

    def set_opacity(self, value: float) -> None:
        self._opacity = max(0.0, min(1.0, float(value)))
        self._mark_dirty()

    def set_text_scale(self, scale: float) -> None:
        self._text_scale = max(0.5, min(2.5, float(scale)))
        self._mark_dirty()

    def set_min_alpha(self, value: float) -> None:
        self._min_alpha = max(0.0, min(1.0, float(value)))
        self._mark_dirty()

    # ------------ Internals ------------

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        # Keep overlay geometry matched to parent
        if obj is self.parent():
            if ev.type() in (QEvent.Resize, QEvent.Move):
                self._attach_to_parent()
            elif ev.type() in (
                QEvent.MouseMove,
                QEvent.HoverMove,
                QEvent.Enter,
                QEvent.Leave,
                QEvent.Wheel,
            ):
                self._mark_dirty()
        return super().eventFilter(obj, ev)

    def _attach_to_parent(self) -> None:
        if self.parent() is None:
            return
        par: QWidget = self.parent()  # type: ignore[assignment]
        self.setGeometry(par.rect())
        self.raise_()
        self.update()

    def showEvent(self, _) -> None:
        self._attach_to_parent()

    # Fade alpha based on cursor distance to centre (0 at centre -> 1 near edges)
    def _compute_alpha(self, cursor_pos: QPointF) -> float:
        w, h = self.width(), self.height()
        if w <= 1 or h <= 1:
            return 0.0
        cx, cy = w * 0.5, h * 0.5
        dx, dy = cursor_pos.x() - cx, cursor_pos.y() - cy
        dist = math.hypot(dx, dy)
        r = 0.5 * min(w, h)
        start = self._fade_center_ratio * r
        end = self._fade_edge_ratio * r
        if dist <= start:
            t = 0.0
        elif dist >= end:
            t = 1.0
        else:
            t = (dist - start) / max(1e-6, (end - start))
        t = t * t * (3 - 2 * t)  # smoothstep easing
        t = max(0.0, min(1.0, t))
        return max(self._min_alpha, t) if self._tools else t

    def _radii(self, w: int, h: int) -> tuple[float, float, float]:
        """Compute (outer_radius, thickness, inner_radius) in pixels.

        - If fit-to-edge is enabled, the outer radius is half of the minimum
          viewport dimension plus an overshoot in pixels (to extend just
          beyond the edge). Otherwise, the outer radius is based on the
          configured ratio.
        - Thickness is always a ratio of half the min dimension to keep
          proportions consistent across sizes.
        """
        half_min = 0.5 * float(min(w, h))
        if self._fit_edge:
            r_outer = half_min + float(max(0, self._edge_overshoot_px))
        else:
            r_outer = float(self._outer_ratio) * half_min
        thickness = float(self._thickness_ratio) * half_min
        r_inner = max(6.0, r_outer - thickness)
        return r_outer, thickness, r_inner

    def _on_tick(self) -> None:
        changed = False

        elapsed_ms = self._time_ref.restart()
        if elapsed_ms <= 0:
            elapsed_ms = float(self._timer.interval() or 16)

        if not self._dragging and abs(self._idle_rotate_speed) > 1e-6:
            prev_rot = self._rotation_deg
            self._rotation_deg = (
                self._rotation_deg + self._idle_rotate_speed * (elapsed_ms / 1000.0)
            ) % 360.0
            if abs(self._rotation_deg - prev_rot) > 1e-3:
                changed = True

        global_pos = QCursor.pos()
        local_pos = self.mapFromGlobal(global_pos)

        new_alpha = self._compute_alpha(local_pos)
        if abs(new_alpha - self._alpha) > 1e-3:
            self._alpha = new_alpha
            changed = True

        ring, idx = self._hit_test(local_pos)
        want_mouse = ring is not None
        if want_mouse != (not self.testAttribute(Qt.WA_TransparentForMouseEvents)):
            self.setAttribute(Qt.WA_TransparentForMouseEvents, not want_mouse)

        new_hover_outer = self._hover_index
        new_hover_inner = self._hover_sub_index

        if ring == "outer":
            new_hover_outer = idx
            new_hover_inner = -1
        elif ring == "inner":
            new_hover_outer = self._active_group
            new_hover_inner = idx
        else:
            new_hover_outer = -1 if not self._submenu_open else self._active_group
            new_hover_inner = -1

        if new_hover_outer != self._hover_index:
            self._hover_index = new_hover_outer
            changed = True
        if new_hover_inner != self._hover_sub_index:
            self._hover_sub_index = new_hover_inner
            changed = True

        if self._needs_repaint:
            changed = True
            self._needs_repaint = False

        if changed:
            self.update()

        active = (
            self._dragging
            or abs(self._idle_rotate_speed) > 1e-6
            or self._hover_index != -1
            or self._hover_sub_index != -1
            or self._alpha > 0.05
        )
        if active:
            self._ensure_timer()
        else:
            if self._timer.isActive():
                self._timer.stop()

    # ---- Drawing ----

    def paintEvent(self, ev: QPaintEvent) -> None:  # noqa: D401 - Qt signature
        if not self._tools:
            return

        global_alpha = max(0.0, min(1.0, self._alpha * self._opacity))
        if global_alpha <= 1e-3:
            return

        w, h = self.width(), self.height()
        if w <= 1 or h <= 1:
            return
        cx, cy = w * 0.5, h * 0.5
        radius_outer, thickness, radius_inner = self._radii(w, h)

        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            render_hints = getattr(QPainter, "RenderHint", None)
            for hint_name in ("HighQualityAntialiasing", "SmoothPixmapTransform"):
                hint = None
                if render_hints is not None and hasattr(render_hints, hint_name):
                    hint = getattr(render_hints, hint_name)
                elif hasattr(QPainter, hint_name):
                    hint = getattr(QPainter, hint_name)
                if hint is not None:
                    try:
                        p.setRenderHint(hint, True)
                    except Exception:
                        pass

            p.save()
            grad = QRadialGradient(QPointF(cx, cy), radius_outer)
            c0 = QColor(self._base_color)
            c1 = QColor(self._base_color)
            c0.setAlphaF(min(1.0, global_alpha * 0.55))
            c1.setAlphaF(0.0)
            grad.setColorAt(0.70, c0)
            grad.setColorAt(1.00, c1)
            outer_rect = QRectF(
                cx - radius_outer, cy - radius_outer, radius_outer * 2, radius_outer * 2
            )
            inner_rect = QRectF(
                cx - radius_inner, cy - radius_inner, radius_inner * 2, radius_inner * 2
            )
            ring_outer = QPainterPath()
            ring_outer.addEllipse(outer_rect)
            ring_inner = QPainterPath()
            ring_inner.addEllipse(inner_rect)
            ring_path = ring_outer.subtracted(ring_inner)
            p.setBrush(QBrush(grad))
            p.setPen(Qt.NoPen)
            p.drawPath(ring_path)
            p.restore()

            seg_count = len(self._tools)
            if seg_count == 0:
                return
            seg_span = 360.0 / max(1.0, float(seg_count))

            p.save()
            p.translate(cx, cy)
            p.rotate(self._rotation_deg)
            p.translate(-cx, -cy)
            clip_outer = QPainterPath()
            clip_outer.addEllipse(outer_rect)
            clip_inner = QPainterPath()
            clip_inner.addEllipse(inner_rect)
            p.setClipPath(clip_outer.subtracted(clip_inner))

            hover_outer = self._hover_index
            hover_inner = self._hover_sub_index
            active_group = self._active_group if self._submenu_open else -1
            dpr = max(1.0, self.devicePixelRatioF())
            base_font_size = max(6.0, float(self._font_pt) * self._text_scale)
            for i, spec in enumerate(self._tools):
                start_deg = i * seg_span

                r_in = radius_inner
                r_out = radius_outer

                path_seg = self._arc_segment_path(QPointF(cx, cy), r_in, r_out, start_deg, seg_span)
                fill_color = QColor(self._base_color)
                if hover_outer == i:
                    fill_alpha = 0.78
                elif active_group == i:
                    fill_alpha = 0.60
                else:
                    fill_alpha = 0.38
                fill_color.setAlphaF(min(1.0, global_alpha * fill_alpha))
                p.fillPath(path_seg, fill_color)

                line_color = QColor(self._line_color)
                if hover_outer == i:
                    line_alpha = 0.95
                elif active_group == i:
                    line_alpha = 0.75
                else:
                    line_alpha = 0.60 if hover_outer >= 0 else 0.70
                line_color.setAlphaF(min(1.0, global_alpha * line_alpha))
                pen = QPen(line_color, 1.25)
                pen.setCosmetic(True)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawPath(path_seg)

                if hover_outer == i:
                    glow = QColor(self._hover_glow)
                    glow.setAlphaF(min(1.0, global_alpha * 0.75))
                    p.fillPath(path_seg, glow)

                label_angle = math.radians(start_deg + seg_span * 0.5)
                rr = 0.5 * (r_in + r_out)
                lx = cx + rr * math.cos(label_angle)
                ly = cy + rr * math.sin(label_angle)

                p.save()
                p.translate(lx, ly)
                p.rotate(-self._rotation_deg)

                f = p.font()
                f.setPointSizeF(base_font_size * dpr)
                p.setFont(f)
                metrics = p.fontMetrics()
                metrics_w = metrics.horizontalAdvance(spec.label)
                metrics_h = metrics.height()
                baseline = metrics_h * 0.35

                label_color = QColor(self._text_color)
                label_color.setAlphaF(min(1.0, label_color.alphaF() * global_alpha))
                if hover_outer >= 0 and i != hover_outer:
                    label_color.setAlphaF(label_color.alphaF() * 0.55)
                elif active_group == i and hover_inner == -1:
                    label_color.setAlphaF(min(1.0, max(0.35, label_color.alphaF() * 1.05)))

                shadow = QColor(0, 0, 0)
                shadow.setAlphaF(label_color.alphaF() * 0.6)
                p.setPen(shadow)
                p.drawText(QPointF(-metrics_w / 2 + 1.0, baseline + 1.0), spec.label)
                p.setPen(QPen(label_color))
                p.drawText(QPointF(-metrics_w / 2, baseline), spec.label)

                p.restore()

            p.restore()

            if self._submenu_open and self._submenu_tools:
                sub_outer, sub_inner = self._submenu_radii(radius_inner, thickness)
                if sub_outer - sub_inner > 2.0 and len(self._submenu_tools) > 0:
                    sub_outer_rect = QRectF(
                        cx - sub_outer, cy - sub_outer, sub_outer * 2, sub_outer * 2
                    )
                    sub_inner_rect = QRectF(
                        cx - sub_inner, cy - sub_inner, sub_inner * 2, sub_inner * 2
                    )
                    sub_span = 360.0 / max(1.0, float(len(self._submenu_tools)))

                    p.save()
                    p.translate(cx, cy)
                    p.rotate(self._rotation_deg)
                    p.translate(-cx, -cy)
                    sub_clip_outer = QPainterPath()
                    sub_clip_outer.addEllipse(sub_outer_rect)
                    sub_clip_inner = QPainterPath()
                    sub_clip_inner.addEllipse(sub_inner_rect)
                    p.setClipPath(sub_clip_outer.subtracted(sub_clip_inner))

                    hover_inner = self._hover_sub_index
                    dpr = max(1.0, self.devicePixelRatioF())
                    for j, spec in enumerate(self._submenu_tools):
                        start_deg = j * sub_span
                        path_seg = self._arc_segment_path(
                            QPointF(cx, cy), sub_inner, sub_outer, start_deg, sub_span
                        )

                        fill_color = QColor(self._base_color)
                        fill_alpha = 0.78 if hover_inner == j else 0.55
                        fill_color.setAlphaF(min(1.0, global_alpha * fill_alpha))
                        p.fillPath(path_seg, fill_color)

                        line_color = QColor(self._line_color)
                        line_alpha = 0.95 if hover_inner == j else 0.65
                        line_color.setAlphaF(min(1.0, global_alpha * line_alpha))
                        pen = QPen(line_color, 1.0)
                        pen.setCosmetic(True)
                        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                        p.setPen(pen)
                        p.drawPath(path_seg)

                        if hover_inner == j:
                            glow = QColor(self._hover_glow)
                            glow.setAlphaF(min(1.0, global_alpha * 0.8))
                            p.fillPath(path_seg, glow)

                        label_angle = math.radians(start_deg + sub_span * 0.5)
                        rr = 0.5 * (sub_inner + sub_outer)
                        lx = cx + rr * math.cos(label_angle)
                        ly = cy + rr * math.sin(label_angle)

                        p.save()
                        p.translate(lx, ly)
                        p.rotate(-self._rotation_deg)
                        f = p.font()
                        f.setPointSizeF(base_font_size * 0.9 * dpr)
                        p.setFont(f)
                        metrics = p.fontMetrics()
                        metrics_w = metrics.horizontalAdvance(spec.label)
                        metrics_h = metrics.height()
                        baseline = metrics_h * 0.35

                        label_color = QColor(self._text_color)
                        label_color.setAlphaF(min(1.0, label_color.alphaF() * global_alpha))
                        if hover_inner >= 0 and j != hover_inner:
                            label_color.setAlphaF(label_color.alphaF() * 0.6)

                        shadow = QColor(0, 0, 0)
                        shadow.setAlphaF(label_color.alphaF() * 0.6)
                        p.setPen(shadow)
                        p.drawText(QPointF(-metrics_w / 2 + 1.0, baseline + 1.0), spec.label)
                        p.setPen(QPen(label_color))
                        p.drawText(QPointF(-metrics_w / 2, baseline), spec.label)
                        p.restore()

                    p.restore()
        finally:
            p.end()

    def _arc_segment_path(
        self, c: QPointF, r_inner: float, r_outer: float, start_deg: float, span_deg: float
    ) -> QPainterPath:
        rect_o = QRectF(c.x() - r_outer, c.y() - r_outer, r_outer * 2, r_outer * 2)
        rect_i = QRectF(c.x() - r_inner, c.y() - r_inner, r_inner * 2, r_inner * 2)

        path = QPainterPath()
        sx = c.x() + r_outer * math.cos(math.radians(start_deg))
        sy = c.y() + r_outer * math.sin(math.radians(start_deg))
        path.moveTo(QPointF(sx, sy))
        path.arcTo(rect_o, start_deg, span_deg)
        end_deg = start_deg + span_deg
        ex = c.x() + r_inner * math.cos(math.radians(end_deg))
        ey = c.y() + r_inner * math.sin(math.radians(end_deg))
        path.lineTo(QPointF(ex, ey))
        path.arcTo(rect_i, end_deg, -span_deg)
        path.closeSubpath()
        return path

    def _submenu_radii(self, radius_inner: float, thickness: float) -> Tuple[float, float]:
        gap = max(6.0, thickness * 0.45)
        outer = max(6.0, radius_inner - gap)
        sub_thickness = max(6.0, thickness * 0.85)
        inner = max(6.0, outer - sub_thickness)
        if inner >= outer:
            inner = max(6.0, outer * 0.65)
        return outer, inner

    def _hit_test(self, pos: QPointF) -> Tuple[Optional[str], int]:
        if not self._tools:
            return (None, -1)
        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            return (None, -1)
        cx, cy = w * 0.5, h * 0.5
        dx, dy = pos.x() - cx, pos.y() - cy
        dist = math.hypot(dx, dy)
        r_outer, thickness, r_inner = self._radii(w, h)

        if self._submenu_open and self._submenu_tools:
            sub_outer, sub_inner = self._submenu_radii(r_inner, thickness)
            if sub_inner <= dist <= sub_outer:
                ang = math.degrees(math.atan2(dy, dx))
                ang = (ang + 360.0) % 360.0
                a = (ang - self._rotation_deg) % 360.0
                seg = int((a / 360.0) * len(self._submenu_tools))
                seg = max(0, min(len(self._submenu_tools) - 1, seg))
                return ("inner", seg)

        if dist < r_inner or dist > r_outer:
            return (None, -1)

        ang = math.degrees(math.atan2(dy, dx))
        ang = (ang + 360.0) % 360.0
        a = (ang - self._rotation_deg) % 360.0
        seg = int((a / 360.0) * len(self._tools))
        seg = max(0, min(len(self._tools) - 1, seg))
        return ("outer", seg)

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: D401
        self._ensure_timer()
        ring, idx = self._hit_test(ev.position())

        if ev.button() == Qt.RightButton:
            self._dragging = True
            self._drag_last_angle = self._angle_at(ev.position())
            if ring is None and self._submenu_open:
                self._close_submenu()
            ev.accept()
            return

        if ev.button() == Qt.LeftButton:
            self._dragging = False
            if ring is None:
                if self._submenu_open:
                    self._close_submenu()
                    ev.accept()
                else:
                    ev.ignore()
                self._press_ring = None
                self._press_index = -1
                return
            self._press_ring = ring
            self._press_index = idx
            ev.accept()
            return

        ev.ignore()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # noqa: D401
        self._ensure_timer()
        if self._dragging:
            a_now = self._angle_at(ev.position())
            if self._drag_last_angle is not None and a_now is not None:
                da = a_now - self._drag_last_angle
                self._rotation_deg = (self._rotation_deg + da) % 360.0
                self._drag_last_angle = a_now
                self._mark_dirty()
            ev.accept()
            return
        ev.ignore()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:  # noqa: D401
        self._ensure_timer()
        ring, idx = self._hit_test(ev.position())

        if ev.button() == Qt.RightButton:
            self._dragging = False
            self._drag_last_angle = None
            ev.accept()
            return

        if ev.button() == Qt.LeftButton:
            pressed_ring, pressed_idx = self._press_ring, self._press_index
            self._press_ring = None
            self._press_index = -1
            if pressed_ring is None:
                ev.ignore()
                return
            if pressed_ring == "outer":
                if ring == "inner" and self._submenu_open and idx >= 0:
                    self._activate_inner(idx)
                    ev.accept()
                    return
                if ring == "outer" and idx == pressed_idx:
                    self._activate_outer(idx)
                    ev.accept()
                    return
                ev.ignore()
                return
            if pressed_ring == "inner" and ring == "inner" and idx == pressed_idx:
                self._activate_inner(idx)
                ev.accept()
                return
            ev.ignore()
            return

        ev.ignore()

    def wheelEvent(self, ev: QWheelEvent) -> None:  # noqa: D401
        self._ensure_timer()
        deg = ev.angleDelta().y() / 8.0
        self._rotation_deg = (self._rotation_deg + deg) % 360.0
        self._mark_dirty()
        ev.accept()

    def _angle_at(self, pos: QPointF) -> Optional[float]:
        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            return None
        cx, cy = w * 0.5, h * 0.5
        dx, dy = pos.x() - cx, pos.y() - cy
        ang = math.degrees(math.atan2(dy, dx))
        ang = (ang + 360.0) % 360.0
        return ang


class RadialToolWheelOverlay(RadialToolWheel):
    """Convenience subclass pinned to a given viewport widget."""

    def __init__(self, viewport_widget: QWidget, tools: Optional[List[ToolSpec]] = None, **kwargs):
        super().__init__(viewport_widget, tools=tools, **kwargs)
        self._attach_to_parent()
