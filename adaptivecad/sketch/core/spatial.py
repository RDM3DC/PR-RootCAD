"""Axis-aligned bounding boxes and a lightweight spatial index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .numeric import get_tolerance

VecLike = Tuple[float, float]


@dataclass(frozen=True)
class AABB2:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @staticmethod
    def from_points(points: Iterable[VecLike]) -> "AABB2":
        iterator = iter(points)
        try:
            first = next(iterator)
        except StopIteration:
            tol = get_tolerance().linear
            return AABB2(0.0, 0.0, tol, tol)
        min_x = max_x = float(first[0])
        min_y = max_y = float(first[1])
        for x, y in iterator:
            x = float(x)
            y = float(y)
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y
        return AABB2(min_x, min_y, max_x, max_y)

    def expanded(self, margin: float) -> "AABB2":
        return AABB2(
            self.xmin - margin,
            self.ymin - margin,
            self.xmax + margin,
            self.ymax + margin,
        )

    def intersects(self, other: "AABB2") -> bool:
        return not (
            self.xmax < other.xmin
            or self.xmin > other.xmax
            or self.ymax < other.ymin
            or self.ymin > other.ymax
        )

    def contains_point(self, point: VecLike) -> bool:
        x, y = point
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def union(self, other: "AABB2") -> "AABB2":
        return AABB2(
            min(self.xmin, other.xmin),
            min(self.ymin, other.ymin),
            max(self.xmax, other.xmax),
            max(self.ymax, other.ymax),
        )


class SpatialIndex:
    """A minimal spatial index with an interface compatible with quadtrees.

    The current implementation keeps entities in a flat dictionary while we
    flesh out the geometry kernel.  The API mirrors what a real quadtree or
    R-tree will expose so the internals can be swapped without ripple effects.
    """

    def __init__(self) -> None:
        self._items: Dict[str, Tuple[AABB2, Optional[object]]] = {}

    def insert(self, entity_id: str, bounds: AABB2, payload: Optional[object] = None) -> None:
        self._items[entity_id] = (bounds, payload)

    def remove(self, entity_id: str) -> None:
        self._items.pop(entity_id, None)

    def update(self, entity_id: str, bounds: AABB2, payload: Optional[object] = None) -> None:
        self.insert(entity_id, bounds, payload)

    def query(self, region: AABB2, *, include_payload: bool = False) -> Iterator[Tuple[str, Optional[object]]]:
        for entity_id, (bounds, payload) in self._items.items():
            if bounds.intersects(region):
                yield (entity_id, payload if include_payload else None)

    def all_items(self) -> List[Tuple[str, AABB2, Optional[object]]]:
        return [(entity_id, bounds, payload) for entity_id, (bounds, payload) in self._items.items()]


__all__ = ["AABB2", "SpatialIndex", "VecLike"]
