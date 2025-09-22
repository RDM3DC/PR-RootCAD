"""Entity interfaces and shared metadata for the sketch subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional, Protocol, Sequence

from .numeric import get_tolerance
from .spatial import AABB2, VecLike


class EntityKind(str, Enum):
    POINT = "point"
    CURVE = "curve"
    REGION = "region"
    ANNOTATION = "annotation"
    AUXILIARY = "auxiliary"


@dataclass
class EntityStyle:
    color: Optional[Sequence[float]] = None  # RGB in 0..1
    linetype: str = "continuous"
    lineweight: float = 0.25  # millimeters
    fill: Optional[Sequence[float]] = None


@dataclass
class EntityMetadata:
    name: Optional[str] = None
    tags: set[str] = field(default_factory=set)
    custom: Dict[str, object] = field(default_factory=dict)


@dataclass
class SketchEntityBase:
    id: str
    kind: EntityKind
    layer: str = "0"
    style: EntityStyle = field(default_factory=EntityStyle)
    construction: bool = False
    metadata: EntityMetadata = field(default_factory=EntityMetadata)

    def bounds(self) -> AABB2:
        raise NotImplementedError

    def snap_points(self) -> Iterable[VecLike]:
        return ()


class ParamCurve2D(Protocol):
    def evaluate(self, t: float) -> VecLike: ...
    def tangent(self, t: float) -> VecLike: ...
    def normal(self, t: float) -> VecLike: ...
    def parameter_range(self) -> tuple[float, float]: ...


@dataclass
class DimensionAnchor:
    entity_id: Optional[str] = None
    point: Optional[VecLike] = None
    param: Optional[float] = None


@dataclass
class DimensionBase2D(SketchEntityBase):
    kind: str = "dimension"
    anchors: tuple[DimensionAnchor, DimensionAnchor] = field(
        default_factory=lambda: (DimensionAnchor(), DimensionAnchor())
    )
    text: str = ""
    precision: int = 3

    def tolerance_band(self) -> float:
        return get_tolerance().linear


__all__ = [
    "EntityKind",
    "EntityStyle",
    "EntityMetadata",
    "SketchEntityBase",
    "ParamCurve2D",
    "DimensionAnchor",
    "DimensionBase2D",
]
