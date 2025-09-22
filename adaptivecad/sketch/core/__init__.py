"""Core primitives shared across the 2D sketch kernel."""

from .numeric import (
    TolerancePolicy,
    get_tolerance,
    set_tolerance,
    on_tolerance_changed,
    nearly_equal,
    stable_average,
)
from .spatial import AABB2, SpatialIndex, VecLike
from .entities import (
    EntityKind,
    EntityStyle,
    EntityMetadata,
    SketchEntityBase,
    ParamCurve2D,
    DimensionAnchor,
    DimensionBase2D,
)
from .history import Command, CommandStack
from .solver import SolverVariable, SolverConstraint, ConstraintSolver

__all__ = [
    "TolerancePolicy",
    "get_tolerance",
    "set_tolerance",
    "on_tolerance_changed",
    "nearly_equal",
    "stable_average",
    "AABB2",
    "SpatialIndex",
    "VecLike",
    "EntityKind",
    "EntityStyle",
    "EntityMetadata",
    "SketchEntityBase",
    "ParamCurve2D",
    "DimensionAnchor",
    "DimensionBase2D",
    "Command",
    "CommandStack",
    "SolverVariable",
    "SolverConstraint",
    "ConstraintSolver",
]
