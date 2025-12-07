"""Core primitives shared across the 2D sketch kernel."""

from .entities import (
    DimensionAnchor,
    DimensionBase2D,
    EntityKind,
    EntityMetadata,
    EntityStyle,
    ParamCurve2D,
    SketchEntityBase,
)
from .history import Command, CommandStack
from .numeric import (
    TolerancePolicy,
    get_tolerance,
    nearly_equal,
    on_tolerance_changed,
    set_tolerance,
    stable_average,
)
from .solver import ConstraintSolver, SolverConstraint, SolverVariable
from .spatial import AABB2, SpatialIndex, VecLike

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
