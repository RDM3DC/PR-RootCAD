"""Phase‑Resolved (PR) modeling primitives.

This package provides a minimal, headless-safe substrate for evolving phase fields
and collecting metrics. It is designed to be callable from the MCP bridge.
"""

from .types import PRFieldConfig, PRFieldState
from .solver import relax_phase_field
from .derived import compute_derived_fields
from .tools import augment_ama_with_derived_fields
from .export import (
    export_phase_field_as_heightmap_stl,
    export_phase_field_as_obj,
    export_phase_field_as_ama,
)

__all__ = [
    "PRFieldConfig",
    "PRFieldState",
    "relax_phase_field",
    "compute_derived_fields",
    "augment_ama_with_derived_fields",
    "export_phase_field_as_heightmap_stl",
    "export_phase_field_as_obj",
    "export_phase_field_as_ama",
]
