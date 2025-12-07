from __future__ import annotations

from enum import Enum


class Units(str, Enum):
    MM = "mm"
    IN = "in"


# Internal unit is millimeters.
_MM_PER_IN = 25.4


def to_internal(value: float, units: Units) -> float:
    """Convert a value from given units to internal (mm)."""
    if units == Units.MM:
        return float(value)
    if units == Units.IN:
        return float(value) * _MM_PER_IN
    raise ValueError(f"Unsupported units: {units}")


def from_internal(value_mm: float, units: Units) -> float:
    """Convert a value from internal (mm) to given units."""
    if units == Units.MM:
        return float(value_mm)
    if units == Units.IN:
        return float(value_mm) / _MM_PER_IN
    raise ValueError(f"Unsupported units: {units}")
