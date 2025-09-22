"""Numeric primitives and tolerance configuration for the 2D sketch kernel."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable, Iterable, Optional


@dataclass(frozen=True)
class TolerancePolicy:
    """Container for numeric tolerances used throughout the sketch kernel."""

    linear: float = 1e-9
    angular: float = 1e-6  # radians
    parametric: float = 1e-9
    max_iterations: int = 64

    def clamp_linear(self, value: float) -> float:
        return max(-self.linear, min(self.linear, value))


_policy_lock = RLock()
_current_policy: TolerancePolicy = TolerancePolicy()
_listeners: list[Callable[[TolerancePolicy], None]] = []


def get_tolerance() -> TolerancePolicy:
    with _policy_lock:
        return _current_policy


def set_tolerance(policy: TolerancePolicy) -> None:
    """Update the global tolerance policy and notify listeners."""

    with _policy_lock:
        global _current_policy
        _current_policy = policy
        listeners = list(_listeners)
    for cb in listeners:
        cb(policy)


def on_tolerance_changed(listener: Callable[[TolerancePolicy], None]) -> None:
    with _policy_lock:
        _listeners.append(listener)


def nearly_equal(a: float, b: float, *, eps: Optional[float] = None) -> bool:
    tol = eps if eps is not None else get_tolerance().linear
    return abs(a - b) <= tol


def stable_average(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for v in values:
        total += v
        count += 1
    if count == 0:
        raise ValueError("stable_average requires at least one value")
    return total / count


