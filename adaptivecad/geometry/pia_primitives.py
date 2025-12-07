from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def pi_circle_points(
    radius: float, cx: float = 0.0, cy: float = 0.0, segments: int = 256
) -> List[Tuple[float, float]]:
    """Sample a circle into evenly spaced points (carrying pi_a semantics elsewhere)."""
    pts: List[Tuple[float, float]] = []
    for i in range(segments):
        t = (2.0 * math.pi) * (i / segments)
        x = cx + radius * math.cos(t)
        y = cy + radius * math.sin(t)
        pts.append((x, y))
    return pts


def make_pi_circle_profile(
    radius: float,
    cx: float = 0.0,
    cy: float = 0.0,
    segments: int = 256,
) -> Dict[str, Any]:
    """Build a profile dict flagged as pi_a superellipse representing a circle."""
    return {
        "type": "profile",
        "family": "pi_a:superellipse",
        "params": {"a": radius, "b": radius, "n": 2.0, "cx": cx, "cy": cy},
        "metric": "pi_a",
        "closed": True,
        "points": pi_circle_points(radius, cx, cy, segments),
    }


def upgrade_profile_meta_to_pia(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *profile* with pi_a metric metadata applied (idempotent)."""
    upgraded = dict(profile)
    upgraded["metric"] = "pi_a"
    upgraded.setdefault("family", "pi_a:generic")
    return upgraded
