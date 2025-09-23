"""AI helpers for AdaptiveCAD.

This package exposes lightweight chat and tool-calling utilities by default.
Heavier optional pieces (like the SymPy-based translator) are imported lazily
so that the GUI can start even when those dependencies aren't installed.
"""

from .openai_bridge import call_openai
from .intent_router import CADActionBus, chat_with_tools, build_tool_schema
from .openai_client import get_client

# Try to expose translator helpers if their optional dependency (sympy) exists.
# If unavailable, we simply omit those names from the public API to keep startup fast.
try:  # optional dependency guard
    from .translator import build_geometry, ImplicitSurface, ExtrudedSolid  # type: ignore
except Exception:  # pragma: no cover - environment may lack sympy
    build_geometry = None  # type: ignore
    ImplicitSurface = None  # type: ignore
    ExtrudedSolid = None  # type: ignore
    _HAS_TRANSLATOR = False
else:
    _HAS_TRANSLATOR = True

__all__ = [
    "call_openai",
    "CADActionBus",
    "chat_with_tools",
    "build_tool_schema",
    "get_client",
]

if _HAS_TRANSLATOR:
    __all__.extend(["build_geometry", "ImplicitSurface", "ExtrudedSolid"])
