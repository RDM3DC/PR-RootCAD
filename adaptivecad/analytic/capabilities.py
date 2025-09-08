from importlib.util import find_spec

__all__ = [
    "has",
    "analytic_available",
    "exporter_available",
    "conversion_available",
    "summary",
]

def has(module: str) -> bool:
    """Return True if a module spec can be found without importing it."""
    try:
        return find_spec(module) is not None
    except Exception:
        return False

def analytic_available() -> bool:
    """Core analytic viewport capability (scene + viewport + OpenGL)."""
    return (
        has("adaptivecad.aacore.sdf") and
        (has("adaptivecad.gui.analytic_viewport") or has("adaptivecad.analytic.analytic_viewport")) and
        has("OpenGL")
    )

def exporter_available() -> bool:
    """Marching exporter + required third-party libs."""
    return (
        has("adaptivecad.aacore.extract.marching_export") and
        has("skimage") and
        has("stl")
    )

def conversion_available() -> bool:
    """Analytic conversion tooling availability."""
    return has("adaptivecad.analytic.conversion")

def summary() -> dict:
    return {
        "analytic": analytic_available(),
        "exporter": exporter_available(),
        "conversion": conversion_available(),
    }
