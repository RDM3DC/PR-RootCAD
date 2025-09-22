# Contributing to AdaptiveCAD

Thanks for your interest in contributing! This project is a Windows‑first Python app with a PySide6 GUI and an analytic SDF renderer. Small, focused changes are best. This document explains how to set up, code, test, and submit PRs that pass CI.

## Quick start
- Windows PowerShell (preferred) with Python 3.12 in a venv or conda env.
- Clone the repo and install dependencies in editable mode:
  - Create/activate your env, then run: `pip install -U pip setuptools wheel`
  - Install project deps: `pip install -e .`
  - Install dev tools: `pip install pytest ruff black isort mypy pre-commit`
- Install pre-commit hooks (required for commits): `pre-commit install`

## Run the app
- VS Code tasks
  - Start Playground (venv): Task “Start Playground (venv)”
  - Analytic Viewport panel only: Task “Launch Analytic Viewport”
- CLI (when needed)
  - `python -m adaptivecad.gui.playground` (see `.github/copilot-instructions.md` for more notes)

## Tests (fast subset)
- Run just the fast unit tests locally:
  - `pytest -q adaptivecad/aacore/tests`
  - Optional GUI unit test (skips if PySide6 not installed):
    - `pytest -q adaptivecad/ui/tests/test_radial_tool_wheel.py`
- Full suite has many optional/deep tests. Keep CI fast by targeting core math and safe GUI tests.

## Formatting, linting, typing
- Format: `black .` (line length 100)
- Imports: `isort .` (profile black)
- Lint: `ruff check .` (quick, strict-ish; see config in pyproject)
- Types: `mypy adaptivecad` (lenient; ignores missing 3rd‑party stubs)
- All of the above run in CI in check mode; pre-commit will run a subset before each commit.

## PR guidelines
- Keep PRs small and focused. Include a brief summary of what changed and why.
- Add or update tests for new behavior. Prefer fast, deterministic tests.
- Ensure:
  - `ruff check .` passes
  - `black --check .` passes
  - `isort --check-only .` passes
  - `mypy adaptivecad` passes
  - `pytest -q adaptivecad/aacore/tests adaptivecad/ui/tests/test_radial_tool_wheel.py` passes
- Avoid large, cross‑cutting refactors. If you need one, discuss first.

## Project conventions
- GUI: PySide6 only. Analytic view uses QOpenGLWidget. Wheel HUD lives in `adaptivecad/ui/radial_tool_wheel.py` and is owned by `adaptivecad/gui/analytic_viewport.py`.
- SDF math: numpy float32; transforms are 4×4 affine; shader packing is column‑major.
- Persistence: analytic panel/user prefs merge into `~/.adaptivecad_analytic.json` (do not clobber unknown keys).
- Tests: Prefer unit tests near modules (see `adaptivecad/**/tests/`).

## Troubleshooting
- Windows timer quirks: The hyperbolic helpers manage resolution locally (no global monkeypatches).
- Qt issues: Try `set QT_QPA_PLATFORM=offscreen` for headless runs; verify PySide6 install.
- OCC optional: Keep UI responsive when OCC is missing. Tests must not depend on OCC in CI.

## Communication
- CODEOWNERS gates reviews. Mention maintainers for design changes.
- See `AGENTS.md` for the long‑term 2D CAD roadmap and `ENVIRONMENT.md` for environment tips.

---
Thanks again for contributing!