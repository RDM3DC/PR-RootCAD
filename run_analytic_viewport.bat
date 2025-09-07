@echo off
echo Starting AdaptiveCAD Analytic Viewport...
echo.
call conda activate adaptivegl
python test_analytic_viewport.py
pause