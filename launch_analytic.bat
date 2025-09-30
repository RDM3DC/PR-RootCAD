@echo off
echo AdaptiveCAD Analytic Viewport Launcher
echo =======================================
cd /d "C:\Users\RDM3D\ADaptCAD\AdaptiveCAD"
echo Current directory: %CD%
echo Activating virtual environment...
call "C:\Users\RDM3D\ADaptCAD\.venv\Scripts\activate.bat"
echo Launching analytic viewport...
python analytic_viewport_launcher.py
echo.
echo Press any key to close...
pause