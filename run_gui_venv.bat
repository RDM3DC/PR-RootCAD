@echo off
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%\.venv\Scripts\python.exe"
if exist "%PY%" (
    "%PY%" -m adaptivecad.gui.playground %*
) else (
    python -m adaptivecad.gui.playground %*
)
endlocal
