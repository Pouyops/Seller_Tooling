@echo off
REM Windows stand-in for GNU make: `make.cmd bench` == `make bench`.
REM Uses the project venv if present, else python on PATH.
setlocal
if exist "%~dp0.env.cmd" call "%~dp0.env.cmd"
if defined ST_VENV (
  set "PYEXE=%ST_VENV%\Scripts\python.exe"
) else if exist "D:\seller-tooling\venv\Scripts\python.exe" (
  set "PYEXE=D:\seller-tooling\venv\Scripts\python.exe"
) else (
  set "PYEXE=python"
)
"%PYEXE%" "%~dp0tools\tasks.py" %*
exit /b %ERRORLEVEL%
