@echo off
REM Native Windows CMD launcher; forwards arguments to PowerShell.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
