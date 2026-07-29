@echo off
if "%~1"=="" (
    echo Usage: push.bat "your commit message"
    pause
    exit /b 1
)
echo Staging all changes...
git add -A
if errorlevel 1 goto :error
echo Committing...
git commit -m "%~1"
echo Pushing to GitHub...
git push origin main
if errorlevel 1 goto :error
echo Push complete.
pause
exit /b 0
:error
echo Push failed. See error above.
pause
exit /b 1