@echo off
echo Fetching latest from GitHub...
git fetch origin
if errorlevel 1 goto :error
echo Overwriting local files with GitHub's main branch...
git reset --hard origin/main
if errorlevel 1 goto :error
echo Pull complete. Local files now match GitHub exactly.
pause
exit /b 0
:error
echo Pull failed. See error above.
pause
exit /b 1