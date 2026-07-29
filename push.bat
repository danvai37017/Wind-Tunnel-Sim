@echo off
if "%~1"=="" (
    echo Usage: push.bat "commit message"
    exit /b 1
)

git add .
git commit -m "%~1"
git push origin main
