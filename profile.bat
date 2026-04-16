@echo off
setlocal

set EXE=%~dp0build_release\Enigma.exe
set FRAMES=120

if not exist "%EXE%" (
    echo ERROR: %EXE% not found. Run build_release.bat first.
    exit /b 1
)

echo.
echo ============================================================
echo  Enigma GPU Profiler  [RelWithDebInfo]
echo  Capturing %FRAMES% frames  ^(30 warmup^)
echo  Move around in the scene for representative coverage.
echo ============================================================
echo.

"%EXE%" --profile %FRAMES%

if exist "%~dp0build_release\enigma_profile.csv" (
    echo.
    echo Opening CSV...
    start "" "%~dp0build_release\enigma_profile.csv"
)

endlocal
