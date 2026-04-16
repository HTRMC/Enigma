@echo off
setlocal

set SOURCE=C:\Users\HTRMC\Dev\Projects\Enigma
set BUILD=C:\Users\HTRMC\Dev\Projects\Enigma\build_release
set NINJA=C:\Users\HTRMC\AppData\Local\Programs\CLion\bin\ninja\win\x64\ninja.exe

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

if not exist "%BUILD%\CMakeCache.txt" (
    echo [configure] first-time RelWithDebInfo configure...
    cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_MAKE_PROGRAM="%NINJA%" -S "%SOURCE%" -B "%BUILD%"
    if %ERRORLEVEL% neq 0 ( echo ERROR: cmake configure failed. & exit /b 1 )
)

cd /d "%BUILD%"
"%NINJA%" Enigma
echo EXIT_CODE=%ERRORLEVEL%
