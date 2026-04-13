@echo off
:: Standalone ECS benchmark build+run.
:: Bypasses CMake's Debug /RTC1 flag (incompatible with /O2) by calling cl.exe directly.

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 > nul 2>&1

set ROOT=C:\Users\HTRMC\Dev\Projects\Enigma
set OUT=C:\Temp\ecs_bench

:: Resolve GLM include path from the CMake _deps directory
set GLM_INC=%ROOT%\cmake-build-debug\_deps\glm-src

if not exist "C:\Temp\ecs_bench_obj\" mkdir "C:\Temp\ecs_bench_obj\"
cl.exe ^
    /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127 ^
    /O2 /Oi /arch:AVX2 /fp:fast /MD /DNDEBUG ^
    /EHsc /nologo ^
    /I"%ROOT%\src" /I"%GLM_INC%" ^
    "%ROOT%\tests\ecs_bench.cpp" "%ROOT%\src\ecs\World.cpp" ^
    /Fe"%OUT%.exe" /Fo"C:\Temp\ecs_bench_obj\\" ^
    > C:\Temp\bench_cl.txt 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED -- see C:\Temp\bench_cl.txt
    type C:\Temp\bench_cl.txt
    exit /b 1
)

echo Build OK. Running benchmark...
"%OUT%.exe"
