@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 2>nul
"C:\Users\HTRMC\AppData\Local\Programs\CLion\bin\cmake\win\x64\bin\cmake.exe" ^
    --build "C:\Users\HTRMC\Dev\Projects\Enigma\cmake-build-debug" ^
    --target ecs_compile_test ^
    2>&1 > "C:\Users\HTRMC\Dev\Projects\Enigma\tests\ecs_build_result.txt"
echo BUILD_EXIT:%ERRORLEVEL% >> "C:\Users\HTRMC\Dev\Projects\Enigma\tests\ecs_build_result.txt"
