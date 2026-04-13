@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 2>nul
cl.exe /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /EHsc /nologo /c "C:\Users\HTRMC\Dev\Projects\Enigma\tests\p2996_probe.cpp" /Fo"C:\Users\HTRMC\Dev\Projects\Enigma\cmake-build-debug\p2996_probe.obj" > "C:\Users\HTRMC\Dev\Projects\Enigma\tests\probe_result.txt" 2>&1
echo %ERRORLEVEL% > "C:\Users\HTRMC\Dev\Projects\Enigma\tests\probe_exit.txt"
