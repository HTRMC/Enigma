@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d C:\Users\HTRMC\Dev\Projects\Enigma\build
"C:\Users\HTRMC\AppData\Local\Programs\CLion\bin\ninja\win\x64\ninja.exe" Enigma
echo EXIT_CODE=%ERRORLEVEL%
