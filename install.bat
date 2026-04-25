@echo off
echo Installing Murmur dependencies...
"C:\Users\USER\AppData\Local\Programs\Python\Python311\python.exe" -m pip install -r "%~dp0.\requirements.txt"
echo.
echo Done. Launch with start_murmur.bat
pause
