@echo off
echo ========================================
echo    Billiards AI Assistant - EXE Creator
echo ========================================
echo.
echo Installing required packages...
pip install streamlit pillow numpy pyinstaller
echo.
echo Creating executable file...
python create_exe.py
echo.
echo Done! Check the 'dist' folder for BilliardsAI.exe
echo.
pause
