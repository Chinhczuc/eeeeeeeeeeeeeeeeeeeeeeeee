@echo off
echo ========================================
echo    Billiards AI - Standalone EXE Builder
echo ========================================
echo.
echo Building standalone executable...
echo.

REM Install required packages
echo Installing PyInstaller...
pip install pyinstaller

echo.
echo Creating executable from standalone_app.py...
pyinstaller --onefile --windowed --name=BilliardsAI --icon=generated-icon.png standalone_app.py

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Your BilliardsAI.exe is ready in the 'dist' folder
echo.
echo Features:
echo - No external dependencies needed
echo - AI shot analysis and recommendations
echo - Interactive billiards table
echo - Multiple game modes
echo - Real-time trajectory calculation
echo.
pause