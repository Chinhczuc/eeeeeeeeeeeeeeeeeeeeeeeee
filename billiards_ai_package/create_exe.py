"""
Script to create standalone .exe file for Billiards AI Assistant
"""

import os
import subprocess
import sys

def create_exe():
    """Create standalone executable"""
    
    print("Creating standalone .exe for Billiards AI Assistant...")
    
    # PyInstaller command to create exe
    cmd = [
        'pyinstaller',
        '--onefile',                    # Single exe file
        '--windowed',                   # No console window
        '--name=BilliardsAI',          # Exe name
        '--icon=generated-icon.png',    # Icon (if available)
        '--add-data=.streamlit;.streamlit',  # Include streamlit config
        '--hidden-import=streamlit',
        '--hidden-import=PIL',
        '--hidden-import=numpy',
        '--hidden-import=simple_detector',
        '--hidden-import=billiards_detector',
        '--hidden-import=overlay_renderer',
        '--hidden-import=trajectory_calculator',
        '--hidden-import=utils',
        'app.py'
    ]
    
    try:
        # Install PyInstaller if not available
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
        print("✓ PyInstaller installed")
        
        # Create exe
        subprocess.check_call(cmd)
        print("✓ EXE created successfully!")
        print("📁 File location: dist/BilliardsAI.exe")
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating exe: {e}")
        return False
    
    return True

def create_batch_installer():
    """Create batch file to install and create exe"""
    batch_content = '''@echo off
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
'''
    
    with open('billiards_ai_package/create_exe.bat', 'w') as f:
        f.write(batch_content)
    
    print("✓ Created create_exe.bat")

if __name__ == "__main__":
    # Copy this script to package folder
    import shutil
    shutil.copy2(__file__, 'billiards_ai_package/')
    
    create_batch_installer()
    print("\n🎯 EXE creation tools ready!")
    print("\nTo create .exe file:")
    print("1. Download the billiards_ai_package folder")
    print("2. Run 'create_exe.bat' on Windows")
    print("3. Your BilliardsAI.exe will be in the 'dist' folder")