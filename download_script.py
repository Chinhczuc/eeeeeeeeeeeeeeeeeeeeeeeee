#!/usr/bin/env python3
"""
Script to download all billiards AI files
Run this to get all necessary files for your local machine
"""

import os
import shutil

def create_billiards_package():
    """Create a complete package of billiards AI files"""
    
    # List of files to include
    files_to_copy = [
        'app.py',
        'simple_detector.py', 
        'billiards_detector.py',
        'overlay_renderer.py',
        'trajectory_calculator.py',
        'utils.py',
        'install_guide.md'
    ]
    
    # Create package directory
    package_dir = 'billiards_ai_package'
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)
    
    # Copy files
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, package_dir)
            print(f"✓ Copied {file}")
        else:
            print(f"✗ Missing {file}")
    
    # Create requirements.txt
    requirements = """streamlit>=1.28.0
pillow>=9.0.0
numpy>=1.21.0
opencv-python>=4.5.0
ultralytics>=8.0.0"""
    
    with open(f"{package_dir}/requirements.txt", 'w') as f:
        f.write(requirements)
    print("✓ Created requirements.txt")
    
    # Create run script for Windows
    run_script = """@echo off
echo Installing dependencies...
pip install streamlit pillow numpy
echo.
echo Starting Billiards AI Assistant...
streamlit run app.py
pause"""
    
    with open(f"{package_dir}/run.bat", 'w') as f:
        f.write(run_script)
    print("✓ Created run.bat")
    
    # Create run script for Mac/Linux
    run_script_unix = """#!/bin/bash
echo "Installing dependencies..."
pip3 install streamlit pillow numpy
echo ""
echo "Starting Billiards AI Assistant..."
streamlit run app.py"""
    
    with open(f"{package_dir}/run.sh", 'w') as f:
        f.write(run_script_unix)
    os.chmod(f"{package_dir}/run.sh", 0o755)
    print("✓ Created run.sh")
    
    print(f"\n🎯 Package created in '{package_dir}' folder")
    print("\nTo use:")
    print("1. Copy the entire 'billiards_ai_package' folder to your computer")
    print("2. Windows: Double-click 'run.bat'")
    print("3. Mac/Linux: Run './run.sh' in terminal")
    
if __name__ == "__main__":
    create_billiards_package()