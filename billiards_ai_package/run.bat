@echo off
echo Installing dependencies...
pip install streamlit pillow numpy
echo.
echo Starting Billiards AI Assistant...
streamlit run app.py
pause