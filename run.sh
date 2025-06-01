#!/bin/bash
echo "Installing dependencies..."
pip3 install streamlit pillow numpy
echo ""
echo "Starting Billiards AI Assistant..."
streamlit run app.py