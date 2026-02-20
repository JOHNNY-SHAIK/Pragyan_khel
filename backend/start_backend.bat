@echo off
echo ========================================
echo   FocusAI Backend - Cricket Edition
echo   YOLOv8 + OpenCV + Deep SORT
echo ========================================
echo.
cd /d "%~dp0"
pip install -r requirements.txt
echo.
echo Starting server on http://localhost:5000
echo.
python app.py
pause
