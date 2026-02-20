@echo off
echo ========================================
echo   Starting FocusAI (Full Stack)
echo ========================================
echo.
echo Starting Backend...
start cmd /k "cd /d %~dp0backend && pip install -r requirements.txt && python app.py"
echo.
echo Waiting 5 seconds for backend...
timeout /t 5 /nobreak
echo.
echo Starting Frontend...
start cmd /k "cd /d %~dp0frontend && npm run dev"
echo.
echo ========================================
echo   Backend: http://localhost:5000
echo   Frontend: http://localhost:3000
echo ========================================
echo.
pause
