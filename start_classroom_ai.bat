@echo off
echo Starting AI Classroom Intelligence System...
cd /d "C:\projects\AI Classroom Intelligence system"

echo Starting Streamlit...
start "Streamlit" cmd /k "cd /d "C:\projects\AI Classroom Intelligence system" && call classroom_env\Scripts\activate && streamlit run app.py --server.port 8501"

echo Waiting for Streamlit to load (60 seconds)...
timeout /t 60

echo Starting ngrok...
start "Ngrok" cmd /k "cd /d "C:\projects\AI Classroom Intelligence system" && .\ngrok http --url=badness-waving-renewed.ngrok-free.dev 8501 --request-header-add "ngrok-skip-browser-warning:true""

echo.
echo Done! Your app URL is: https://badness-waving-renewed.ngrok-free.dev
pause