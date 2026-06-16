@echo off
cd "C:\projects\AI Classroom Intelligence system"
call classroom_env\Scripts\activate
streamlit run app.py --server.address 0.0.0.0 --server.port 8501