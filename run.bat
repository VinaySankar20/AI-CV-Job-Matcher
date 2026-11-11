@echo off
REM Double-click to start the CV Job Matcher (after the one-time setup in README.md)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
