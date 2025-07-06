@echo off
REM RAG ChatBot - Setup and Run Script for Windows
REM This script sets up the environment and runs the RAG chatbot

echo ============================================================
echo 📚 RAG ChatBot - Multiple PDF Document Chat
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if .env file exists
if not exist ".env" (
    echo.
    echo 🔧 Setting up environment file...
    if exist "env_template.txt" (
        copy "env_template.txt" ".env" >nul
        echo ✅ .env file created from template
        echo ⚠️  Please edit .env file and add your GROQ_API_KEY
        echo    Get your free API key from: https://console.groq.com/
        echo.
        echo Press any key to open .env file for editing...
        pause >nul
        notepad .env
    ) else (
        echo ❌ env_template.txt not found
        pause
        exit /b 1
    )
) else (
    echo ✅ .env file exists
)

REM Check if virtual environment exists
if not exist "rag_env" (
    echo.
    echo 🔧 Creating virtual environment...
    python -m venv rag_env
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment exists
)

REM Activate virtual environment and install dependencies
echo.
echo 📦 Installing dependencies...
call rag_env\Scripts\activate.bat
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed successfully

REM Run the application
echo.
echo 🚀 Starting RAG ChatBot...
echo.
echo 📋 Instructions:
echo 1. Upload your PDF documents using the sidebar
echo 2. Click "Process Documents" to analyze them
echo 3. Ask questions about your documents
echo 4. The app will answer based on your document content
echo.
echo 🌐 The app will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run app.py

REM If we get here, the app was stopped
echo.
echo 👋 RAG ChatBot stopped
pause 