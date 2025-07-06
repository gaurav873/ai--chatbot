# 🐍 Python Installation & Setup Guide

## Method 1: Install Python from Microsoft Store (Recommended)

1. **Open Microsoft Store:**
   - Press `Win + S` and search for "Microsoft Store"
   - Open Microsoft Store

2. **Search for Python:**
   - In Microsoft Store, search for "Python 3.11" or "Python 3.10"
   - Select "Python 3.11" (or latest version)
   - Click "Install"

3. **Verify Installation:**
   - Open a new PowerShell/Command Prompt
   - Run: `python --version`
   - Should show: `Python 3.11.x`

## Method 2: Install from Python.org

1. **Download Python:**
   - Go to https://www.python.org/downloads/
   - Click "Download Python 3.11.x"

2. **Install Python:**
   - Run the downloaded installer
   - ✅ **IMPORTANT:** Check "Add Python to PATH"
   - Click "Install Now"

3. **Verify Installation:**
   - Open a new PowerShell/Command Prompt
   - Run: `python --version`

## 🚀 After Python is Installed

### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python -m venv rag_env

# Activate virtual environment
# On Windows:
rag_env\Scripts\activate

# You should see (rag_env) in your prompt
```

### Step 2: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables
1. Make sure you have your Google API key in the `.env` file
2. The `.env` file should look like:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

### Step 4: Run the Application
```bash
# Run the basic version
streamlit run app.py

# OR run the enhanced version (recommended)
streamlit run enhanced_app.py
```

## 🔧 Troubleshooting

### If Python is not found:
- Make sure you checked "Add Python to PATH" during installation
- Restart your computer
- Try using `py` instead of `python`

### If pip is not found:
```bash
python -m pip install --upgrade pip
```

### If streamlit is not found:
```bash
pip install streamlit
```

## 🎯 Quick Test
Once Python is installed, test it:
```bash
python --version
pip --version
```

Both commands should work without errors.

## 📞 Need Help?
If you're still having issues:
1. Restart your computer after Python installation
2. Open a new PowerShell window
3. Make sure you're in the project directory
4. Try the commands again

Your RAG chatbot project is ready to run once Python is properly installed! 🚀 