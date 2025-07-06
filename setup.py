#!/usr/bin/env python3
"""
Setup script for RAG ChatBot - Multiple PDF Document Chat
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print the setup banner"""
    print("=" * 60)
    print("📚 RAG ChatBot - Multiple PDF Document Chat")
    print("=" * 60)
    print("Setting up your RAG chatbot environment...")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path("rag_env")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return str(venv_path)
    
    print("🔧 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "rag_env"], check=True)
        print("✅ Virtual environment created successfully")
        return str(venv_path)
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        sys.exit(1)

def install_dependencies(venv_path):
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Determine the pip executable
    if os.name == 'nt':  # Windows
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/Mac
        pip_path = venv_path / "bin" / "pip"
    
    try:
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def setup_environment_file():
    """Set up the .env file"""
    env_file = Path(".env")
    env_template = Path("env_template.txt")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    if not env_template.exists():
        print("❌ env_template.txt not found")
        return
    
    print("🔧 Setting up environment file...")
    try:
        shutil.copy(env_template, env_file)
        print("✅ .env file created from template")
        print("⚠️  Please edit .env file and add your GROQ_API_KEY")
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")

def check_gitignore():
    """Check if .env is in .gitignore"""
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        print("🔧 Creating .gitignore file...")
        with open(gitignore_path, "w") as f:
            f.write("# Environment variables\n")
            f.write(".env\n")
            f.write("\n# Virtual environment\n")
            f.write("rag_env/\n")
            f.write("__pycache__/\n")
            f.write("*.pyc\n")
            f.write("\n# IDE files\n")
            f.write(".vscode/\n")
            f.write(".idea/\n")
        print("✅ .gitignore file created")
    else:
        print("✅ .gitignore file exists")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Edit the .env file and add your GROQ_API_KEY")
    print("   - Get your free API key from: https://console.groq.com/")
    print("   - Add it to the .env file: GROQ_API_KEY=your_key_here")
    print()
    print("2. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   rag_env\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print("   source rag_env/bin/activate")
    print()
    print("3. Run the application:")
    print("   streamlit run app.py")
    print()
    print("4. Open your browser and go to: http://localhost:8501")
    print()
    print("📚 Happy Document Chatting!")
    print("=" * 60)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    venv_path = create_virtual_environment()
    
    # Install dependencies
    install_dependencies(venv_path)
    
    # Setup environment file
    setup_environment_file()
    
    # Check gitignore
    check_gitignore()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 