"""
Utility functions for the RAG ChatBot project
"""

import os
import hashlib
import json
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st

def hash_file(file_content: bytes) -> str:
    """Generate a hash for file content"""
    return hashlib.md5(file_content).hexdigest()

def get_file_stats(file_path: str) -> Dict[str, Any]:
    """Get file statistics"""
    if not os.path.exists(file_path):
        return {}
    
    stat = os.stat(file_path)
    return {
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
    }

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters that might cause issues
    text = text.replace('\x00', '')  # Remove null bytes
    text = text.replace('\r', ' ')   # Replace carriage returns
    text = text.replace('\t', ' ')   # Replace tabs
    
    return text.strip()

def save_chat_history(chat_history: List[Dict], filename: str = "chat_history.json"):
    """Save chat history to a JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")
        return False

def load_chat_history(filename: str = "chat_history.json") -> List[Dict]:
    """Load chat history from a JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Failed to load chat history: {e}")
    return []

def create_temp_directory() -> str:
    """Create a temporary directory"""
    return tempfile.mkdtemp()

def cleanup_temp_directory(temp_dir: str):
    """Clean up temporary directory"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def validate_pdf_file(file) -> bool:
    """Validate if uploaded file is a valid PDF"""
    if not file.name.lower().endswith('.pdf'):
        return False
    
    # Check file size (limit to 100MB)
    if file.size > 100 * 1024 * 1024:
        st.error(f"File {file.name} is too large. Maximum size is 100MB.")
        return False
    
    # Check if file is not empty
    if file.size == 0:
        st.error(f"File {file.name} is empty.")
        return False
    
    return True

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def count_tokens_estimate(text: str) -> int:
    """Estimate token count (rough approximation)"""
    # Rough estimate: 1 token ≈ 0.75 words
    words = len(text.split())
    return int(words / 0.75)

def format_chat_message(message: str, role: str = "user") -> Dict[str, Any]:
    """Format a chat message"""
    return {
        "role": role,
        "content": message,
        "timestamp": datetime.now().isoformat()
    }

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage (if psutil is available)"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent()
        }
    except ImportError:
        return {}

def display_system_info():
    """Display system information in Streamlit"""
    with st.expander("🔧 System Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environment")
            st.write(f"Python: {os.sys.version}")
            st.write(f"Platform: {os.name}")
            st.write(f"CPU Count: {os.cpu_count()}")
        
        with col2:
            st.subheader("Memory Usage")
            memory = get_memory_usage()
            if memory:
                st.write(f"RSS: {memory['rss']:.1f} MB")
                st.write(f"VMS: {memory['vms']:.1f} MB")
                st.write(f"Percent: {memory['percent']:.1f}%")
            else:
                st.write("Memory info not available")

def create_download_link(text: str, filename: str, link_text: str = "Download") -> str:
    """Create a download link for text content"""
    import base64
    
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64}" download="{filename}">{link_text}</a>'

def log_interaction(question: str, answer: str, sources: List[str] = None):
    """Log user interactions for analytics"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources or [],
        "question_length": len(question),
        "answer_length": len(answer)
    }
    
    # You can extend this to log to a database or file
    # For now, we'll just store in session state
    if "interaction_log" not in st.session_state:
        st.session_state.interaction_log = []
    
    st.session_state.interaction_log.append(log_entry)

def get_interaction_stats() -> Dict[str, Any]:
    """Get interaction statistics"""
    if "interaction_log" not in st.session_state:
        return {}
    
    log = st.session_state.interaction_log
    
    if not log:
        return {}
    
    return {
        "total_interactions": len(log),
        "avg_question_length": sum(entry["question_length"] for entry in log) / len(log),
        "avg_answer_length": sum(entry["answer_length"] for entry in log) / len(log),
        "first_interaction": log[0]["timestamp"] if log else None,
        "last_interaction": log[-1]["timestamp"] if log else None
    }

def display_interaction_stats():
    """Display interaction statistics in Streamlit"""
    stats = get_interaction_stats()
    
    if not stats:
        st.info("No interactions recorded yet.")
        return
    
    with st.expander("📊 Interaction Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Questions", stats["total_interactions"])
            st.metric("Avg Question Length", f"{stats['avg_question_length']:.1f} chars")
        
        with col2:
            st.metric("Avg Answer Length", f"{stats['avg_answer_length']:.1f} chars")
            st.write(f"Session started: {stats['first_interaction'][:19]}")

class StreamlitUtils:
    """Utility class for Streamlit-specific functions"""
    
    @staticmethod
    def show_loading_spinner(text: str = "Processing..."):
        """Show loading spinner with custom text"""
        return st.spinner(text)
    
    @staticmethod
    def show_success_message(message: str):
        """Show success message"""
        st.success(f"✅ {message}")
    
    @staticmethod
    def show_error_message(message: str):
        """Show error message"""
        st.error(f"❌ {message}")
    
    @staticmethod
    def show_warning_message(message: str):
        """Show warning message"""
        st.warning(f"⚠️ {message}")
    
    @staticmethod
    def show_info_message(message: str):
        """Show info message"""
        st.info(f"ℹ️ {message}")
    
    @staticmethod
    def create_two_column_layout():
        """Create a two-column layout"""
        return st.columns(2)
    
    @staticmethod
    def create_three_column_layout():
        """Create a three-column layout"""
        return st.columns(3)
    
    @staticmethod
    def add_vertical_space(lines: int = 1):
        """Add vertical space"""
        for _ in range(lines):
            st.write("") 