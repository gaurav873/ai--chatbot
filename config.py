import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Groq AI Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    
    # Text Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Vector Store Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    SEARCH_K = int(os.getenv("SEARCH_K", "3"))
    
    # Streamlit Configuration
    PAGE_TITLE = "RAG ChatBot - Multiple PDF Chat"
    PAGE_ICON = "📚"
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required. Please set it in your .env file")
        
        return True 