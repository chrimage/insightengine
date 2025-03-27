# Configuration settings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the memory system."""
    
    # Database settings
    DATABASE_PATH = os.environ.get("DB_PATH", "memory.db")
    
    # API keys and model settings
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", None)
    LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/text-embedding-004")
    
    # Memory parameters
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8000"))
    QUALITY_THRESHOLD = float(os.environ.get("QUALITY_THRESHOLD", "0.6"))
    
    # Forgetting mechanism
    DAYS_THRESHOLD = int(os.environ.get("DAYS_THRESHOLD", "180"))
    
    def __init__(self, **kwargs):
        """Initialize configuration with overrides."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)