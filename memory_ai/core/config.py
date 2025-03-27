# Configuration settings

class Config:
    """Configuration settings for the memory system."""
    
    # Database settings
    DATABASE_PATH = "memory.db"
    
    # API keys and model settings
    GOOGLE_API_KEY = None
    LLM_MODEL = "gemini-2.0-flash"
    EMBEDDING_MODEL = "models/text-embedding-004"
    
    # Memory parameters
    BATCH_SIZE = 10
    MAX_TOKENS = 8000
    QUALITY_THRESHOLD = 0.6
    
    # Forgetting mechanism
    DAYS_THRESHOLD = 180
    
    def __init__(self, **kwargs):
        """Initialize configuration with overrides."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)