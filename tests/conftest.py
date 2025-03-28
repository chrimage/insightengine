"""
Pytest configuration and fixtures for InsightEngine tests.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add the parent directory to the path so we can import memory_ai
sys.path.append(str(Path(__file__).parent.parent))

from memory_ai.core.database import DatabaseManager
from memory_ai.core.config import settings, LLMSettings, DatabaseSettings, MemorySettings, AppSettings, Settings


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
        # Close the file so SQLite can open it on Windows
        tmp.close()
        yield tmp.name
        # Clean up if it still exists
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


@pytest.fixture
def temp_vector_db_path():
    """Create a temporary vector database path for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def test_settings(temp_db_path, temp_vector_db_path):
    """Create test settings with temporary paths."""
    # Save original settings
    original_settings = {
        'llm': settings.llm,
        'database': settings.database,
        'memory': settings.memory,
        'app': settings.app
    }
    
    # Create test settings
    settings.database = DatabaseSettings(
        db_path=temp_db_path,
        vector_db_path=temp_vector_db_path,
        vector_db_type="faiss",  # Use FAISS for tests since it's in-memory
        embedding_dimension=4,   # Use small embeddings for tests
        connection_pool_size=1
    )
    
    settings.llm = LLMSettings(
        google_api_key="fake_api_key_for_testing",
        default_provider="gemini",
        max_tokens=1000,
        temperature=0.0  # Deterministic for testing
    )
    
    settings.memory = MemorySettings(
        max_context_tokens=1000,
        quality_threshold=0.5,
        days_threshold=30,
        chunk_size=2,
        chunk_overlap=1
    )
    
    settings.app = AppSettings(
        batch_size=2,
        processing_threads=1,
        log_level="DEBUG"
    )
    
    yield settings
    
    # Restore original settings
    settings.llm = original_settings['llm']
    settings.database = original_settings['database']
    settings.memory = original_settings['memory']
    settings.app = original_settings['app']


@pytest.fixture
def test_db(test_settings):
    """Create a test database instance."""
    db_manager = DatabaseManager()
    yield db_manager
    # Explicitly close connections
    db_manager.engine.dispose()


@pytest.fixture
def mock_gemini_client():
    """Create a mock Gemini client for testing."""
    mock = MagicMock()
    
    # Mock the generate_embeddings method
    mock.generate_embeddings.return_value = [0.1, 0.2, 0.3, 0.4]
    
    # Mock the generate_text method
    mock.generate_text.return_value = "This is a mock response"
    
    # Mock the generate_structured_response method
    mock.generate_structured_response.return_value = {"result": "mock result"}
    
    return mock