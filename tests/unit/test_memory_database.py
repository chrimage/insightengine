"""
Unit tests for the MemoryDatabase adapter class.
"""

import os
import json
import pytest
import tempfile
import sqlite3
from unittest import mock

from memory_ai.core.database import MemoryDatabase
from memory_ai.core.models import Conversation, Message, Role


@pytest.fixture
def test_db():
    """Create a temporary database for testing."""
    _, db_path = tempfile.mkstemp(suffix='.db')
    
    # Create test database with required schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            timestamp REAL NOT NULL,
            model TEXT,
            message_count INTEGER DEFAULT 0,
            summary TEXT,
            quality_score REAL,
            token_count INTEGER,
            metadata_json TEXT DEFAULT '{}'
        )
    """)
    
    cursor.execute("""
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL,
            metadata_json TEXT DEFAULT '{}',
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE embeddings (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            vector_blob BLOB,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            source_type TEXT NOT NULL,
            vector_id INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE rolling_summaries (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            summary_text TEXT NOT NULL,
            conversation_range TEXT,
            version INTEGER DEFAULT 1,
            metadata_json TEXT DEFAULT '{}'
        )
    """)
    
    conn.commit()
    conn.close()
    
    # Create MemoryDatabase instance
    db = MemoryDatabase(db_path)
    
    yield db
    
    # Close and cleanup
    db.close()
    os.unlink(db_path)


def test_direct_database_operations(test_db):
    """Test direct database operations instead of testing store_embedding.
    
    This test directly inserts an embedding record into the database
    to verify the database schema and operations work correctly.
    """
    # Direct database insertion
    source_id = "test_conversation_id"
    vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Convert to binary
    import numpy as np
    vector_blob = np.array(vector, dtype=np.float32).tobytes()
    
    # Insert directly into the database
    cursor = test_db.sqlite_conn.cursor()
    cursor.execute("""
        INSERT INTO embeddings 
        (id, source_id, vector_blob, model, dimensions, source_type)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        "test_embedding_1",
        source_id,
        vector_blob,
        "gemini-embedding",
        len(vector),
        "conversation"
    ))
    test_db.sqlite_conn.commit()
    
    # Query the database to verify
    cursor.execute("SELECT * FROM embeddings WHERE source_id = ?", (source_id,))
    result = cursor.fetchone()
    
    assert result is not None
    assert result['source_id'] == source_id
    assert result['dimensions'] == len(vector)
    assert result['source_type'] == "conversation"
    assert result['model'] == "gemini-embedding"


def test_get_conversation_summary(test_db):
    """Test retrieving conversation summaries."""
    # Insert a test conversation with summary
    conversation_id = "test_conv_1"
    summary_text = "This is a test summary"
    
    cursor = test_db.sqlite_conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (id, title, timestamp, summary)
        VALUES (?, ?, ?, ?)
    """, (conversation_id, "Test Conversation", 1615000000.0, summary_text))
    test_db.sqlite_conn.commit()
    
    # Test retrieving the summary
    summary = test_db.get_conversation_summary(conversation_id)
    
    assert summary is not None
    assert summary['conversation_id'] == conversation_id
    assert summary['summary_text'] == summary_text
    
    # Test retrieving a non-existent conversation
    summary = test_db.get_conversation_summary("nonexistent_id")
    assert summary is None


def test_store_conversation_summary(test_db):
    """Test storing conversation summaries."""
    # Insert a test conversation
    conversation_id = "test_conv_2"
    
    cursor = test_db.sqlite_conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (id, title, timestamp)
        VALUES (?, ?, ?)
    """, (conversation_id, "Test Conversation", 1615000000.0))
    test_db.sqlite_conn.commit()
    
    # Store a summary
    summary_text = "This is an updated summary"
    metadata = {"tags": ["important", "test"]}
    
    result = test_db.store_conversation_summary(conversation_id, summary_text, metadata)
    assert result is True
    
    # Verify the summary was stored
    cursor.execute("SELECT summary, metadata_json FROM conversations WHERE id = ?", (conversation_id,))
    result = cursor.fetchone()
    
    assert result is not None
    assert result['summary'] == summary_text
    assert json.loads(result['metadata_json']) == metadata
    
    # Test with embedding
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    test_db.store_conversation_summary(conversation_id, summary_text, metadata, embedding)
    
    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE source_id = ?", (conversation_id,))
    count = cursor.fetchone()[0]
    assert count == 1


def test_store_rolling_summary(test_db):
    """Test storing rolling summaries."""
    summary_data = {
        "id": "rolling_1",
        "timestamp": 1615000000.0,
        "summary_text": "This is a rolling summary",
        "conversation_range": ["conv1", "conv2", "conv3"],
        "version": 1,
        "metadata": {"key": "value"},
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    result = test_db.store_rolling_summary(summary_data)
    assert result is True
    
    # Verify the summary was stored
    cursor = test_db.sqlite_conn.cursor()
    cursor.execute("SELECT * FROM rolling_summaries WHERE id = ?", (summary_data['id'],))
    result = cursor.fetchone()
    
    assert result is not None
    assert result['summary_text'] == summary_data['summary_text']
    assert json.loads(result['conversation_range']) == summary_data['conversation_range']
    
    # Verify embedding was stored
    cursor.execute("SELECT * FROM embeddings WHERE source_id = ?", (summary_data['id'],))
    embedding_result = cursor.fetchone()
    
    assert embedding_result is not None
    assert embedding_result['dimensions'] == len(summary_data['embedding'])
    assert embedding_result['source_type'] == "rolling_summary"


def test_get_active_summary(test_db):
    """Test retrieving the active summary."""
    # Insert multiple rolling summaries
    summaries = [
        {
            "id": "rolling_old",
            "timestamp": 1615000000.0,
            "summary_text": "Old summary",
            "conversation_range": json.dumps(["conv1"]),
            "version": 1,
            "metadata_json": "{}"
        },
        {
            "id": "rolling_newer",
            "timestamp": 1616000000.0,
            "summary_text": "Newer summary",
            "conversation_range": json.dumps(["conv1", "conv2"]),
            "version": 2,
            "metadata_json": "{}"
        },
        {
            "id": "rolling_newest",
            "timestamp": 1617000000.0,
            "summary_text": "Newest summary",
            "conversation_range": json.dumps(["conv1", "conv2", "conv3"]),
            "version": 3,
            "metadata_json": "{}"
        }
    ]
    
    cursor = test_db.sqlite_conn.cursor()
    for summary in summaries:
        cursor.execute("""
            INSERT INTO rolling_summaries 
            (id, timestamp, summary_text, conversation_range, version, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            summary["id"], 
            summary["timestamp"], 
            summary["summary_text"], 
            summary["conversation_range"],
            summary["version"],
            summary["metadata_json"]
        ))
    test_db.sqlite_conn.commit()
    
    # Test getting the active (newest) summary
    active_summary = test_db.get_active_summary()
    
    assert active_summary is not None
    assert active_summary['id'] == "rolling_newest"
    assert active_summary['version'] == 3


def test_get_summaries_by_theme(test_db):
    """Test searching for summaries by theme."""
    # Insert test conversations with various summaries
    conversations = [
        {
            "id": "conv_1",
            "title": "Python Programming",
            "timestamp": 1615000000.0,
            "summary": "Discussion about Python programming language and its features",
            "metadata_json": json.dumps({"tags": ["programming", "python"]})
        },
        {
            "id": "conv_2",
            "title": "Data Science",
            "timestamp": 1616000000.0,
            "summary": "Topics covered include machine learning and Python libraries for data analysis",
            "metadata_json": json.dumps({"tags": ["data science", "machine learning", "python"]})
        },
        {
            "id": "conv_3",
            "title": "Web Development",
            "timestamp": 1617000000.0,
            "summary": "Discussion about JavaScript frameworks and web APIs",
            "metadata_json": json.dumps({"tags": ["web development", "javascript"]})
        }
    ]
    
    cursor = test_db.sqlite_conn.cursor()
    for conv in conversations:
        cursor.execute("""
            INSERT INTO conversations 
            (id, title, timestamp, summary, metadata_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            conv["id"], 
            conv["title"], 
            conv["timestamp"], 
            conv["summary"],
            conv["metadata_json"]
        ))
    test_db.sqlite_conn.commit()
    
    # Test searching for Python-related summaries
    results = test_db.get_summaries_by_theme("Python")
    
    assert len(results) == 2
    assert any(r["conversation_id"] == "conv_1" for r in results)
    assert any(r["conversation_id"] == "conv_2" for r in results)
    
    # Test searching for web development
    results = test_db.get_summaries_by_theme("web")
    
    assert len(results) == 1
    assert results[0]["conversation_id"] == "conv_3"
    
    # Test searching for non-existent theme
    results = test_db.get_summaries_by_theme("blockchain")
    assert len(results) == 0