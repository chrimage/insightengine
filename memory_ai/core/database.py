"""
Database layer with SQLAlchemy ORM and repository pattern.

This module implements the data access layer for the InsightEngine system.
It uses SQLAlchemy for ORM and transaction management and provides
a clean repository interface for each domain entity.
"""

from contextlib import contextmanager
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Generic, TypeVar

import json
import sqlalchemy as sa
from sqlalchemy import create_engine, text, Column, String, Float, Integer, Boolean, ForeignKey, MetaData, Table
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session, Mapped, mapped_column
from sqlalchemy.pool import QueuePool
from sqlalchemy.ext.declarative import declared_attr

from memory_ai.core.config import get_settings
from memory_ai.core.models import (
    Conversation, Message, Chunk, Embedding, RollingSummary, Insight, Evaluation, 
    SourceType, Role, InsightCategory
)

# Setup logging
logger = logging.getLogger(__name__)

# Base class for all ORM models
Base = declarative_base()

# Type variable for repository generic
T = TypeVar('T')


class ConversationORM(Base):
    """ORM model for conversations."""
    
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    timestamp = Column(Float, nullable=False)
    model = Column(String)
    message_count = Column(Integer, default=0)
    summary = Column(String)
    quality_score = Column(Float)
    token_count = Column(Integer)
    metadata_json = Column(String, default="{}")  # JSON string
    
    # Relationships
    messages = relationship("MessageORM", back_populates="conversation", cascade="all, delete-orphan")


class MessageORM(Base):
    """ORM model for messages."""
    
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    timestamp = Column(Float, nullable=False)
    metadata_json = Column(String, default="{}")  # JSON string
    
    # Relationships
    conversation = relationship("ConversationORM", back_populates="messages")


class ChunkORM(Base):
    """ORM model for chunked content."""
    
    __tablename__ = "chunks"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    message_ids = Column(String)  # JSON array of message IDs
    content = Column(String, nullable=False)
    timestamp = Column(Float, nullable=False)
    metadata_json = Column(String, default="{}")  # JSON string
    
    # Relationships
    conversation = relationship("ConversationORM")


class EmbeddingORM(Base):
    """ORM model for vector embeddings."""
    
    __tablename__ = "embeddings"
    
    id = Column(String, primary_key=True)
    source_id = Column(String, nullable=False)
    vector_blob = Column(sa.LargeBinary)  # Binary serialized vector
    model = Column(String, nullable=False)
    dimensions = Column(Integer, nullable=False)
    source_type = Column(String, nullable=False)
    vector_id = Column(Integer)  # Reference to external vector store


class RollingSummaryORM(Base):
    """ORM model for rolling summaries."""
    
    __tablename__ = "rolling_summaries"
    
    id = Column(String, primary_key=True)
    timestamp = Column(Float, nullable=False)
    summary_text = Column(String, nullable=False)
    conversation_range = Column(String)  # JSON array of conversation IDs
    version = Column(Integer, default=1)
    metadata_json = Column(String, default="{}")  # JSON string


class InsightORM(Base):
    """ORM model for insights."""
    
    __tablename__ = "insights"
    
    id = Column(String, primary_key=True)
    text = Column(String, nullable=False)
    category = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(Float, nullable=False)
    updated_at = Column(Float, nullable=False)
    evidence = Column(String)  # JSON array
    application_count = Column(Integer, default=0)
    success_rate = Column(Float)
    applications = Column(String)  # JSON array
    metadata_json = Column(String, default="{}")  # JSON string


class EvaluationORM(Base):
    """ORM model for response evaluations."""
    
    __tablename__ = "evaluations"
    
    id = Column(String, primary_key=True)
    timestamp = Column(Float, nullable=False)
    query = Column(String, nullable=False)
    evaluation = Column(String, nullable=False)  # JSON object
    structured = Column(Boolean, default=True)
    meta_evaluation = Column(String)  # JSON object


class BaseRepository(Generic[T]):
    """Base repository interface for data access."""
    
    def __init__(self, session: Session):
        """Initialize with a database session.
        
        Args:
            session: SQLAlchemy session object
        """
        self.session = session


class ConversationRepository(BaseRepository[Conversation]):
    """Repository for conversation operations."""
    
    def add(self, conversation: Conversation) -> str:
        """Add a new conversation.
        
        Args:
            conversation: Conversation object
            
        Returns:
            str: ID of the created conversation
        """
        orm_obj = ConversationORM(
            id=conversation.id,
            title=conversation.title,
            timestamp=conversation.timestamp,
            model=conversation.model,
            message_count=len(conversation.messages) if conversation.messages else 0,
            summary=conversation.summary,
            quality_score=conversation.quality_score,
            token_count=conversation.token_count,
            metadata_json=json.dumps(conversation.metadata)
        )
        
        self.session.add(orm_obj)
        return orm_obj.id
    
    def get(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Optional[Conversation]: Found conversation or None
        """
        orm_obj = self.session.query(ConversationORM).get(conversation_id)
        if not orm_obj:
            return None
        
        return self._to_domain(orm_obj)
    
    def update(self, conversation: Conversation) -> bool:
        """Update an existing conversation.
        
        Args:
            conversation: Updated conversation object
            
        Returns:
            bool: True if updated, False if not found
        """
        orm_obj = self.session.query(ConversationORM).get(conversation.id)
        if not orm_obj:
            return False
        
        orm_obj.title = conversation.title
        orm_obj.timestamp = conversation.timestamp
        orm_obj.model = conversation.model
        orm_obj.message_count = len(conversation.messages) if conversation.messages else 0
        orm_obj.summary = conversation.summary
        orm_obj.quality_score = conversation.quality_score
        orm_obj.token_count = conversation.token_count
        orm_obj.metadata_json = json.dumps(conversation.metadata)
        
        return True
    
    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation by ID.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            bool: True if deleted, False if not found
        """
        orm_obj = self.session.query(ConversationORM).get(conversation_id)
        if not orm_obj:
            return False
        
        self.session.delete(orm_obj)
        return True
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[Conversation]:
        """Get multiple conversations with pagination.
        
        Args:
            limit: Maximum number of conversations to retrieve
            offset: Number of conversations to skip
            
        Returns:
            List[Conversation]: List of conversations
        """
        orm_objs = self.session.query(ConversationORM) \
            .order_by(ConversationORM.timestamp.desc()) \
            .limit(limit).offset(offset).all()
            
        return [self._to_domain(obj) for obj in orm_objs]
    
    def get_by_time_range(self, start_time: float, end_time: float) -> List[Conversation]:
        """Get conversations within a time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List[Conversation]: List of conversations
        """
        orm_objs = self.session.query(ConversationORM) \
            .filter(ConversationORM.timestamp >= start_time) \
            .filter(ConversationORM.timestamp <= end_time) \
            .order_by(ConversationORM.timestamp).all()
            
        return [self._to_domain(obj) for obj in orm_objs]
    
    def _to_domain(self, orm_obj: ConversationORM) -> Conversation:
        """Convert ORM object to domain model.
        
        Args:
            orm_obj: ORM object
            
        Returns:
            Conversation: Domain object
        """
        metadata = {}
        if orm_obj.metadata_json:
            try:
                metadata = json.loads(orm_obj.metadata_json)
            except json.JSONDecodeError:
                pass
        
        # Don't load messages here to avoid N+1 query problems
        # They should be loaded separately when needed
        return Conversation(
            id=orm_obj.id,
            title=orm_obj.title,
            timestamp=orm_obj.timestamp,
            model=orm_obj.model,
            summary=orm_obj.summary,
            quality_score=orm_obj.quality_score,
            token_count=orm_obj.token_count,
            metadata=metadata,
            messages=[]  # Empty list to be loaded separately
        )


class MessageRepository(BaseRepository[Message]):
    """Repository for message operations."""
    
    def add(self, message: Message) -> str:
        """Add a new message.
        
        Args:
            message: Message object
            
        Returns:
            str: ID of the created message
        """
        orm_obj = MessageORM(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role.value,
            content=message.content,
            timestamp=message.timestamp,
            metadata_json=json.dumps(message.metadata)
        )
        
        self.session.add(orm_obj)
        return orm_obj.id
    
    def get(self, message_id: str) -> Optional[Message]:
        """Get a message by ID.
        
        Args:
            message_id: ID of the message
            
        Returns:
            Optional[Message]: Found message or None
        """
        orm_obj = self.session.query(MessageORM).get(message_id)
        if not orm_obj:
            return None
            
        return self._to_domain(orm_obj)
    
    def get_by_conversation(self, conversation_id: str) -> List[Message]:
        """Get all messages for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List[Message]: List of messages
        """
        orm_objs = self.session.query(MessageORM) \
            .filter(MessageORM.conversation_id == conversation_id) \
            .order_by(MessageORM.timestamp).all()
            
        return [self._to_domain(obj) for obj in orm_objs]
    
    def _to_domain(self, orm_obj: MessageORM) -> Message:
        """Convert ORM object to domain model.
        
        Args:
            orm_obj: ORM object
            
        Returns:
            Message: Domain object
        """
        metadata = {}
        if orm_obj.metadata_json:
            try:
                metadata = json.loads(orm_obj.metadata_json)
            except json.JSONDecodeError:
                pass
        
        return Message(
            id=orm_obj.id,
            conversation_id=orm_obj.conversation_id,
            role=Role(orm_obj.role),
            content=orm_obj.content,
            timestamp=orm_obj.timestamp,
            metadata=metadata
        )


class ChunkRepository(BaseRepository[Chunk]):
    """Repository for chunk operations."""
    
    def add(self, chunk: Chunk) -> str:
        """Add a new chunk.
        
        Args:
            chunk: Chunk object
            
        Returns:
            str: ID of the created chunk
        """
        orm_obj = ChunkORM(
            id=chunk.id,
            conversation_id=chunk.conversation_id,
            message_ids=json.dumps(chunk.message_ids),
            content=chunk.content,
            timestamp=chunk.timestamp,
            metadata_json=json.dumps(chunk.metadata)
        )
        
        self.session.add(orm_obj)
        return orm_obj.id
    
    def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Optional[Chunk]: Found chunk or None
        """
        orm_obj = self.session.query(ChunkORM).get(chunk_id)
        if not orm_obj:
            return None
            
        return self._to_domain(orm_obj)
    
    def get_by_conversation(self, conversation_id: str) -> List[Chunk]:
        """Get all chunks for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List[Chunk]: List of chunks
        """
        orm_objs = self.session.query(ChunkORM) \
            .filter(ChunkORM.conversation_id == conversation_id) \
            .order_by(ChunkORM.timestamp).all()
            
        return [self._to_domain(obj) for obj in orm_objs]
    
    def _to_domain(self, orm_obj: ChunkORM) -> Chunk:
        """Convert ORM object to domain model.
        
        Args:
            orm_obj: ORM object
            
        Returns:
            Chunk: Domain object
        """
        message_ids = []
        if orm_obj.message_ids:
            try:
                message_ids = json.loads(orm_obj.message_ids)
            except json.JSONDecodeError:
                pass
                
        metadata = {}
        if orm_obj.metadata_json:
            try:
                metadata = json.loads(orm_obj.metadata_json)
            except json.JSONDecodeError:
                pass
        
        return Chunk(
            id=orm_obj.id,
            conversation_id=orm_obj.conversation_id,
            message_ids=message_ids,
            content=orm_obj.content,
            timestamp=orm_obj.timestamp,
            metadata=metadata
        )


class UnitOfWork:
    """Manages a transaction with multiple repositories."""
    
    def __init__(self, session_factory: sessionmaker):
        """Initialize with a session factory.
        
        Args:
            session_factory: SQLAlchemy sessionmaker
        """
        self.session_factory = session_factory
        self.session = None
        
        # Repositories will be instantiated during __enter__
        self.conversations = None
        self.messages = None
        self.chunks = None
        # Add other repositories as needed
    
    def __enter__(self):
        """Start a new transaction.
        
        Returns:
            UnitOfWork: Self for context manager
        """
        self.session = self.session_factory()
        
        # Create repositories with the current session
        self.conversations = ConversationRepository(self.session)
        self.messages = MessageRepository(self.session)
        self.chunks = ChunkRepository(self.session)
        # Add other repositories as needed
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the transaction with commit or rollback.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        if exc_type:
            self.rollback()
        else:
            self.commit()
            
        self.session.close()
        self.session = None
    
    def commit(self):
        """Commit the current transaction."""
        self.session.commit()
    
    def rollback(self):
        """Rollback the current transaction."""
        self.session.rollback()


class DatabaseManager:
    """Manages database connections and migrations."""
    
    def __init__(self):
        """Initialize database connection and setup."""
        settings = get_settings()
        db_path = settings.database.db_path
        
        # Create directory if needed
        db_dir = Path(db_path).parent
        if db_dir and not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create engine with connection pooling
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            poolclass=QueuePool,
            pool_size=settings.database.connection_pool_size,
            max_overflow=10,
            connect_args={"check_same_thread": False},
        )
        
        # Create session factory
        self.session_factory = sessionmaker(bind=self.engine)
        
        # Initialize schema
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Create tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        
        # Initialize any additional database functionality
        # such as extensions or custom functions
        with self.engine.connect() as conn:
            # Example: Enable foreign keys in SQLite
            conn.execute(text("PRAGMA foreign_keys = ON"))
            
            # Here we would enable vector search extensions
            # This is a placeholder for integrating with vector DBs
    
    def get_unit_of_work(self) -> UnitOfWork:
        """Get a unit of work for transaction management.
        
        Returns:
            UnitOfWork: A new unit of work
        """
        return UnitOfWork(self.session_factory)
    
    @contextmanager
    def session(self) -> Session:
        """Context manager for database sessions.
        
        Yields:
            Session: SQLAlchemy session
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Create database manager singleton
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the database manager instance.
    
    Returns:
        DatabaseManager: The database manager
    """
    return db_manager


class MemoryDatabase:
    """Legacy adapter class for the indexer tool.
    
    This class provides a compatibility layer for code that was written
    for the older database API, allowing it to work with the new repository
    pattern implementation.
    """
    
    def __init__(self, db_path: str):
        """Initialize the memory database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.manager = DatabaseManager()
        self.engine = self.manager.engine
        self.conn = self.engine

        # For direct SQL queries - use SQLite connection for legacy code
        import sqlite3
        self.sqlite_conn = sqlite3.connect(db_path)
        self.sqlite_conn.row_factory = sqlite3.Row
        
    def store_embedding(self, source_id: str, vector: List[float], source_type: str = "conversation", model: str = "gemini-embedding") -> None:
        """Store an embedding in the database.
        
        Args:
            source_id: ID of the source (conversation, chunk, etc.)
            vector: Embedding vector
            source_type: Type of the source (defaults to "conversation")
            model: Name of the embedding model (defaults to "gemini-embedding")
        """
        # Convert source_type string to enum
        try:
            source_type_enum = SourceType(source_type)
        except ValueError:
            source_type_enum = SourceType.CONVERSATION_SUMMARY
            
        with self.manager.session() as session:
            # Convert vector to binary for storage
            import numpy as np
            import uuid as uuid_module  # Import uuid to avoid name collision
            vector_blob = np.array(vector, dtype=np.float32).tobytes()
            
            # Create embedding ORM object
            embedding = EmbeddingORM(
                id=f"emb_{uuid_module.uuid4()}",
                source_id=source_id,
                vector_blob=vector_blob,
                model=model,
                dimensions=len(vector),
                source_type=source_type_enum.value
            )
            
            session.add(embedding)
    
    # Legacy methods for the summarization tool
    def get_active_summary(self):
        """Get the active global summary."""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            SELECT * FROM rolling_summaries
            ORDER BY version DESC, timestamp DESC
            LIMIT 1
        """)
        summary = cursor.fetchone()
        if summary:
            return dict(summary)
        return None
    
    def get_conversation_summary(self, conv_id):
        """Get the summary for a specific conversation."""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            SELECT id as conversation_id, summary as summary_text, metadata_json as metadata
            FROM conversations
            WHERE id = ?
        """, (conv_id,))
        summary = cursor.fetchone()
        if summary and summary['summary_text']:
            return dict(summary)
        return None
    
    def get_summaries_by_theme(self, query, embedding=None):
        """Search for summaries by theme."""
        # This is a simplified version - in reality would use vector search
        cursor = self.sqlite_conn.cursor()
        # Get conversation summaries containing the query
        query_param = f"%{query}%"
        cursor.execute("""
            SELECT id as conversation_id, title, summary as summary_text, metadata_json as metadata, timestamp
            FROM conversations
            WHERE summary LIKE ? OR metadata_json LIKE ?
            ORDER BY timestamp DESC
            LIMIT 10
        """, (query_param, query_param))
        return [dict(row) for row in cursor.fetchall()]
    
    def store_conversation_summary(self, conversation_id, summary_text, metadata=None, embedding=None):
        """Store a summary for a specific conversation."""
        cursor = self.sqlite_conn.cursor()
        
        # Convert metadata to JSON if it's a dictionary
        if isinstance(metadata, dict):
            metadata_json = json.dumps(metadata)
        else:
            metadata_json = "{}"
            
        # Update the conversation with the summary
        cursor.execute("""
            UPDATE conversations
            SET summary = ?, metadata_json = ?
            WHERE id = ?
        """, (summary_text, metadata_json, conversation_id))
        
        # Store the embedding if provided
        if embedding is not None:
            import numpy as np
            import uuid as uuid_module
            embedding_id = f"summary_{conversation_id}"
            
            # Store in embeddings table
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings
                (id, source_id, vector_blob, model, dimensions, source_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                embedding_id,
                conversation_id,
                np.array(embedding, dtype=np.float32).tobytes(),
                "summary-embedding",
                len(embedding),
                "conv_summary"
            ))
        
        self.sqlite_conn.commit()
        return True
    
    def store_rolling_summary(self, summary_data):
        """Store a rolling summary."""
        cursor = self.sqlite_conn.cursor()
        
        # Extract data from the summary
        summary_id = summary_data['id']
        timestamp = summary_data['timestamp']
        summary_text = summary_data['summary_text']
        conversation_range = json.dumps(summary_data['conversation_range'])
        version = summary_data.get('version', 1)
        metadata = summary_data.get('metadata', {})
        
        if isinstance(metadata, dict):
            metadata_json = json.dumps(metadata)
        else:
            metadata_json = "{}"
        
        # Store the summary
        cursor.execute("""
            INSERT INTO rolling_summaries
            (id, timestamp, summary_text, conversation_range, version, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (summary_id, timestamp, summary_text, conversation_range, version, metadata_json))
        
        # Store the embedding if provided
        embedding = summary_data.get('embedding')
        if embedding is not None:
            import numpy as np
            
            # Store in embeddings table
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings
                (id, source_id, vector_blob, model, dimensions, source_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                f"rolling_{summary_id}",
                summary_id,
                np.array(embedding, dtype=np.float32).tobytes(),
                "summary-embedding",
                len(embedding),
                "rolling_summary"
            ))
        
        self.sqlite_conn.commit()
        return True
            
    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'sqlite_conn') and self.sqlite_conn:
            self.sqlite_conn.close()