"""
Domain models with validation using Pydantic.

This module defines the core domain entities for the InsightEngine system.
Each model includes validation rules and relationships.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import uuid


class SourceType(str, Enum):
    """Type of content source for embeddings and retrieval."""
    
    MESSAGE = "message"
    CHUNK = "chunk"
    CONVERSATION_SUMMARY = "conv_summary" 
    ROLLING_SUMMARY = "rolling_summary"
    INSIGHT = "insight"


class Role(str, Enum):
    """Participant roles in a conversation."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageBase(BaseModel):
    """Base class for message data."""
    
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4()}")
    role: Role
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    
    class Config:
        arbitrary_types_allowed = True


class Message(MessageBase):
    """A single message in a conversation."""
    
    conversation_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationBase(BaseModel):
    """Base class for conversation data."""
    
    id: str = Field(default_factory=lambda: f"conv_{uuid.uuid4()}")
    title: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    model: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class Conversation(ConversationBase):
    """A conversation with messages."""
    
    messages: List[Message] = Field(default_factory=list)
    summary: Optional[str] = None
    quality_score: Optional[float] = None
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('quality_score')
    def validate_quality_score(cls, value):
        """Validate quality score is between 0 and 1."""
        if value is not None and (value < 0 or value > 1):
            raise ValueError("Quality score must be between 0 and 1")
        return value


class ChunkBase(BaseModel):
    """Base class for content chunks."""
    
    id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4()}")
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    
    class Config:
        arbitrary_types_allowed = True


class Chunk(ChunkBase):
    """A chunk of related messages for efficient retrieval."""
    
    conversation_id: str
    message_ids: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingBase(BaseModel):
    """Base class for vector embeddings."""
    
    id: str = Field(default_factory=lambda: f"emb_{uuid.uuid4()}")
    model: str
    dimensions: int
    
    class Config:
        arbitrary_types_allowed = True


class Embedding(EmbeddingBase):
    """Vector embedding for similarity search."""
    
    source_id: str
    source_type: SourceType
    vector: List[float]


class SummaryBase(BaseModel):
    """Base class for summaries."""
    
    id: str = Field(default_factory=lambda: f"summary_{uuid.uuid4()}")
    summary_text: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    
    class Config:
        arbitrary_types_allowed = True


class RollingSummary(SummaryBase):
    """A summary of multiple conversations over time."""
    
    conversation_ids: List[str]
    version: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InsightCategory(str, Enum):
    """Categories for system insights."""
    
    USER_PREFERENCE = "user_preference"
    INTERACTION_PATTERN = "interaction_pattern"
    KNOWLEDGE_AREA = "knowledge_area"
    RESPONSE_STRATEGY = "response_strategy"
    GENERAL = "general"


class InsightBase(BaseModel):
    """Base class for system insights."""
    
    id: str = Field(default_factory=lambda: f"insight_{uuid.uuid4()}")
    text: str
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    
    class Config:
        arbitrary_types_allowed = True


class Insight(InsightBase):
    """An insight extracted from conversation patterns and evaluations."""
    
    category: InsightCategory
    confidence: float = Field(ge=0.0, le=1.0)
    updated_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    evidence: List[str] = Field(default_factory=list)
    application_count: int = 0
    success_rate: Optional[float] = None
    applications: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationBase(BaseModel):
    """Base class for response evaluations."""
    
    id: str = Field(default_factory=lambda: f"eval_{uuid.uuid4()}")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    
    class Config:
        arbitrary_types_allowed = True


class Evaluation(EvaluationBase):
    """An evaluation of a system response."""
    
    query: str
    evaluation_data: Dict[str, Any]
    structured: bool = True
    meta_evaluation: Optional[Dict[str, Any]] = None