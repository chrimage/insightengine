# Data models for system

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class Message:
    """A message in a conversation."""
    role: str
    content: str
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Conversation:
    """A conversation consisting of messages."""
    id: str
    title: str
    messages: List[Message]
    model: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Summary:
    """A summary of multiple conversations."""
    id: str
    summary_text: str
    conversation_range: List[str]
    version: int
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Insight:
    """An insight derived from conversations."""
    id: str
    text: str
    category: str
    confidence: float
    evidence: List[dict]
    application_count: int = 0
    success_rate: float = 0.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        now = datetime.now()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now