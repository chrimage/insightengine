"""
Unit tests for domain models.
"""

import pytest
from datetime import datetime, timedelta
import uuid
from pydantic import ValidationError

from memory_ai.core.models import (
    Message, Conversation, Chunk, Embedding, RollingSummary, Insight,
    Role, SourceType, InsightCategory
)


class TestMessageModel:
    """Tests for the Message model."""
    
    def test_message_creation_valid(self):
        """Test valid message creation."""
        # Create a message with required fields
        message = Message(
            role=Role.USER,
            content="Hello, world!",
            conversation_id="conv_123"
        )
        
        # Check that id was auto-generated
        assert message.id.startswith("msg_")
        assert isinstance(uuid.UUID(message.id[4:]), uuid.UUID)
        
        # Check other fields
        assert message.role == Role.USER
        assert message.content == "Hello, world!"
        assert message.conversation_id == "conv_123"
        assert isinstance(message.timestamp, float)
        assert message.metadata == {}
    
    def test_message_creation_with_all_fields(self):
        """Test message creation with all fields specified."""
        now = datetime.now().timestamp()
        message = Message(
            id="msg_custom",
            role=Role.ASSISTANT,
            content="I'm an assistant",
            conversation_id="conv_123",
            timestamp=now,
            metadata={"source": "test"}
        )
        
        assert message.id == "msg_custom"
        assert message.role == Role.ASSISTANT
        assert message.content == "I'm an assistant"
        assert message.conversation_id == "conv_123"
        assert message.timestamp == now
        assert message.metadata == {"source": "test"}
    
    def test_message_missing_required_fields(self):
        """Test message creation with missing required fields."""
        # Missing role
        with pytest.raises(ValidationError):
            Message(content="Hello", conversation_id="conv_123")
        
        # Missing content
        with pytest.raises(ValidationError):
            Message(role=Role.USER, conversation_id="conv_123")
        
        # Missing conversation_id
        with pytest.raises(ValidationError):
            Message(role=Role.USER, content="Hello")


class TestConversationModel:
    """Tests for the Conversation model."""
    
    def test_conversation_creation_valid(self):
        """Test valid conversation creation."""
        # Create a conversation with required fields
        conversation = Conversation(
            title="Test Conversation"
        )
        
        # Check that id was auto-generated
        assert conversation.id.startswith("conv_")
        assert isinstance(uuid.UUID(conversation.id[5:]), uuid.UUID)
        
        # Check other fields
        assert conversation.title == "Test Conversation"
        assert isinstance(conversation.timestamp, float)
        assert conversation.model is None
        assert conversation.messages == []
        assert conversation.summary is None
        assert conversation.quality_score is None
        assert conversation.token_count is None
        assert conversation.metadata == {}
    
    def test_conversation_with_messages(self):
        """Test conversation with messages."""
        # Create messages
        message1 = Message(
            role=Role.USER,
            content="Hello",
            conversation_id="conv_temp"
        )
        
        message2 = Message(
            role=Role.ASSISTANT,
            content="Hi there!",
            conversation_id="conv_temp"
        )
        
        # Create conversation with messages
        conversation = Conversation(
            title="Test Conversation",
            messages=[message1, message2]
        )
        
        # Update message conversation_ids to match the conversation
        for msg in conversation.messages:
            msg.conversation_id = conversation.id
        
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == Role.USER
        assert conversation.messages[0].content == "Hello"
        assert conversation.messages[0].conversation_id == conversation.id
        
        assert conversation.messages[1].role == Role.ASSISTANT
        assert conversation.messages[1].content == "Hi there!"
        assert conversation.messages[1].conversation_id == conversation.id
    
    def test_conversation_quality_score_validation(self):
        """Test quality score validation."""
        # Valid quality scores
        Conversation(title="Test", quality_score=0.0)
        Conversation(title="Test", quality_score=0.5)
        Conversation(title="Test", quality_score=1.0)
        
        # Invalid quality scores
        with pytest.raises(ValidationError):
            Conversation(title="Test", quality_score=-0.1)
            
        with pytest.raises(ValidationError):
            Conversation(title="Test", quality_score=1.1)


class TestInsightModel:
    """Tests for the Insight model."""
    
    def test_insight_creation(self):
        """Test valid insight creation."""
        insight = Insight(
            text="User prefers detailed explanations",
            category=InsightCategory.USER_PREFERENCE,
            confidence=0.8,
            evidence=["conv_123", "conv_456"]
        )
        
        assert insight.id.startswith("insight_")
        assert insight.text == "User prefers detailed explanations"
        assert insight.category == InsightCategory.USER_PREFERENCE
        assert insight.confidence == 0.8
        assert insight.evidence == ["conv_123", "conv_456"]
        assert insight.application_count == 0
        assert insight.success_rate is None
        assert insight.applications == []
        assert isinstance(insight.created_at, float)
        assert isinstance(insight.updated_at, float)
    
    def test_insight_confidence_validation(self):
        """Test confidence validation."""
        # Valid confidence values
        Insight(
            text="Test", 
            category=InsightCategory.GENERAL,
            confidence=0.0
        )
        
        Insight(
            text="Test", 
            category=InsightCategory.GENERAL,
            confidence=1.0
        )
        
        # Invalid confidence values
        with pytest.raises(ValidationError):
            Insight(
                text="Test", 
                category=InsightCategory.GENERAL,
                confidence=-0.1
            )
            
        with pytest.raises(ValidationError):
            Insight(
                text="Test", 
                category=InsightCategory.GENERAL,
                confidence=1.1
            )