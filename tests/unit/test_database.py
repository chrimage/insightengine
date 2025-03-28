"""
Unit tests for database operations.
"""

import pytest
import json
from datetime import datetime

from memory_ai.core.database import UnitOfWork
from memory_ai.core.models import Message, Conversation, Chunk, Role


class TestDatabaseOperations:
    """Tests for database operations."""
    
    def test_conversation_repository(self, test_db):
        """Test conversation repository operations."""
        # Create a conversation
        conversation = Conversation(
            title="Test Conversation",
            model="test-model",
            summary="Test summary",
            quality_score=0.8,
            token_count=100,
            metadata={"source": "test"}
        )
        
        # Save the conversation
        with test_db.get_unit_of_work() as uow:
            conv_id = uow.conversations.add(conversation)
            uow.commit()
            
            # Verify it was saved
            assert conv_id == conversation.id
            
            # Retrieve the conversation
            retrieved = uow.conversations.get(conv_id)
            assert retrieved is not None
            assert retrieved.id == conversation.id
            assert retrieved.title == conversation.title
            assert retrieved.model == conversation.model
            assert retrieved.summary == conversation.summary
            assert retrieved.quality_score == conversation.quality_score
            assert retrieved.token_count == conversation.token_count
            assert retrieved.metadata == conversation.metadata
    
    def test_message_repository(self, test_db):
        """Test message repository operations."""
        # Create a conversation first
        conversation = Conversation(title="Test Conversation")
        
        with test_db.get_unit_of_work() as uow:
            conv_id = uow.conversations.add(conversation)
            uow.commit()
        
        # Create a message
        message = Message(
            role=Role.USER,
            content="Hello, world!",
            conversation_id=conv_id,
            metadata={"source": "test"}
        )
        
        # Save the message
        with test_db.get_unit_of_work() as uow:
            msg_id = uow.messages.add(message)
            uow.commit()
            
            # Verify it was saved
            assert msg_id == message.id
            
            # Retrieve the message
            retrieved = uow.messages.get(msg_id)
            assert retrieved is not None
            assert retrieved.id == message.id
            assert retrieved.role == message.role
            assert retrieved.content == message.content
            assert retrieved.conversation_id == conv_id
            assert retrieved.metadata == message.metadata
            
            # Retrieve by conversation ID
            messages = uow.messages.get_by_conversation(conv_id)
            assert len(messages) == 1
            assert messages[0].id == message.id
    
    def test_chunk_repository(self, test_db):
        """Test chunk repository operations."""
        # Create a conversation first
        conversation = Conversation(title="Test Conversation")
        
        with test_db.get_unit_of_work() as uow:
            conv_id = uow.conversations.add(conversation)
            
            # Create messages
            message1 = Message(
                role=Role.USER,
                content="Message 1",
                conversation_id=conv_id
            )
            
            message2 = Message(
                role=Role.ASSISTANT,
                content="Message 2",
                conversation_id=conv_id
            )
            
            msg1_id = uow.messages.add(message1)
            msg2_id = uow.messages.add(message2)
            
            uow.commit()
        
        # Create a chunk
        chunk = Chunk(
            conversation_id=conv_id,
            message_ids=[msg1_id, msg2_id],
            content="Message 1\nMessage 2",
            metadata={"type": "test"}
        )
        
        # Save the chunk
        with test_db.get_unit_of_work() as uow:
            chunk_id = uow.chunks.add(chunk)
            uow.commit()
            
            # Verify it was saved
            assert chunk_id == chunk.id
            
            # Retrieve the chunk
            retrieved = uow.chunks.get(chunk_id)
            assert retrieved is not None
            assert retrieved.id == chunk.id
            assert retrieved.conversation_id == conv_id
            assert retrieved.message_ids == [msg1_id, msg2_id]
            assert retrieved.content == "Message 1\nMessage 2"
            assert retrieved.metadata == {"type": "test"}
            
            # Retrieve chunks by conversation ID
            chunks = uow.chunks.get_by_conversation(conv_id)
            assert len(chunks) == 1
            assert chunks[0].id == chunk.id
    
    def test_transaction_rollback(self, test_db):
        """Test transaction rollback."""
        # Create a conversation
        conversation = Conversation(title="Test Conversation")
        
        # Start a transaction
        with test_db.get_unit_of_work() as uow:
            conv_id = uow.conversations.add(conversation)
            
            # Create a message
            message = Message(
                role=Role.USER,
                content="Test message",
                conversation_id=conv_id
            )
            
            msg_id = uow.messages.add(message)
            
            # Explicitly rollback
            uow.rollback()
        
        # Verify that nothing was saved
        with test_db.get_unit_of_work() as uow:
            # Conversation should not exist
            assert uow.conversations.get(conv_id) is None
            
            # Message should not exist
            assert uow.messages.get(msg_id) is None
    
    def test_conversation_retrieval_by_time(self, test_db):
        """Test conversation retrieval by time range."""
        # Create conversations with different timestamps
        timestamps = [
            datetime(2023, 1, 1).timestamp(),
            datetime(2023, 1, 15).timestamp(),
            datetime(2023, 2, 1).timestamp(),
        ]
        
        conv_ids = []
        with test_db.get_unit_of_work() as uow:
            for i, ts in enumerate(timestamps):
                conv = Conversation(
                    title=f"Conversation {i+1}",
                    timestamp=ts
                )
                conv_id = uow.conversations.add(conv)
                conv_ids.append(conv_id)
            uow.commit()
        
        # Retrieve conversations by time range
        with test_db.get_unit_of_work() as uow:
            # All conversations
            start_time = datetime(2022, 12, 1).timestamp()
            end_time = datetime(2023, 3, 1).timestamp()
            all_convs = uow.conversations.get_by_time_range(start_time, end_time)
            assert len(all_convs) == 3
            
            # Only January conversations
            start_time = datetime(2023, 1, 1).timestamp()
            end_time = datetime(2023, 1, 31).timestamp()
            jan_convs = uow.conversations.get_by_time_range(start_time, end_time)
            assert len(jan_convs) == 2
            assert jan_convs[0].id == conv_ids[0]
            assert jan_convs[1].id == conv_ids[1]
            
            # Only February conversations
            start_time = datetime(2023, 2, 1).timestamp()
            end_time = datetime(2023, 2, 28).timestamp()
            feb_convs = uow.conversations.get_by_time_range(start_time, end_time)
            assert len(feb_convs) == 1
            assert feb_convs[0].id == conv_ids[2]