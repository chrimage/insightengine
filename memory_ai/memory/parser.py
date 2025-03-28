# Conversation parsing utilities

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from memory_ai.core.models import Conversation, Message

class OpenAIParser:
    """Parser for OpenAI conversation exports."""
    
    def parse(self, data: Dict[str, Any]) -> Conversation:
        """Parse OpenAI conversation export data into a Conversation object."""
        # Check if data is a list (conversations.json is an array of conversations)
        if isinstance(data, list):
            # If it's a list, just use the first item or create an empty conversation
            if data and isinstance(data[0], dict):
                data = data[0]
            else:
                # Create a minimal conversation if data is an empty list
                return Conversation(
                    id=str(uuid.uuid4()),
                    title="Empty Conversation",
                    messages=[],
                    model="unknown"
                )
        
        # Extract basic conversation info
        conv_id = data.get('id', str(uuid.uuid4()))
        title = data.get('title', 'Untitled Conversation')
        
        # Extract timestamp - all timestamps are floats in the observed data
        create_time = data.get('create_time')
        if create_time:
            try:
                # Convert to datetime from timestamp
                timestamp = datetime.fromtimestamp(float(create_time))
            except (ValueError, TypeError):
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()
        
        # Extract model information
        model = data.get('model_slug', '')
        # If model is missing or empty, use default_model_slug as fallback
        if not model:
            model = data.get('default_model_slug', 'unknown')
        
        # Extract conversation thread
        messages = self._extract_conversation_thread(data)
        
        # Create and return conversation object
        return Conversation(
            id=conv_id,
            title=title,
            messages=messages,
            model=model,
            timestamp=timestamp.timestamp() if isinstance(timestamp, datetime) else timestamp
        )
    
    def _extract_conversation_thread(self, conversation: Dict[str, Any]) -> List[Message]:
        """Extract messages from a conversation in the correct order."""
        # Get conversation mapping
        mapping = conversation.get('mapping', {})
        if not mapping:
            return []
        
        # Get conversation ID
        conv_id = conversation.get('id', str(uuid.uuid4()))
        
        # Start from the current node
        current_node_id = conversation.get('current_node')
        if not current_node_id or current_node_id not in mapping:
            # If we can't find the current node, let's try to find a root node
            for node_id, node in mapping.items():
                if not node.get('parent'):
                    current_node_id = node_id
                    break
            
            # If still no node found, return empty list
            if not current_node_id or current_node_id not in mapping:
                return []
        
        # Traverse up to find the root node
        node_id = current_node_id
        while True:
            node = mapping.get(node_id)
            if not node:
                break
            
            parent_id = node.get('parent')
            if not parent_id or parent_id not in mapping:
                break
                
            node_id = parent_id
        
        # Root node found, now traverse down to build the thread
        thread = []
        visited = set()
        
        def traverse(node_id):
            if not node_id or node_id in visited or node_id not in mapping:
                return
                
            visited.add(node_id)
            node = mapping.get(node_id)
            
            # Add message to thread if it exists
            if 'message' in node and node['message'] is not None:
                message = node['message']
                
                # Extract role
                author = message.get('author', {})
                role = author.get('role', 'system') if author else 'system'
                
                # Map any non-standard roles to system
                if role not in ['user', 'assistant', 'system']:
                    role = 'system'
                
                # Extract content - handle different content types
                content = ''
                content_parts = message.get('content', {})
                
                if isinstance(content_parts, dict):
                    # Standard format: extract from parts array
                    parts = content_parts.get('parts', [])
                    if parts and len(parts) > 0:
                        # Handle different part types (text, image, etc.)
                        first_part = parts[0]
                        if isinstance(first_part, dict) and 'url' in first_part:
                            # Image or other media
                            content = f"[Image or media: {first_part.get('url', '')}]"
                        else:
                            # Regular text
                            content = str(first_part) if first_part is not None else ''
                elif isinstance(content_parts, str):
                    # Direct string content
                    content = content_parts
                
                # Extract timestamp - also floats in messages
                msg_timestamp = None
                create_time = message.get('create_time')
                if create_time:
                    try:
                        msg_timestamp = datetime.fromtimestamp(float(create_time))
                    except (ValueError, TypeError):
                        msg_timestamp = datetime.now()
                else:
                    msg_timestamp = datetime.now()
                
                # Create message object
                if content:  # Only add non-empty messages
                    thread.append(Message(
                        role=role, 
                        content=content, 
                        timestamp=msg_timestamp.timestamp() if msg_timestamp else datetime.now().timestamp(),
                        conversation_id=conv_id
                    ))
            
            # Process children in order
            for child_id in node.get('children', []):
                traverse(child_id)
        
        # Start traversal from the root node
        traverse(node_id)
        
        return thread
