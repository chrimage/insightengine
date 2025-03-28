# Tool for indexing conversations

import argparse
import os
import json
from tqdm import tqdm
import uuid
from datetime import datetime
from memory_ai.core.database import MemoryDatabase, ConversationORM, MessageORM
from memory_ai.memory.parser import OpenAIParser
from memory_ai.utils.gemini import GeminiClient

def index_conversations(input_dir, db_path, max_conversations=None):
    """Index OpenAI conversations into the memory system."""
    print(f"Indexing conversations from {input_dir} into {db_path}")
    
    # Initialize components
    db = MemoryDatabase(db_path)
    parser = OpenAIParser()
    gemini = GeminiClient()
    
    # Only look for the main conversations.json file
    conversations_file = os.path.join(input_dir, 'conversations.json')
    json_files = [conversations_file] if os.path.exists(conversations_file) else []
    
    print(f"Found {len(json_files)} JSON files")
    
    # Limit if specified
    if max_conversations and max_conversations < len(json_files):
        json_files = json_files[:max_conversations]
        print(f"Limiting to {max_conversations} files")
    
    # Process each file
    for file_path in tqdm(json_files, desc="Processing conversations"):
        try:
            # Load and parse the conversation file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # If this is conversations.json, it contains multiple conversations
            if os.path.basename(file_path) == 'conversations.json' and isinstance(data, list):
                print(f"Processing {len(data)} conversations from conversations.json")
                # Process each conversation in the file
                # Filter out null or invalid conversations
                valid_conversations = [c for c in data if isinstance(c, dict) and c.get('mapping')]
                print(f"Found {len(valid_conversations)} valid conversations out of {len(data)}")
                
                for i, conv_data in enumerate(valid_conversations[:max_conversations] if max_conversations else valid_conversations):
                    try:
                        conversation = parser.parse(conv_data)
                        store_conversation(db, conversation, gemini)
                        # Display a cleaner, more informative summary
                        first_msg = conversation.messages[0].content if conversation.messages else "No messages"
                        preview = first_msg[:100] + "..." if len(first_msg) > 100 else first_msg
                        print(f"  Indexed [{i+1}/{len(valid_conversations)}] \"{conversation.title}\" ({len(conversation.messages)} messages)")
                        print(f"  First message: {preview}")
                        print("  " + "-"*50)
                    except Exception as e:
                        print(f"  Error processing conversation {i+1}/{len(valid_conversations)}: {str(e)}")
                # Skip the regular processing
                continue
            
            # For other files, process as a single conversation
            conversation = parser.parse(data)
            
            # Store in database
            store_conversation(db, conversation, gemini)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Close database
    db.close()
    print(f"Indexed {len(json_files)} conversations")

def store_conversation(db, conversation, gemini):
    """Store a conversation and its embeddings in the database."""
    # Store conversation using SQLAlchemy sessionmaker
    with db.manager.session() as session:
        # Create conversation ORM object
        conv_orm = ConversationORM(
            id=conversation.id,
            title=conversation.title,
            timestamp=conversation.timestamp,
            model=conversation.model or "unknown",
            message_count=len(conversation.messages),
            summary="",  # Summary will be generated later
            quality_score=None,  # Quality score to be determined
            token_count=0,  # Token count to be calculated
            metadata_json="{}"
        )
        
        # Add conversation to session
        session.merge(conv_orm)
        
        # Store messages
        for message in conversation.messages:
            message_id = str(uuid.uuid4())
            msg_orm = MessageORM(
                id=message_id,
                conversation_id=conversation.id,
                role=message.role,
                content=message.content,
                timestamp=message.timestamp,
                metadata_json="{}"
            )
            session.merge(msg_orm)
        
        # Commit to store conversation and messages
        session.commit()
    
    # Generate and store embedding
    try:
        # Combine messages for embedding
        content = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in conversation.messages])
        
        # Generate embedding
        embedding = gemini.generate_embeddings(content)
        
        # Store embedding
        db.store_embedding(conversation.id, embedding)
        
    except Exception as e:
        print(f"Error generating embedding for conversation {conversation.id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index conversations for the memory system")
    parser.add_argument("--input", required=True, help="Input directory with conversation files")
    parser.add_argument("--db", default="memory.db", help="Output database path")
    parser.add_argument("--max", type=int, help="Maximum number of conversations to process")
    
    args = parser.parse_args()
    
    index_conversations(args.input, args.db, args.max)
