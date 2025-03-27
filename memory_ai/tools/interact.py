# Interactive chat interface

import argparse
import os
import time
import json
from datetime import datetime

from memory_ai.core.database import MemoryDatabase
from memory_ai.memory.retriever import MemoryRetriever
from memory_ai.memory.context import ContextAssembler
from memory_ai.memory.quality import MemoryQualityAssessor
from memory_ai.reflection.insights import InsightRepository
from memory_ai.reflection.evaluator import ResponseEvaluator
from memory_ai.chat.agent import MemoryEnhancedAgent
from memory_ai.utils.gemini import GeminiClient

def start_interactive_chat(db_path):
    """Start an interactive chat session with memory enhancement."""
    print(f"Initializing memory-enhanced chat system using {db_path}")
    
    # Initialize components
    db = MemoryDatabase(db_path)
    gemini = GeminiClient()
    
    # Set up repositories
    insight_repo = InsightRepository(db)
    
    # Set up memory components
    retriever = MemoryRetriever(db, gemini)
    quality_assessor = MemoryQualityAssessor(db, gemini)
    retriever.set_quality_assessor(quality_assessor)
    
    # Set up evaluator
    evaluator = ResponseEvaluator(db, gemini, insight_repo)
    
    # Set up context assembler
    context_assembler = ContextAssembler(db, gemini, retriever, insight_repo)
    
    # Set up chat agent
    agent = MemoryEnhancedAgent(db, gemini, context_assembler, evaluator)
    
    print("\n==================================================")
    print("Memory-Enhanced Chat System")
    print("==================================================")
    print("- Type 'exit' to quit")
    print("- Type 'clear' to clear conversation history")
    print("- Type 'save' to save the conversation")
    print("==================================================\n")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Handle special commands
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            agent.clear_history()
            print("Conversation history cleared.")
            continue
        elif user_input.lower() == 'save':
            save_conversation(agent.conversation_history)
            continue
        
        # Generate response
        start_time = time.time()
        response = agent.chat(user_input)
        duration = time.time() - start_time
        
        print(f"\nAssistant: {response}")
        print(f"\n[Response time: {duration:.2f}s]")
    
    # Close database
    db.close()
    print("Chat session ended.")

def save_conversation(conversation_history):
    """Save the conversation history to a file."""
    if not conversation_history:
        print("No conversation to save.")
        return
    
    # Create a filename based on timestamp and first message
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_msg = conversation_history[0]["content"][:20].replace(" ", "_")
    filename = f"conversation_{timestamp}_{first_msg}.json"
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "messages": conversation_history
        }, f, indent=2)
    
    print(f"Conversation saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive memory-enhanced chat")
    parser.add_argument("--db", required=True, help="Database path")
    
    args = parser.parse_args()
    
    start_interactive_chat(args.db)