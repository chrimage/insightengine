# Tool for generating summaries

import argparse
from memory_ai.core.database import MemoryDatabase
from memory_ai.memory.summary import RollingSummaryProcessor
from memory_ai.utils.gemini import GeminiClient

def generate_summaries(db_path, batch_size=10, apply_forgetting=False, days_threshold=180):
    """Generate rolling summaries for conversations."""
    print(f"Generating rolling summaries for conversations in {db_path}")
    
    # Initialize components
    db = MemoryDatabase(db_path)
    gemini = GeminiClient()
    processor = RollingSummaryProcessor(db, gemini, batch_size=batch_size)
    
    # Process conversations
    processor.process_conversations()
    
    # Apply forgetting if requested
    if apply_forgetting:
        print(f"Applying forgetting mechanism (threshold: {days_threshold} days)")
        updated_summary = processor.implement_forgetting(days_threshold=days_threshold)
        if updated_summary:
            print(f"Forgetting mechanism applied. New summary version: {updated_summary['version']}")
        else:
            print("No summaries available to apply forgetting mechanism.")
    
    # Close database
    db.close()
    print("Summary generation complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rolling summaries")
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--batch-size", type=int, default=10, help="Conversation batch size")
    parser.add_argument("--apply-forgetting", action="store_true", help="Apply forgetting mechanism")
    parser.add_argument("--days-threshold", type=int, default=180, help="Days threshold for forgetting")
    
    args = parser.parse_args()
    
    generate_summaries(args.db, args.batch_size, args.apply_forgetting, args.days_threshold)