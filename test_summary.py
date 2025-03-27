#!/usr/bin/env python3
# Test script for the summary functionality

import os
from memory_ai.core.database import MemoryDatabase
from memory_ai.memory.summary import RollingSummaryProcessor
from memory_ai.utils.gemini import GeminiClient

# Set environment variables
os.environ['GENERATE_SUMMARY_EMBEDDINGS'] = 'false'
os.environ['EVALUATE_SUMMARIES'] = 'false'
os.environ['SUMMARY_MAX_WORKERS'] = '1'
os.environ['USE_DUMMY_EMBEDDINGS'] = 'true'  # Use dummy embeddings to avoid API calls

# Initialize components
db_path = "memory.db"
print(f"Using database: {db_path}")

db = MemoryDatabase(db_path)
gemini = GeminiClient(use_dummy_embeddings=True)
processor = RollingSummaryProcessor(db, gemini, batch_size=3)

# Get conversation count
with db.conn as conn:
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM conversations")
    count = c.fetchone()[0]
    print(f"Found {count} conversations in database")

    # Get unsummarized conversation count
    c.execute("SELECT COUNT(*) FROM conversations WHERE summary IS NULL OR summary = ''")
    unsummarized = c.fetchone()[0]
    print(f"Found {unsummarized} conversations without summaries")

    # Get rolling summary count
    c.execute("SELECT COUNT(*) FROM rolling_summaries")
    summary_count = c.fetchone()[0]
    print(f"Found {summary_count} rolling summaries")

# Process a limited number of conversations
limit = 5
print(f"Processing up to {limit} conversations...")

# Test conversation-specific summaries
print("Generating conversation-specific summaries...")
processed = processor.process_conversation_specific_summaries(limit=limit)
print(f"Processed {processed} conversation summaries")

# Check conversation summaries
with db.conn as conn:
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM conversations WHERE summary IS NOT NULL AND summary != ''")
    summarized = c.fetchone()[0]
    print(f"Database now has {summarized} conversation-specific summaries")

print("Test complete!")