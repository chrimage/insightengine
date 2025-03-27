#!/bin/bash
#
# Script to generate summaries for indexed conversations
#

# Default values
DB_PATH="memory.db"
BATCH_SIZE=20
APPLY_FORGETTING=false
DAYS_THRESHOLD=180
VERBOSE=false
REBUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --db)
      DB_PATH="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --forget)
      APPLY_FORGETTING=true
      shift
      ;;
    --days)
      DAYS_THRESHOLD="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      export VERBOSE_EMBEDDINGS=true
      shift
      ;;
    --debug)
      export DEBUG_EMBEDDINGS=true
      shift
      ;;
    --rebuild)
      # Delete all summaries and rebuild from scratch
      REBUILD=true
      shift
      ;;
    --themes)
      # Search for themes and topics
      THEME_QUERY="$2"
      echo "🔍 Searching for summaries related to '$THEME_QUERY' in $DB_PATH..."
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python -c "
from memory_ai.core.database import MemoryDatabase
from datetime import datetime

db = MemoryDatabase('$DB_PATH')
theme_summaries = db.get_summaries_by_theme('$THEME_QUERY')

if theme_summaries:
    print(f\"Found {len(theme_summaries)} matching summaries\\n\")
    for i, summary in enumerate(theme_summaries, 1):
        print(f\"SUMMARY #{i} (Version: {summary['version']}):\")
        
        # Print metadata if available
        if 'metadata' in summary:
            if 'themes' in summary['metadata']:
                print(f\"Themes: {', '.join(summary['metadata']['themes'])}\")
            if 'topics' in summary['metadata']:
                print(f\"Topics: {', '.join(summary['metadata']['topics'][:10])}\")
            if 'quality_score' in summary['metadata']:
                print(f\"Quality Score: {summary['metadata']['quality_score']:.2f}\")
        
        # Print timestamp
        created = datetime.fromtimestamp(summary['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        print(f\"Created: {created}\")
        
        print(f\"\\n{summary['summary_text'][:1000]}...\")
        print(\"\\n\" + \"-\"*50 + \"\\n\")
else:
    print('No summaries found matching your query.')
db.close()
"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      exit 0
      ;;
    --view)
      # Just view the latest summary and exit
      echo "🔍 Viewing latest summary from $DB_PATH..."
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python -c "
from memory_ai.core.database import MemoryDatabase
db = MemoryDatabase('$DB_PATH')
summary = db.get_latest_summary()
if summary:
    print(f\"Summary version: {summary['version']}\\n\")
    print(f\"Covers {len(summary['conversation_range'])} conversations\\n\")
    
    # Print metadata if available
    if 'metadata' in summary:
        if 'themes' in summary['metadata']:
            print(f\"Themes: {', '.join(summary['metadata']['themes'])}\")
        if 'topics' in summary['metadata']:
            print(f\"Topics: {', '.join(summary['metadata']['topics'][:10])}\")
        if 'quality_score' in summary['metadata']:
            print(f\"Quality Score: {summary['metadata']['quality_score']:.2f}\")
        print()
    
    print(summary['summary_text'])
    print(f\"\\nLast updated: {__import__('datetime').datetime.fromtimestamp(summary['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\")
else:
    print('No summaries found in the database.')
db.close()
"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      exit 0
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --db PATH         Database path (default: memory.db)"
      echo "  --batch-size N    Number of conversations to process at once (default: 20)"
      echo "  --forget          Apply forgetting mechanism to reduce outdated information"
      echo "  --days N          Days threshold for forgetting mechanism (default: 180)"
      echo "  --verbose         Show detailed process information"
      echo "  --debug           Show API debug information"
      echo "  --rebuild         Delete all existing summaries and rebuild from scratch"
      echo "  --themes QUERY    Search for summaries related to a specific theme or topic"
      echo "  --view            View the latest summary and exit"
      echo "  -h, --help        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Start the process with some visual feedback
echo "🧠 Generating summaries from conversations in $DB_PATH..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Count conversations in the database
CONV_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM conversations;")
MSG_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM messages;")
SUMMARY_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM rolling_summaries;")

if $VERBOSE; then
  echo "Settings:"
  echo "  Database:       $DB_PATH"
  echo "  Batch size:     $BATCH_SIZE conversations"
  echo "  Rebuild:        $(if $REBUILD; then echo "Yes (deleting existing summaries)"; else echo "No"; fi)"
  echo "  Forgetting:     $(if $APPLY_FORGETTING; then echo "Enabled (${DAYS_THRESHOLD} days)"; else echo "Disabled"; fi)"
  echo
fi

echo "Database Stats:"
echo "  📚 Conversations: $CONV_COUNT"
echo "  💬 Messages: $MSG_COUNT"
echo "  📝 Summaries: $SUMMARY_COUNT"
echo

# Export settings as environment variables for Python
export MEMORY_DB_PATH="$DB_PATH"
export MEMORY_BATCH_SIZE="$BATCH_SIZE"
export MEMORY_REBUILD="$REBUILD"
export MEMORY_APPLY_FORGETTING="$APPLY_FORGETTING"
export MEMORY_DAYS_THRESHOLD="$DAYS_THRESHOLD"

# Run the summarization with better error catching
python -c "
import sys
import traceback
import os

try:
    from memory_ai.core.database import MemoryDatabase
    from memory_ai.memory.summary import RollingSummaryProcessor
    from memory_ai.utils.gemini import GeminiClient
    
    # Get settings from environment variables
    db_path = os.environ['MEMORY_DB_PATH']
    batch_size = int(os.environ['MEMORY_BATCH_SIZE'])
    rebuild = os.environ['MEMORY_REBUILD'].lower() == 'true'
    apply_forgetting = os.environ['MEMORY_APPLY_FORGETTING'].lower() == 'true'
    days_threshold = int(os.environ['MEMORY_DAYS_THRESHOLD'])
    
    db = MemoryDatabase(db_path)
    
    # Delete all summaries if rebuilding
    if rebuild:
        print('Rebuilding summaries: deleting all existing summaries...')
        db.conn.execute('DELETE FROM rolling_summaries')
        db.conn.commit()
        print('All existing summaries deleted. Starting fresh.')
    
    gemini = GeminiClient()
    processor = RollingSummaryProcessor(db, gemini, batch_size=batch_size)
    
    print('Processing conversations...')
    try:
        processor.process_conversations()
        print('Processing completed successfully!')
    except Exception as e:
        print(f'ERROR during summary generation: {e}')
        traceback.print_exc()
        
    # Apply forgetting if requested
    if apply_forgetting:
        print(f'Applying forgetting mechanism (threshold: {days_threshold} days)')
        try:
            updated_summary = processor.implement_forgetting(days_threshold=days_threshold)
            if updated_summary:
                print(f'Forgetting mechanism applied. New summary version: {updated_summary[\"version\"]}')
            else:
                print('No summaries available to apply forgetting mechanism.')
        except Exception as e:
            print(f'ERROR during forgetting mechanism: {e}')
            traceback.print_exc()
    
    # Get final count of summaries
    cur = db.conn.cursor()
    cur.execute('SELECT COUNT(*) FROM rolling_summaries')
    final_count = cur.fetchone()[0]
    print(f'Summary generation complete. Database now contains {final_count} summaries.')
    
    db.close()
    
except ImportError as e:
    print(f'ERROR: Failed to import required modules: {e}')
except Exception as e:
    print(f'Unexpected error: {e}')
    traceback.print_exc()
"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Summary generation complete!"

# Offer to view the summary if not verbose
if ! $VERBOSE; then
  echo
  read -p "Would you like to view the latest summary? (y/n) " ANSWER
  if [[ "$ANSWER" == "y" || "$ANSWER" == "Y" ]]; then
    echo
    echo "🔍 Latest summary:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python -c "
from memory_ai.core.database import MemoryDatabase
db = MemoryDatabase('$DB_PATH')
summary = db.get_latest_summary()
if summary:
    print(f\"Summary version: {summary['version']}\\n\")
    print(f\"Covers {len(summary['conversation_range'])} conversations\\n\")
    
    # Print metadata if available
    if 'metadata' in summary:
        if 'themes' in summary['metadata']:
            print(f\"Themes: {', '.join(summary['metadata']['themes'])}\")
        if 'topics' in summary['metadata']:
            print(f\"Topics: {', '.join(summary['metadata']['topics'][:10])}\")
        if 'quality_score' in summary['metadata']:
            print(f\"Quality Score: {summary['metadata']['quality_score']:.2f}\")
        print()
    
    print(summary['summary_text'])
    print(f\"\\nLast updated: {__import__('datetime').datetime.fromtimestamp(summary['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\")
else:
    print('No summaries found in the database.')
db.close()
"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  fi
fi