#!/bin/bash
#
# Script to generate summaries for indexed conversations
#

# Check for virtual environment and activate if possible
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    # Only source if not already in the venv
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "📦 Activating virtual environment..."
        source venv/bin/activate
    fi
elif [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    # Check for alternate venv directory
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "📦 Activating virtual environment..."
        source .venv/bin/activate
    fi
fi

# Default values
DB_PATH="memory.db"
BATCH_SIZE=20
APPLY_FORGETTING=false
DAYS_THRESHOLD=180
VERBOSE=false
REBUILD=false
CONV_SUMMARIES=false
GENERATE_EMBEDDINGS=true
EVALUATE_SUMMARIES=true
MAX_WORKERS=4

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
      export DEBUG_EVALUATION=true
      shift
      ;;
    --rebuild)
      # Delete all summaries and rebuild from scratch
      REBUILD=true
      shift
      ;;
    --conversation-summaries|--conv)
      # Generate conversation-specific summaries
      CONV_SUMMARIES=true
      shift
      ;;
    --no-embeddings)
      GENERATE_EMBEDDINGS=false
      shift
      ;;
    --no-evaluation)
      EVALUATE_SUMMARIES=false
      shift
      ;;
    --workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --embeddings)
      # Generate embeddings for existing summaries
      GENERATE_EMBEDDINGS_ONLY=true
      echo "🔍 Generating embeddings for existing summaries in $DB_PATH..."
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python -c "
from memory_ai.core.database import MemoryDatabase
from memory_ai.memory.summary import RollingSummaryProcessor
from memory_ai.utils.gemini import GeminiClient

db = MemoryDatabase('$DB_PATH')
gemini = GeminiClient()
processor = RollingSummaryProcessor(db, gemini)

# Generate embeddings for existing summaries
processed_global, processed_conv = processor.generate_embeddings_for_summaries()
print(f'Generated embeddings for {processed_global} global summaries and {processed_conv} conversation summaries')
"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      exit 0
      ;;
    --view-conversation|--view-conv)
      # View a specific conversation summary
      CONV_ID="$2"
      echo "🔍 Viewing summary for conversation $CONV_ID from $DB_PATH..."
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python -c "
from memory_ai.core.database import MemoryDatabase
from datetime import datetime
import json

db = MemoryDatabase('$DB_PATH')
summary = db.get_conversation_summary('$CONV_ID')

if summary:
    print(f\"CONVERSATION SUMMARY: {summary['conversation_id']}\\n\")
    
    # Print metadata if available
    if 'metadata' in summary and summary['metadata']:
        metadata = summary['metadata']
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                pass
                
        if isinstance(metadata, dict):
            if 'key_points' in metadata:
                print(\"KEY POINTS:\")
                for point in metadata['key_points']:
                    print(f\"- {point}\")
                print()
                
            if 'themes' in metadata:
                print(f\"THEMES: {', '.join(metadata['themes'])}\")
                
            if 'entities' in metadata:
                print(f\"ENTITIES: {', '.join(metadata['entities'][:10])}\")
                
            if 'sentiment' in metadata:
                print(f\"SENTIMENT: {metadata['sentiment']}\")
                
            if 'technical_level' in metadata:
                print(f\"TECHNICAL LEVEL: {metadata['technical_level']}\")
            print()
    
    print(summary['summary_text'])
    
    # Check if we have conversation information
    c = db.conn.cursor()
    c.execute('SELECT title, timestamp FROM conversations WHERE id = ?', (summary['conversation_id'],))
    conv_data = c.fetchone()
    
    if conv_data:
        title, timestamp = conv_data
        print(f\"\\nConversation: {title}\")
        print(f\"Date: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}\")
else:
    print('No summary found for this conversation.')
    
    # Check if conversation exists
    c = db.conn.cursor()
    c.execute('SELECT id, title FROM conversations WHERE id = ?', ('$CONV_ID',))
    conv = c.fetchone()
    
    if conv:
        print(f\"Conversation exists: {conv[1]}\")
        print(\"But no summary has been generated yet.\")
    else:
        print(f\"Conversation with ID '$CONV_ID' not found in database.\")
        
db.close()
"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      exit 0
      ;;
    --themes)
      # Search for themes and topics
      THEME_QUERY="$2"
      echo "🔍 Searching for summaries related to '$THEME_QUERY' in $DB_PATH..."
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python -c "
from memory_ai.core.database import MemoryDatabase
from memory_ai.utils.gemini import GeminiClient
from datetime import datetime

db = MemoryDatabase('$DB_PATH')
embedding = None

# Try to use vector search if possible
try:
    gemini = GeminiClient()
    embedding = gemini.generate_embedding('$THEME_QUERY')
    print('Using vector similarity search with embeddings')
except Exception as e:
    print(f'Falling back to text search: {e}')

theme_summaries = db.get_summaries_by_theme('$THEME_QUERY', embedding=embedding)

if theme_summaries:
    print(f\"Found {len(theme_summaries)} matching summaries\\n\")
    for i, summary in enumerate(theme_summaries, 1):
        print(f\"SUMMARY #{i} (Version: {summary['version']}):\")
        
        # Print relevance score
        if 'relevance' in summary:
            print(f\"Relevance: {summary['relevance']:.2f}\")
        
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
      echo "🔍 Viewing active summary from $DB_PATH..."
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python -c "
from memory_ai.core.database import MemoryDatabase
db = MemoryDatabase('$DB_PATH')
summary = db.get_active_summary()
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
    --list-conversations|--list-convs)
      # List conversations without summaries
      echo "🔍 Listing conversations without summaries in $DB_PATH..."
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python -c "
from memory_ai.core.database import MemoryDatabase
from datetime import datetime

db = MemoryDatabase('$DB_PATH')
c = db.conn.cursor()

# Get conversations without summaries
c.execute('''
SELECT id, title, timestamp, message_count 
FROM conversations
WHERE summary IS NULL OR summary = ''
ORDER BY timestamp DESC
''')

conversations = c.fetchall()
if conversations:
    print(f'Found {len(conversations)} conversations without summaries:\n')
    
    for conv in conversations:
        conv_id, title, timestamp, message_count = conv
        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        print(f'ID: {conv_id}')
        print(f'Title: {title}')
        print(f'Date: {date}')
        print(f'Messages: {message_count}')
        print('-' * 40)
else:
    print('All conversations have summaries.')

db.close()
"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      exit 0
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --db PATH                Database path (default: memory.db)"
      echo "  --batch-size N           Number of conversations to process at once (default: 20)"
      echo "  --forget                 Apply forgetting mechanism to reduce outdated information"
      echo "  --days N                 Days threshold for forgetting mechanism (default: 180)"
      echo "  --verbose                Show detailed process information"
      echo "  --debug                  Show API debug information"
      echo "  --rebuild                Delete all existing summaries and rebuild from scratch"
      echo "  --conversation-summaries Generate summaries for individual conversations"
      echo "  --conv                   Short for --conversation-summaries"
      echo "  --no-embeddings          Don't generate embeddings for summaries"
      echo "  --no-evaluation          Don't evaluate summary quality"
      echo "  --workers N              Maximum parallel workers (default: 4)"
      echo "  --limit N                Limit number of conversations to process"
      echo "  --embeddings             Generate embeddings for existing summaries"
      echo "  --themes QUERY           Search for summaries related to a specific theme or topic"
      echo "  --view                   View the active global summary and exit"
      echo "  --view-conv CONV_ID      View summary for a specific conversation"
      echo "  --list-convs             List conversations without summaries"
      echo "  -h, --help               Show this help message"
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
export MEMORY_CONV_SUMMARIES="$CONV_SUMMARIES"
export SUMMARY_MAX_WORKERS="$MAX_WORKERS"
export MEMORY_LIMIT="${LIMIT:-0}"  # 0 means no limit

# Convert boolean flags to string values for Python
if $GENERATE_EMBEDDINGS; then
  export GENERATE_SUMMARY_EMBEDDINGS="true"
else
  export GENERATE_SUMMARY_EMBEDDINGS="false"
fi

if $EVALUATE_SUMMARIES; then
  export EVALUATE_SUMMARIES="true"
else
  export EVALUATE_SUMMARIES="false"
fi

# Enable debug mode if verbose is set
if $VERBOSE; then
  export DEBUG_EVALUATION="true"
  export VERBOSE_EMBEDDINGS="true"
fi

# Use dummy embeddings to avoid API calls when running in test mode
if [[ "$DB_PATH" == *"test"* ]]; then
  export USE_DUMMY_EMBEDDINGS="true"
fi

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
    conv_summaries = os.environ['MEMORY_CONV_SUMMARIES'].lower() == 'true'
    max_workers = int(os.environ['SUMMARY_MAX_WORKERS'])
    limit = int(os.environ.get('MEMORY_LIMIT', 0))
    
    db = MemoryDatabase(db_path)
    
    # Delete all summaries if rebuilding
    if rebuild:
        print('Rebuilding summaries: deleting all existing summaries...')
        db.conn.execute('DELETE FROM rolling_summaries')
        
        # Also clear conversation summaries if doing a full rebuild
        c = db.conn.cursor()
        c.execute('UPDATE conversations SET summary = NULL, metadata = NULL')
        
        # Delete summary embeddings
        c.execute('DELETE FROM embeddings WHERE id LIKE \"summary_%\"')
        
        db.conn.commit()
        print('All existing summaries deleted. Starting fresh.')
    
    gemini = GeminiClient()
    processor = RollingSummaryProcessor(db, gemini, batch_size=batch_size)
    
    # If only generating conversation-specific summaries
    if conv_summaries and not rebuild:
        print('Processing only conversation-specific summaries...')
        try:
            count = processor.process_conversation_specific_summaries(limit=limit if limit > 0 else None)
            print(f'Generated {count} conversation-specific summaries!')
            
            # Generate embeddings for summaries if needed
            if count > 0:
                processor.generate_embeddings_for_summaries()
                
        except Exception as e:
            print(f'ERROR during conversation summary generation: {e}')
            traceback.print_exc()
    else:
        # Generate global rolling summaries
        print('Processing global rolling summaries...')
        try:
            processor.process_conversations()
            print('Global summary processing completed successfully!')
            
            # Always process conversation summaries for new batches
            unsummarized_count = db.conn.execute(
                'SELECT COUNT(*) FROM conversations WHERE summary IS NULL OR summary = \"\"'
            ).fetchone()[0]
            
            if unsummarized_count > 0:
                print(f'Found {unsummarized_count} conversations without summaries')
                limit_to_use = limit if limit > 0 else None
                count = processor.process_conversation_specific_summaries(limit=limit_to_use)
                print(f'Generated {count} conversation-specific summaries')
            
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
    
    # Get final counts
    cur = db.conn.cursor()
    cur.execute('SELECT COUNT(*) FROM rolling_summaries')
    global_count = cur.fetchone()[0]
    
    cur.execute('SELECT COUNT(*) FROM conversations WHERE summary IS NOT NULL AND summary != \"\"')
    conv_count = cur.fetchone()[0]
    
    cur.execute('SELECT COUNT(*) FROM conversations')
    total_conv = cur.fetchone()[0]
    
    print(f'Summary generation complete.')
    print(f'- Global summaries: {global_count}')
    print(f'- Conversation summaries: {conv_count}/{total_conv} ({int(conv_count/total_conv*100)}%)')
    
    db.close()
    
except ImportError as e:
    print(f'ERROR: Failed to import required modules: {e}')
except Exception as e:
    print(f'Unexpected error: {e}')
    traceback.print_exc()
"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Summary generation complete!"

# Offer to view summaries if not verbose
if ! $VERBOSE; then
  echo
  if $CONV_SUMMARIES; then
    # If we generated conversation summaries, ask to view one
    read -p "Would you like to view a specific conversation summary? (y/n) " ANSWER
    if [[ "$ANSWER" == "y" || "$ANSWER" == "Y" ]]; then
      echo
      echo "Recent conversations with summaries:"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python -c "
from memory_ai.core.database import MemoryDatabase
from datetime import datetime
db = MemoryDatabase('$DB_PATH')
c = db.conn.cursor()

# Get 5 recent conversations with summaries
c.execute('''
SELECT id, title, timestamp
FROM conversations
WHERE summary IS NOT NULL
ORDER BY timestamp DESC
LIMIT 5
''')

convs = c.fetchall()
if convs:
    for i, conv in enumerate(convs, 1):
        conv_id, title, timestamp = conv
        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        print(f\"{i}. {title} - {date} (ID: {conv_id})\")
else:
    print('No conversation summaries found.')
db.close()
"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      read -p "Enter a conversation ID to view its summary: " CONV_ID
      if [[ -n "$CONV_ID" ]]; then
        echo
        echo "🔍 Conversation summary:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python -c "
from memory_ai.core.database import MemoryDatabase
from datetime import datetime
import json

db = MemoryDatabase('$DB_PATH')
summary = db.get_conversation_summary('$CONV_ID')

if summary:
    print(f\"CONVERSATION SUMMARY: {summary['conversation_id']}\\n\")
    
    # Print metadata if available
    if 'metadata' in summary and summary['metadata']:
        metadata = summary['metadata']
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                pass
                
        if isinstance(metadata, dict):
            if 'key_points' in metadata:
                print(\"KEY POINTS:\")
                for point in metadata['key_points']:
                    print(f\"- {point}\")
                print()
                
            if 'themes' in metadata:
                print(f\"THEMES: {', '.join(metadata['themes'])}\")
                
            if 'entities' in metadata:
                print(f\"ENTITIES: {', '.join(metadata['entities'][:10])}\")
                
            if 'sentiment' in metadata:
                print(f\"SENTIMENT: {metadata['sentiment']}\")
                
            if 'technical_level' in metadata:
                print(f\"TECHNICAL LEVEL: {metadata['technical_level']}\")
            print()
    
    print(summary['summary_text'])
    
    # Check if we have conversation information
    c = db.conn.cursor()
    c.execute('SELECT title, timestamp FROM conversations WHERE id = ?', (summary['conversation_id'],))
    conv_data = c.fetchone()
    
    if conv_data:
        title, timestamp = conv_data
        print(f\"\\nConversation: {title}\")
        print(f\"Date: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}\")
else:
    print('No summary found for this conversation.')
    
    # Check if conversation exists
    c = db.conn.cursor()
    c.execute('SELECT id, title FROM conversations WHERE id = ?', ('$CONV_ID',))
    conv = c.fetchone()
    
    if conv:
        print(f\"Conversation exists: {conv[1]}\")
        print(\"But no summary has been generated yet.\")
    else:
        print(f\"Conversation with ID '$CONV_ID' not found in database.\")
        
db.close()
"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      fi
    fi
  else
    # Otherwise offer to view the global summary
    read -p "Would you like to view the active global summary? (y/n) " ANSWER
    if [[ "$ANSWER" == "y" || "$ANSWER" == "Y" ]]; then
      echo
      echo "🔍 Active global summary:"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python -c "
from memory_ai.core.database import MemoryDatabase
db = MemoryDatabase('$DB_PATH')
summary = db.get_active_summary()
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
fi