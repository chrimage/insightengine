#!/bin/bash
#
# Script to launch interactive chat with the memory-enhanced AI system
#

# Default values
DB_PATH="memory.db"
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --db)
      DB_PATH="$2"
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
    --summary)
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
      echo "  --verbose         Show detailed process information"
      echo "  --debug           Show API debug information"
      echo "  --summary         Just view the latest summary and exit"
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
echo "🧠 Launching memory-enhanced chat system..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if $VERBOSE; then
  echo "Settings:"
  echo "  Database: $DB_PATH"
  echo
fi

# Count summaries and conversations
SUMMARY_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM rolling_summaries;")
CONV_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM conversations;")
MSG_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM messages;")

echo "Database stats:"
echo "  📚 Conversations: $CONV_COUNT"
echo "  💬 Messages: $MSG_COUNT"
echo "  📝 Summaries: $SUMMARY_COUNT"
echo

# Run the interactive chat with better error handling
python -m memory_ai.tools.interact --db "$DB_PATH"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Chat session ended!"