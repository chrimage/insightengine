#!/bin/bash
#
# Script to index conversations with cleaner, prettier output
#

# Default values
INPUT_DIR="openai-conversations"
DB_PATH="memory.db"
MAX_CONVERSATIONS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)
      INPUT_DIR="$2"
      shift 2
      ;;
    --db)
      DB_PATH="$2"
      shift 2
      ;;
    --max)
      MAX_CONVERSATIONS="$2"
      shift 2
      ;;
    --verbose)
      export VERBOSE_EMBEDDINGS=true
      shift
      ;;
    --debug)
      export DEBUG_EMBEDDINGS=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --input DIR       Input directory with conversation files (default: openai-conversations)"
      echo "  --db PATH         Output database path (default: memory.db)"
      echo "  --max NUMBER      Maximum number of conversations to process"
      echo "  --verbose         Show detailed embedding process information"
      echo "  --debug           Show API response debug information"
      echo "  -h, --help        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Format the max conversations argument if provided
if [ -n "$MAX_CONVERSATIONS" ]; then
  MAX_ARG="--max $MAX_CONVERSATIONS"
else
  MAX_ARG=""
fi

# Run the indexer
echo "🔍 Indexing conversations from $INPUT_DIR into $DB_PATH..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m memory_ai.tools.index --input "$INPUT_DIR" --db "$DB_PATH" $MAX_ARG

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Indexing complete!"