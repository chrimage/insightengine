# InsightEngine: Memory-Enhanced AI System

InsightEngine is a comprehensive memory-enhanced AI system that provides context-aware responses by leveraging past conversations and learned insights. This project implements a sophisticated memory system with rolling summaries, vector-based retrieval, quality assessment, and self-reflection capabilities.

## Features

- **Chronological Memory**: Rolling summaries that build narrative understanding over time
- **Specific Memory**: Vector-based retrieval for precise information access
- **Memory Quality**: Filtering mechanism to ensure high-quality context
- **Self-Reflection**: System to learn from past interactions and improve

## Project Structure

```
memory_ai/
├── core/           # Core system components
├── memory/         # Memory processing and retrieval
├── reflection/     # Response evaluation and insight extraction
├── chat/           # Chat agent and prompt templates
├── utils/          # Utility functions
└── tools/          # CLI tools for indexing and interaction
```

## Getting Started

### Prerequisites

- Python 3.8+
- Google API key for Gemini access

### Installation on Ubuntu

1. Clone the repository:
   ```bash
   git clone https://github.com/chrimage/insightengine.git
   cd insightengine
   ```

2. Set up a Python virtual environment (recommended):
   ```bash
   # Install virtualenv if not already installed
   sudo apt update
   sudo apt install python3-venv

   # Create a virtual environment
   python3 -m venv venv

   # Activate the virtual environment
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit the .env file with your preferred text editor
   nano .env
   ```

   Update the `.env` file with your Google API key and other settings:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   # Uncomment and modify other settings as needed
   ```

### Important Notes About `.env` File

The system uses a `.env` file for configuration. A `.env.example` file is provided as a template. Make sure to:

1. Create your `.env` file from the example: `cp .env.example .env`
2. Add your Google API key for Gemini in the `.env` file
3. Adjust other settings as needed (see [Environment Variables](#environment-variables) section)

All scripts automatically load settings from this file, so you don't need to export environment variables manually.

### Usage

#### Indexing OpenAI Conversations

To index your OpenAI conversation exports, use the provided shell script:

```bash
./index_conversations.sh /path/to/openai_conversations memory.db
```

Or run the Python module directly:

```bash
python -m memory_ai.tools.index --input /path/to/openai_conversations --db memory.db
```

#### Generating Summaries

Generate rolling summaries from the indexed conversations using the provided shell script:

```bash
./summarize_conversations.sh memory.db
```

Or run the Python module directly:

```bash
python -m memory_ai.tools.summarize --db memory.db
```

You can also use additional options:

```bash
# Rebuild all summaries
./summarize_conversations.sh memory.db --rebuild

# Search for summaries by theme
./summarize_conversations.sh memory.db --themes "artificial intelligence"
```

#### Interactive Chat

Start an interactive chat session:

```bash
./chat.sh memory.db
```

Or run the Python module directly:

```bash
python -m memory_ai.tools.interact --db memory.db
```

### Commands during Chat

- Type `exit` to quit the chat session
- Type `clear` to clear the current conversation history
- Type `save` to save the current conversation to a file

## How It Works

1. **Indexing**: OpenAI conversation exports are parsed, indexed, and stored in a SQLite database with vector embeddings for search.

2. **Rolling Summaries**: The system processes conversations in chronological order to build a comprehensive understanding of topics, preferences, and information. Summaries include structured sections for core interests, technical expertise, and recurring topics.

3. **Context Assembly**: When responding to queries, the system combines:
   - Rolling summaries for broad context
   - Relevant specific memories for detailed information
   - Learned insights from past interactions
   - Current conversation history

4. **Quality Assessment**: Memory quality is evaluated based on information density, coherence, specificity, and factual likelihood. The system prioritizes high-quality conversations while maintaining chronological ordering.

5. **Self-Reflection**: The system evaluates its own responses to extract actionable insights for improving future interactions.

## Customization

- Adjust memory parameters in `core/config.py` or through your `.env` file
- Modify prompt templates in `chat/prompts.py`
- Extend quality assessment in `memory/quality.py`

### Environment Variables

InsightEngine supports configuration through environment variables in a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Your Google API key for Gemini access | None |
| `DB_PATH` | Path to the SQLite database file | `memory.db` |
| `LLM_MODEL` | Gemini model to use for text generation | `gemini-2.0-flash` |
| `EMBEDDING_MODEL` | Model to use for embeddings | `models/text-embedding-004` |
| `USE_DUMMY_EMBEDDINGS` | Use deterministic embeddings (no API calls) | `false` |
| `VERBOSE_EMBEDDINGS` | Enable verbose logging for embeddings | `false` |
| `DEBUG_EMBEDDINGS` | Enable debug logging for embedding API calls | `false` |
| `BATCH_SIZE` | Batch size for processing | `10` |
| `MAX_TOKENS` | Maximum tokens for LLM context | `8000` |
| `QUALITY_THRESHOLD` | Threshold for memory quality filtering | `0.6` |
| `DAYS_THRESHOLD` | Days before memories start to decay | `180` |

## License

This project is licensed under the MIT License - see the LICENSE file for details.
