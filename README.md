# InsightEngine: Memory-Enhanced AI System

InsightEngine is a sophisticated memory system for LLMs that provides context-aware responses by leveraging past conversations and learned insights. This project implements a comprehensive memory architecture with rolling summaries, vector-based retrieval, quality assessment, and self-reflection capabilities.

## Features

- **Chronological Memory**: Rolling summaries that build narrative understanding over time
- **Specific Memory**: Vector-based retrieval for precise information access 
- **Memory Quality**: Filtering mechanism to ensure high-quality context
- **Self-Reflection**: System to learn from past interactions and improve
- **Token Budget Management**: Optimized context assembly for large context windows

## Project Structure

```
insightengine/
├── memory_ai/
│   ├── core/           # Core domain models and database layer
│   ├── memory/         # Memory processing and retrieval
│   ├── reflection/     # Response evaluation and insight extraction
│   ├── chat/           # Chat agent and prompt templates
│   ├── utils/          # Utility functions and LLM clients
│   └── tools/          # CLI tools for indexing and interaction
├── tests/              # Test cases
├── alembic/            # Database migrations
└── scripts/            # Utility scripts
```

## Getting Started

### Prerequisites

- Python 3.8+
- Google API key for Gemini access (or OpenAI key)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/insightengine.git
   cd insightengine
   ```

2. Set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit the .env file with your preferred text editor
   ```

   Update the `.env` file with your Google API key and other settings.

### Usage

#### Indexing Conversations

To index your conversation exports:

```bash
python -m memory_ai.tools.index --input /path/to/conversations --db memory.db
```

#### Generating Summaries

Generate rolling summaries from the indexed conversations:

```bash
python -m memory_ai.tools.summarize --db memory.db
```

Options:
```bash
# Rebuild all summaries
python -m memory_ai.tools.summarize --db memory.db --rebuild

# Search for summaries by theme
python -m memory_ai.tools.summarize --db memory.db --themes "artificial intelligence"
```

#### Interactive Chat

Start an interactive chat session:

```bash
python -m memory_ai.tools.interact --db memory.db
```

### How It Works

1. **Indexing**: Conversations are parsed, chunked, and stored in a database with vector embeddings for search.

2. **Rolling Summaries**: The system processes conversations in chronological order to build a comprehensive understanding of topics, preferences, and information.

3. **Context Assembly**: When responding to queries, the system combines:
   - Rolling summaries for broad context
   - Relevant specific memories for detailed information
   - Learned insights from past interactions
   - Current conversation history

4. **Quality Assessment**: Memory quality is evaluated based on information density, coherence, specificity, and factual likelihood.

5. **Self-Reflection**: The system evaluates its own responses to extract actionable insights for improving future interactions.

## Customization

- Adjust memory parameters in your `.env` file
- Modify prompt templates in `chat/prompts.py`
- Extend quality assessment in `memory/quality.py`

## Environment Variables

InsightEngine supports extensive configuration through environment variables in a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Your Google API key for Gemini | Required |
| `OPENAI_API_KEY` | Your OpenAI API key (optional) | None |
| `DEFAULT_LLM_PROVIDER` | LLM provider to use ('gemini' or 'openai') | `gemini` |
| `DB_PATH` | Path to the SQLite database file | `memory.db` |
| `VECTOR_DB_TYPE` | Vector database type | `chroma` |
| `MAX_CONTEXT_TOKENS` | Maximum tokens for LLM context | `8000` |
| `QUALITY_THRESHOLD` | Threshold for memory quality filtering | `0.6` |

For the complete list of configuration options, see `.env.example`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.