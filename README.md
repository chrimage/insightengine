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

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/insightengine.git
   cd insightengine
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set your Google API key:
   ```
   export GOOGLE_API_KEY=your_key_here
   ```

### Usage

#### Indexing OpenAI Conversations

To index your OpenAI conversation exports:

```bash
python -m memory_ai.tools.index --input /path/to/openai_conversations --db memory.db
```

#### Generating Summaries

Generate rolling summaries from the indexed conversations:

```bash
python -m memory_ai.tools.summarize --db memory.db
```

Apply the forgetting mechanism to remove outdated information:

```bash
python -m memory_ai.tools.summarize --db memory.db --apply-forgetting
```

#### Interactive Chat

Start an interactive chat session:

```bash
python -m memory_ai.tools.interact --db memory.db
```

### Commands during Chat

- Type `exit` to quit the chat session
- Type `clear` to clear the current conversation history
- Type `save` to save the current conversation to a file

## How It Works

1. **Indexing**: OpenAI conversation exports are parsed, indexed, and stored in a SQLite database with vector embeddings for search.

2. **Rolling Summaries**: The system processes conversations in chronological order to build a comprehensive understanding of topics, preferences, and information.

3. **Context Assembly**: When responding to queries, the system combines:
   - Rolling summaries for broad context
   - Relevant specific memories for detailed information
   - Learned insights from past interactions
   - Current conversation history

4. **Quality Assessment**: Memory quality is evaluated based on information density, coherence, specificity, and factual likelihood.

5. **Self-Reflection**: The system evaluates its own responses to extract actionable insights for improving future interactions.

## Customization

- Adjust memory parameters in `core/config.py`
- Modify prompt templates in `chat/prompts.py`
- Extend quality assessment in `memory/quality.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.