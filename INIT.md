# Memory-Enhanced AI: Streamlined Implementation Guide

This guide provides a focused, practical roadmap for implementing a memory-enhanced AI system with rolling summaries and self-improvement capabilities. The architecture balances comprehensive memory with quality control and continuous learning.

## 1. System Overview

The Memory-Enhanced AI combines four key components:

1. **Chronological Memory**: Rolling summaries that build a narrative understanding
2. **Specific Memory**: Vector-based retrieval for precise information access
3. **Memory Quality**: Filtering mechanism to ensure high-quality context
4. **Self-Reflection**: System to learn from past interactions and improve

<div align="center">
  <img src="https://i.imgur.com/mV2bDa8.png" alt="Simplified Memory AI Architecture" width="700">
</div>

## 2. Project Structure

```
memory_ai/
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── database.py            # Database operations with sqlite-vec
│   └── models.py              # Data models for system
├── memory/
│   ├── __init__.py
│   ├── parser.py              # Conversation parsing utilities
│   ├── summary.py             # Rolling summary implementation
│   ├── retriever.py           # Vector-based memory retrieval
│   ├── context.py             # Context assembly from multiple sources
│   └── quality.py             # Memory quality assessment
├── reflection/
│   ├── __init__.py
│   ├── evaluator.py           # Response quality evaluation
│   ├── insights.py            # Wisdom extraction and application
│   └── repository.py          # Storage for learned insights
├── chat/
│   ├── __init__.py
│   ├── agent.py               # Main chat agent implementation
│   └── prompts.py             # Templated prompts for different scenarios
├── utils/
│   ├── __init__.py
│   ├── gemini.py              # Gemini API wrapper with optimizations
│   ├── tokens.py              # Token counting and management
│   └── optimize.py            # Performance optimization utilities
├── tools/
│   ├── __init__.py
│   ├── index.py               # Tool for indexing conversations
│   ├── summarize.py           # Tool for generating summaries
│   └── interact.py            # Interactive chat interface
└── requirements.txt           # Project dependencies
```

## 3. Core Implementation

### 3.1 Database Setup with sqlite-vec

```python
# core/database.py

import sqlite3
import os
import sqlite_vec
import json
import numpy as np

class MemoryDatabase:
    """Database for storing conversations, embeddings, and summaries."""
    
    def __init__(self, db_path):
        """Initialize the database connection."""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = self._connect()
        self._initialize_schema()
    
    def _connect(self):
        """Create a connection with sqlite-vec enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Enable the sqlite-vec extension
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        
        return conn
    
    def _initialize_schema(self):
        """Initialize the database schema."""
        c = self.conn.cursor()
        
        # Create tables
        c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            timestamp REAL,
            model TEXT,
            message_count INTEGER,
            summary TEXT,
            quality_score REAL,
            token_count INTEGER
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            source_id TEXT,
            embedding BLOB,  # Binary vector for sqlite-vec
            model TEXT,
            dimensions INTEGER
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS rolling_summaries (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            summary_text TEXT,
            conversation_range TEXT,  # JSON array of conversation IDs
            version INTEGER
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS insights (
            id TEXT PRIMARY KEY,
            text TEXT,
            category TEXT,
            confidence REAL,
            created_at REAL,
            updated_at REAL,
            evidence TEXT,  # JSON array
            application_count INTEGER,
            success_rate REAL
        )
        ''')
        
        # Create indices for better performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_source_id ON embeddings(source_id)')
        
        self.conn.commit()
    
    def store_embedding(self, source_id, embedding_array, model="gemini-embedding"):
        """Store an embedding in sqlite-vec format."""
        c = self.conn.cursor()
        
        # Convert numpy array or list to sqlite-vec format
        if isinstance(embedding_array, np.ndarray):
            embedding_bytes = sqlite_vec.serialize_float32(embedding_array.tolist())
        else:
            embedding_bytes = sqlite_vec.serialize_float32(embedding_array)
        
        embedding_id = f"emb_{source_id}"
        
        c.execute('''
        INSERT OR REPLACE INTO embeddings (id, source_id, embedding, model, dimensions)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            embedding_id,
            source_id,
            embedding_bytes,
            model,
            len(embedding_array)
        ))
        
        self.conn.commit()
    
    def search_similar(self, query_embedding, top_k=5):
        """Find similar items using vector similarity search."""
        c = self.conn.cursor()
        
        # Convert query embedding to sqlite-vec format
        if isinstance(query_embedding, np.ndarray):
            query_bytes = sqlite_vec.serialize_float32(query_embedding.tolist())
        else:
            query_bytes = sqlite_vec.serialize_float32(query_embedding)
        
        # Perform cosine similarity search
        c.execute('''
        SELECT source_id, vec_distance_cosine(embedding, ?) AS distance
        FROM embeddings
        ORDER BY distance ASC
        LIMIT ?
        ''', (query_bytes, top_k))
        
        # Get results and convert to similarity scores (1 - distance)
        results = []
        for row in c.fetchall():
            results.append({
                'source_id': row[0],
                'similarity': 1 - row[1]  # Convert distance to similarity score
            })
        
        return results
    
    def store_rolling_summary(self, summary_data):
        """Store a rolling summary."""
        c = self.conn.cursor()
        
        c.execute('''
        INSERT OR REPLACE INTO rolling_summaries
        (id, timestamp, summary_text, conversation_range, version)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            summary_data['id'],
            summary_data['timestamp'],
            summary_data['summary_text'],
            json.dumps(summary_data['conversation_range']),
            summary_data['version']
        ))
        
        self.conn.commit()
    
    def get_latest_summary(self):
        """Get the most recent rolling summary."""
        c = self.conn.cursor()
        
        c.execute('''
        SELECT id, timestamp, summary_text, conversation_range, version
        FROM rolling_summaries
        ORDER BY timestamp DESC
        LIMIT 1
        ''')
        
        row = c.fetchone()
        if not row:
            return None
        
        return {
            'id': row[0],
            'timestamp': row[1],
            'summary_text': row[2],
            'conversation_range': json.loads(row[3]),
            'version': row[4]
        }
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
```

### 3.2 Rolling Summary Implementation

```python
# memory/summary.py

import uuid
from datetime import datetime
import tiktoken
from core.database import MemoryDatabase
from utils.gemini import GeminiClient

class RollingSummaryProcessor:
    """Processes conversations to generate rolling summaries."""
    
    def __init__(self, db, gemini_client, batch_size=10, max_tokens=8000):
        """Initialize the rolling summary processor."""
        self.db = db
        self.gemini_client = gemini_client
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def process_conversations(self):
        """Process all conversations to generate rolling summaries."""
        # Get conversations ordered by timestamp
        conversations = self._get_chronological_conversations()
        
        # Get the latest summary (if any)
        latest_summary = self.db.get_latest_summary()
        current_summary_text = latest_summary['summary_text'] if latest_summary else None
        processed_convs = set(latest_summary['conversation_range'] if latest_summary else [])
        
        # Process conversations in batches
        batches = [conversations[i:i+self.batch_size] for i in range(0, len(conversations), self.batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            # Skip if all conversations in this batch are already processed
            if all(conv['id'] in processed_convs for conv in batch):
                continue
            
            # Prepare batch content
            batch_content = self._format_batch_content(batch)
            
            # Generate updated summary
            new_summary = self._generate_summary(batch_content, current_summary_text)
            
            # Store the new summary
            summary_id = str(uuid.uuid4())
            version = (latest_summary['version'] + 1) if latest_summary else 1
            
            # Update processed conversations
            for conv in batch:
                processed_convs.add(conv['id'])
            
            summary_data = {
                'id': summary_id,
                'timestamp': datetime.now().timestamp(),
                'summary_text': new_summary,
                'conversation_range': list(processed_convs),
                'version': version
            }
            
            self.db.store_rolling_summary(summary_data)
            
            # Update current summary for next iteration
            current_summary_text = new_summary
            latest_summary = summary_data
    
    def _get_chronological_conversations(self):
        """Get all conversations ordered by timestamp."""
        c = self.db.conn.cursor()
        c.execute('''
        SELECT id, title, timestamp, model, message_count, summary, quality_score
        FROM conversations
        ORDER BY timestamp ASC
        ''')
        
        return [dict(row) for row in c.fetchall()]
    
    def _format_batch_content(self, batch):
        """Format a batch of conversations for the summary prompt."""
        formatted_content = ""
        
        for conv in batch:
            formatted_content += f"\n## CONVERSATION: {conv['title']}\n"
            formatted_content += f"Date: {datetime.fromtimestamp(conv['timestamp']).strftime('%Y-%m-%d')}\n"
            
            # Get messages for this conversation
            c = self.db.conn.cursor()
            c.execute('''
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            ''', (conv['id'],))
            
            messages = c.fetchall()
            for msg in messages:
                formatted_content += f"{msg[0].upper()}: {msg[1]}\n\n"
            
            formatted_content += "\n---\n"
        
        return formatted_content
    
    def _generate_summary(self, batch_content, previous_summary=None):
        """Generate a new summary incorporating the previous summary and new content."""
        if previous_summary:
            prompt = f"""
            Your task is to maintain an evolving, comprehensive summary of all conversations with a user over time.

            PREVIOUS CUMULATIVE SUMMARY:
            {previous_summary}

            NEW CONVERSATIONS:
            {batch_content}

            Update the cumulative summary to incorporate these new conversations. This is a chronological summary that should:

            1. Maintain important information from the previous summary
            2. Add new topics, preferences, and insights from these new conversations
            3. Note evolving patterns, interests, or changes in opinion
            4. Explicitly mark outdated information (when new information contradicts old)
            5. Organize information by themes while preserving chronological developments

            Focus on being COMPREHENSIVE rather than concise.
            """
        else:
            prompt = f"""
            Create a comprehensive summary of these initial conversations:

            {batch_content}

            Focus on capturing all important information, organized by themes but maintaining chronological awareness.
            """
        
        # Generate summary with Gemini
        response = self.gemini_client.generate_content(
            prompt=prompt,
            max_output_tokens=self.max_tokens,
            temperature=0.2
        )
        
        return response.text
    
    def implement_forgetting(self, days_threshold=180):
        """Apply a forgetting mechanism to gradually remove outdated information."""
        latest_summary = self.db.get_latest_summary()
        if not latest_summary:
            return None
        
        # Create a prompt to update the summary by forgetting outdated information
        prompt = f"""
        Your task is to implement a "forgetting mechanism" on this comprehensive conversation summary.

        CURRENT SUMMARY:
        {latest_summary['summary_text']}

        Please update the summary to:

        1. Retain important, timeless information (core facts about the user, long-term interests)
        2. Remove or generalize information that is likely outdated (older than {days_threshold} days)
        3. Prioritize more recent information when there are contradictions
        4. Clearly mark information that is being retained despite being old (e.g., "Long-standing interest:...")

        The goal is to focus the summary on more relevant information while maintaining a comprehensive 
        understanding of the user's history.

        Provide a complete, updated summary that replaces the previous one.
        """
        
        response = self.gemini_client.generate_content(
            prompt=prompt,
            max_output_tokens=self.max_tokens,
            temperature=0.2
        )
        
        # Store the updated summary
        summary_id = str(uuid.uuid4())
        
        summary_data = {
            'id': summary_id,
            'timestamp': datetime.now().timestamp(),
            'summary_text': response.text,
            'conversation_range': latest_summary['conversation_range'],
            'version': latest_summary['version'] + 1
        }
        
        self.db.store_rolling_summary(summary_data)
        
        return summary_data
```

### 3.3 Memory Quality Assessment

```python
# memory/quality.py

class MemoryQualityAssessor:
    """Assesses and filters memories based on quality metrics."""
    
    def __init__(self, db, gemini_client, quality_threshold=0.6):
        """Initialize the quality assessor."""
        self.db = db
        self.gemini_client = gemini_client
        self.quality_threshold = quality_threshold
    
    def assess_memory_quality(self, memory_id, content):
        """Assess the quality of a memory based on multiple factors."""
        # Calculate a quality score (0.0-1.0) based on heuristics
        
        # 1. Information density - more information is better
        info_density = self._calculate_info_density(content)
        
        # 2. Coherence - well-structured, logical content is better
        coherence = self._calculate_coherence(content)
        
        # 3. Specificity - specific, detailed information is better than vague
        specificity = self._calculate_specificity(content)
        
        # 4. Factual likelihood - estimate factuality
        factuality = self._estimate_factuality(content)
        
        # Calculate weighted score
        quality_score = (
            info_density * 0.25 +
            coherence * 0.25 +
            specificity * 0.25 +
            factuality * 0.25
        )
        
        # Store the quality score
        self._update_memory_quality(memory_id, quality_score)
        
        return quality_score
    
    def _calculate_info_density(self, content):
        """Calculate information density score."""
        # Simple heuristic based on:
        # - Length of content
        # - Ratio of unique words to total words
        # - Presence of numbers, dates, named entities, etc.
        
        # (Implementation details...)
        return score  # 0.0-1.0
    
    def _calculate_coherence(self, content):
        """Calculate coherence score."""
        # (Implementation details...)
        return score  # 0.0-1.0
    
    def _calculate_specificity(self, content):
        """Calculate specificity score."""
        # (Implementation details...)
        return score  # 0.0-1.0
    
    def _estimate_factuality(self, content):
        """Estimate factuality score."""
        # (Implementation details...)
        return score  # 0.0-1.0
    
    def _update_memory_quality(self, memory_id, quality_score):
        """Update the quality score in the database."""
        c = self.db.conn.cursor()
        c.execute('UPDATE conversations SET quality_score = ? WHERE id = ?', 
                 (quality_score, memory_id))
        self.db.conn.commit()
    
    def filter_memories(self, memories):
        """Filter memories based on quality scores."""
        filtered_memories = []
        
        for memory in memories:
            memory_id = memory['source_id']
            
            # Get stored quality score
            c = self.db.conn.cursor()
            c.execute('SELECT quality_score FROM conversations WHERE id = ?', (memory_id,))
            row = c.fetchone()
            
            if row and row[0] is not None:
                quality_score = row[0]
            else:
                # Get content to assess quality
                c.execute('''
                SELECT content FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
                ''', (memory_id,))
                
                messages = c.fetchall()
                content = "\n".join([msg[0] for msg in messages])
                
                # Assess quality if not already scored
                quality_score = self.assess_memory_quality(memory_id, content)
            
            # Apply quality threshold
            if quality_score >= self.quality_threshold:
                memory['quality_score'] = quality_score
                filtered_memories.append(memory)
        
        # Sort by combination of similarity and quality
        filtered_memories.sort(
            key=lambda x: (x['similarity'] * 0.7) + (x.get('quality_score', 0) * 0.3), 
            reverse=True
        )
        
        return filtered_memories
    
    def update_quality_from_usage(self, memory_id, was_helpful):
        """Update quality score based on usage feedback."""
        # Get current score
        c = self.db.conn.cursor()
        c.execute('SELECT quality_score FROM conversations WHERE id = ?', (memory_id,))
        row = c.fetchone()
        
        if not row:
            return
            
        current_score = row[0] or 0.5  # Default if NULL
        
        # Adjust score based on feedback
        adjustment = 0.05 if was_helpful else -0.05
        new_score = max(0.0, min(1.0, current_score + adjustment))
        
        # Update the score
        c.execute('UPDATE conversations SET quality_score = ? WHERE id = ?', 
                 (new_score, memory_id))
        self.db.conn.commit()
```

### 3.4 Self-Reflection System

```python
# reflection/insights.py

import uuid
from datetime import datetime
import json

class InsightRepository:
    """Repository for storing and retrieving insights from self-reflection."""
    
    def __init__(self, db):
        """Initialize the insight repository."""
        self.db = db
    
    def add_insight(self, text, category, confidence, evidence=None):
        """Add a new insight to the repository."""
        insight_id = str(uuid.uuid4())
        now = datetime.now().timestamp()
        
        c = self.db.conn.cursor()
        c.execute('''
        INSERT INTO insights
        (id, text, category, confidence, created_at, updated_at, evidence, 
         application_count, success_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            insight_id,
            text,
            category,
            confidence,
            now,
            now,
            json.dumps(evidence or []),
            0,
            0.0
        ))
        
        self.db.conn.commit()
        return insight_id
    
    def get_relevant_insights(self, query, top_k=3):
        """Get insights relevant to the query."""
        # This could be implemented in various ways:
        # 1. Keyword matching
        # 2. Embedding similarity (more advanced)
        # 3. Category-based retrieval
        
        # Simple keyword approach for now
        c = self.db.conn.cursor()
        
        # Get all insights
        c.execute('''
        SELECT id, text, category, confidence, application_count, success_rate
        FROM insights
        ORDER BY confidence DESC, success_rate DESC
        ''')
        
        all_insights = [dict(row) for row in c.fetchall()]
        
        # Filter for relevance
        relevant_insights = []
        query_words = set(query.lower().split())
        
        for insight in all_insights:
            insight_words = set(insight['text'].lower().split())
            # Check word overlap as a simple relevance measure
            overlap = len(query_words.intersection(insight_words))
            if overlap > 0:
                insight['relevance'] = overlap / len(insight_words)
                relevant_insights.append(insight)
        
        # Sort by relevance and quality
        relevant_insights.sort(
            key=lambda x: (x.get('relevance', 0) * 0.3) + 
                         (x['confidence'] * 0.4) + 
                         (x['success_rate'] * 0.3),
            reverse=True
        )
        
        return relevant_insights[:top_k]
    
    def update_insight_performance(self, insight_id, was_successful):
        """Update the performance metrics for an insight."""
        c = self.db.conn.cursor()
        
        # Get current metrics
        c.execute('''
        SELECT application_count, success_rate
        FROM insights
        WHERE id = ?
        ''', (insight_id,))
        
        row = c.fetchone()
        if not row:
            return
            
        count, rate = row
        
        # Update metrics
        new_count = count + 1
        success_count = (count * rate) + (1 if was_successful else 0)
        new_rate = success_count / new_count
        
        # Save updated metrics
        c.execute('''
        UPDATE insights
        SET application_count = ?, success_rate = ?, updated_at = ?
        WHERE id = ?
        ''', (new_count, new_rate, datetime.now().timestamp(), insight_id))
        
        self.db.conn.commit()

class ResponseEvaluator:
    """Evaluates AI responses to extract insights and improve future responses."""
    
    def __init__(self, db, gemini_client, insight_repository):
        """Initialize the response evaluator."""
        self.db = db
        self.gemini_client = gemini_client
        self.insight_repository = insight_repository
    
    def evaluate_response(self, query, response, conversation_history=None):
        """Evaluate a response and extract insights."""
        # Format the conversation for evaluation
        conv_text = self._format_conversation(query, response, conversation_history)
        
        evaluation_prompt = f"""
        Analyze this AI interaction objectively:
        
        {conv_text}
        
        Evaluate the AI's response on these dimensions:
        1. Relevance to the query (1-10)
        2. Accuracy of information (1-10)
        3. Helpfulness to the user (1-10)
        4. Coherence and clarity (1-10)
        5. Overall quality (1-10)
        
        Then identify:
        - What specifically worked well in this response
        - What could be improved
        - One specific lesson that could be applied to future responses
        
        Format your evaluation as a structured assessment with scores and detailed feedback.
        Include ONE clear, actionable insight in the format:
        INSIGHT: [Category] - [Specific lesson learned]
        """
        
        try:
            eval_response = self.gemini_client.generate_content(
                prompt=evaluation_prompt,
                max_output_tokens=1000,
                temperature=0.2
            )
            
            # Extract insight from evaluation
            insight_text = None
            category = "response_quality"
            
            # Simple extraction with string operations
            if "INSIGHT:" in eval_response.text:
                insight_section = eval_response.text.split("INSIGHT:")[1].strip()
                if "-" in insight_section:
                    parts = insight_section.split("-", 1)
                    category = parts[0].strip()
                    insight_text = parts[1].strip()
                else:
                    insight_text = insight_section
            
            # Add insight if found
            if insight_text:
                evidence = [{
                    "query": query,
                    "response": response,
                    "evaluation": eval_response.text
                }]
                
                self.insight_repository.add_insight(
                    text=insight_text,
                    category=category,
                    confidence=0.7,  # Start with moderate confidence
                    evidence=evidence
                )
            
            return {
                "evaluation": eval_response.text,
                "insight_extracted": bool(insight_text)
            }
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {
                "evaluation": "Evaluation failed",
                "insight_extracted": False
            }
    
    def _format_conversation(self, query, response, history=None):
        """Format the conversation for evaluation."""
        formatted = ""
        
        # Add history if available
        if history:
            formatted += "PREVIOUS CONVERSATION:\n"
            for msg in history:
                role = "USER" if msg.get('role') == 'user' else "AI"
                formatted += f"{role}: {msg.get('content', '')}\n\n"
        
        # Add the current interaction
        formatted += "CURRENT INTERACTION:\n"
        formatted += f"USER: {query}\n\n"
        formatted += f"AI: {response}\n"
        
        return formatted
```

### 3.5 Context Assembly and Chat Agent

```python
# memory/context.py

import tiktoken

class ContextAssembler:
    """Assembles context from multiple sources for use in prompts."""
    
    def __init__(self, db, gemini_client, retriever, insight_repository):
        """Initialize the context assembler."""
        self.db = db
        self.gemini_client = gemini_client
        self.retriever = retriever
        self.insight_repository = insight_repository
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def assemble_context(self, query, conversation_history=None, max_tokens=900000):
        """Assemble comprehensive context for the query."""
        context_parts = []
        metadata = {
            "components": [],
            "token_counts": {},
            "total_tokens": 0
        }
        
        # 1. Get the latest rolling summary
        summary = self.db.get_latest_summary()
        if summary:
            summary_text = f"## CONVERSATION HISTORY SUMMARY\n\n{summary['summary_text']}"
            summary_tokens = len(self.tokenizer.encode(summary_text))
            
            # Truncate if too long (but keep substantial portion)
            if summary_tokens > 80000:  # Arbitrary limit to leave room for other components
                summary_text = self.tokenizer.decode(
                    self.tokenizer.encode(summary_text)[:80000]
                ) + "\n[Summary truncated for length...]"
                summary_tokens = 80000
            
            context_parts.append(summary_text)
            metadata["components"].append("rolling_summary")
            metadata["token_counts"]["rolling_summary"] = summary_tokens
            metadata["total_tokens"] += summary_tokens
        
        # 2. Get relevant memories
        memories = self.retriever.retrieve_memories(query, top_k=5)
        if memories:
            memories_text = "## SPECIFIC RELEVANT MEMORIES\n\n"
            for i, memory in enumerate(memories, 1):
                memory_text = f"MEMORY {i}:\n"
                
                # Get conversation details
                c = self.db.conn.cursor()
                c.execute('''
                SELECT title, timestamp FROM conversations
                WHERE id = ?
                ''', (memory['source_id'],))
                
                conv = c.fetchone()
                if conv:
                    memory_text += f"Title: {conv[0]}\n"
                    memory_text += f"Date: {datetime.fromtimestamp(conv[1]).strftime('%Y-%m-%d')}\n"
                
                # Get content
                c.execute('''
                SELECT role, content FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT 20  # Limit number of messages
                ''', (memory['source_id'],))
                
                messages = c.fetchall()
                content = "\n".join([f"{msg[0].upper()}: {msg[1]}" for msg in messages])
                
                # Truncate if too long
                content_tokens = len(self.tokenizer.encode(content))
                if content_tokens > 2000:  # Limit per memory
                    content = self.tokenizer.decode(
                        self.tokenizer.encode(content)[:2000]
                    ) + "\n[Content truncated for length...]"
                
                memory_text += f"Content:\n{content}\n\n"
                memories_text += memory_text
            
            memories_tokens = len(self.tokenizer.encode(memories_text))
            context_parts.append(memories_text)
            metadata["components"].append("relevant_memories")
            metadata["token_counts"]["relevant_memories"] = memories_tokens
            metadata["total_tokens"] += memories_tokens
            metadata["memory_ids"] = [m['source_id'] for m in memories]
        
        # 3. Get relevant insights
        insights = self.insight_repository.get_relevant_insights(query)
        if insights:
            insights_text = "## RESPONSE GUIDANCE\n\n"
            for insight in insights:
                insights_text += f"- {insight['text']}\n"
            
            insights_tokens = len(self.tokenizer.encode(insights_text))
            context_parts.append(insights_text)
            metadata["components"].append("insights")
            metadata["token_counts"]["insights"] = insights_tokens
            metadata["total_tokens"] += insights_tokens
            metadata["insight_ids"] = [i['id'] for i in insights]
        
        # 4. Add conversation history
        if conversation_history:
            history_text = "## CURRENT CONVERSATION\n\n"
            for msg in conversation_history:
                role = "USER" if msg.get('role') == 'user' else "AI"
                history_text += f"{role}: {msg.get('content', '')}\n\n"
            
            history_tokens = len(self.tokenizer.encode(history_text))
            context_parts.append(history_text)
            metadata["components"].append("conversation_history")
            metadata["token_counts"]["conversation_history"] = history_tokens
            metadata["total_tokens"] += history_tokens
        
        # Combine and return
        combined_context = "\n\n".join(context_parts)
        
        # Enforce overall token limit if needed
        if metadata["total_tokens"] > max_tokens:
            truncated_context = self.tokenizer.decode(
                self.tokenizer.encode(combined_context)[:max_tokens]
            ) + "\n[Context truncated for length...]"
            metadata["total_tokens"] = max_tokens
            metadata["truncated"] = True
            return truncated_context, metadata
        
        return combined_context, metadata

# chat/agent.py

class MemoryEnhancedAgent:
    """Chat agent with enhanced memory capabilities."""
    
    def __init__(self, db, gemini_client, context_assembler, evaluator):
        """Initialize the memory-enhanced chat agent."""
        self.db = db
        self.gemini_client = gemini_client
        self.context_assembler = context_assembler
        self.evaluator = evaluator
        self.conversation_history = []
    
    def chat(self, message):
        """Process a message and generate a response."""
        # Add message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Assemble context
        context, metadata = self.context_assembler.assemble_context(
            query=message,
            conversation_history=self.conversation_history
        )
        
        # Build prompt
        prompt = self._build_prompt(context, metadata)
        
        # Generate response
        response = self.gemini_client.generate_content(
            prompt=prompt,
            max_output_tokens=8000,
            temperature=0.7
        )
        
        response_text = response.text
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Evaluate response in the background
        # This could be run asynchronously or in a separate thread
        self.evaluator.evaluate_response(
            query=message,
            response=response_text,
            conversation_history=self.conversation_history[:-2]  # Exclude current interaction
        )
        
        # Update memory quality scores based on usage
        if "memory_ids" in metadata:
            # This would ideally be done based on feedback, but we'll assume 
            # memories were helpful for now
            for memory_id in metadata["memory_ids"]:
                self.db.conn.execute('''
                UPDATE conversations 
                SET quality_score = quality_score + 0.01
                WHERE id = ? AND quality_score < 1.0
                ''', (memory_id,))
                self.db.conn.commit()
        
        # Update insight performance if insights were used
        if "insight_ids" in metadata:
            for insight_id in metadata["insight_ids"]:
                # We would ideally determine this based on actual outcome
                was_successful = True  # Placeholder
                evaluator.insight_repository.update_insight_performance(
                    insight_id, was_successful
                )
        
        return response_text
    
    def _build_prompt(self, context, metadata):
        """Build a prompt for Gemini with system instructions and context."""
        system_prompt = """You are an AI assistant with an enhanced memory system that allows you to reference past conversations and learned insights. Use this context to provide personalized, helpful responses. Incorporate relevant information from your memory when appropriate, but do not explicitly mention the memory system unless asked directly about your capabilities."""
        
        # Add specific guidance based on available context components
        if "rolling_summary" in metadata.get("components", []):
            system_prompt += "\nRefer to the conversation history summary for overall context about the user."
        
        if "relevant_memories" in metadata.get("components", []):
            system_prompt += "\nIncorporate specific details from relevant memories when they directly relate to the query."
        
        if "insights" in metadata.get("components", []):
            system_prompt += "\nApply the response guidance to improve your answer quality."
        
        # Combine system prompt with context and query
        full_prompt = f"{system_prompt}\n\n{context}\n\nProvide a helpful, thorough response based on the information above."
        
        return full_prompt
```

### 3.6 Gemini API Wrapper

```python
# utils/gemini.py

from google import genai
import os
import time
import random
from collections import deque

class GeminiClient:
    """Optimized client for interacting with Gemini API."""
    
    def __init__(self, api_key=None, model="gemini-2.0-flash"):
        """Initialize the Gemini client."""
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided and GOOGLE_API_KEY not set")
        
        self.model = model
        self.client = genai.Client(api_key=self.api_key)
        
        # Rate limiting parameters
        self.max_requests_per_minute = 60  # Adjust based on your quota
        self.request_timestamps = deque(maxlen=self.max_requests_per_minute)
        self.max_retries = 5
        self.base_retry_delay = 2
    
    def generate_content(self, prompt, max_output_tokens=8000, temperature=0.7, 
                        top_p=0.95, top_k=64):
        """Generate content with automatic rate limiting and retries."""
        retry_count = 0
        backoff_time = self.base_retry_delay
        
        while retry_count <= self.max_retries:
            # Apply rate limiting
            self._handle_rate_limiting()
            
            try:
                # Make the API call
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                        "top_p": top_p,
                        "top_k": top_k
                    }
                )
                
                return response
                
            except Exception as e:
                if self._is_rate_limit_error(e):
                    retry_count += 1
                    
                    if retry_count <= self.max_retries:
                        # Apply exponential backoff with jitter
                        jitter = random.uniform(0.8, 1.2)
                        actual_delay = backoff_time * jitter
                        
                        print(f"Rate limit exceeded. Retrying in {actual_delay:.1f}s ({retry_count}/{self.max_retries})")
                        time.sleep(actual_delay)
                        
                        # Increase backoff for next attempt
                        backoff_time = min(backoff_time * 2, 60)
                    else:
                        print(f"Max retries exceeded.")
                        raise
                else:
                    # Not a rate limit error
                    raise
        
        raise Exception("Failed after maximum retries")
    
    def generate_embeddings(self, text, embedding_model="models/text-embedding-004"):
        """Generate embeddings with rate limiting and retries."""
        retry_count = 0
        backoff_time = self.base_retry_delay
        
        while retry_count <= self.max_retries:
            # Apply rate limiting
            self._handle_rate_limiting()
            
            try:
                # Make the API call
                result = self.client.models.embed_content(
                    model=embedding_model,
                    contents=text
                )
                
                # Extract and return embeddings
                if hasattr(result, 'embeddings') and result.embeddings:
                    return result.embeddings[0].values
                else:
                    raise ValueError("No embeddings in response")
                    
            except Exception as e:
                if self._is_rate_limit_error(e):
                    # Handle rate limit errors (same as in generate_content)
                    # ...
                    pass
                else:
                    raise
        
        raise Exception("Failed after maximum retries")
    
    def _handle_rate_limiting(self):
        """Handle rate limiting logic."""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
        
        # If we're at the limit, wait
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.request_timestamps[0]) + 1  # +1 buffer
            print(f"Rate limit approaching. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        # Record this request
        self.request_timestamps.append(time.time())
    
    def _is_rate_limit_error(self, error):
        """Check if an error is related to rate limiting."""
        error_str = str(error).lower()
        return any(term in error_str for term in 
                  ["429", "resource exhausted", "quota", "rate limit"])
```

## 4. CLI Tools Implementation

### 4.1 Indexing Tool

```python
# tools/index.py

import argparse
import os
import json
from tqdm import tqdm
import uuid
from datetime import datetime
from core.database import MemoryDatabase
from memory.parser import OpenAIParser
from utils.gemini import GeminiClient

def index_conversations(input_dir, db_path, max_conversations=None):
    """Index OpenAI conversations into the memory system."""
    print(f"Indexing conversations from {input_dir} into {db_path}")
    
    # Initialize components
    db = MemoryDatabase(db_path)
    parser = OpenAIParser()
    gemini = GeminiClient()
    
    # Find JSON files
    json_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files")
    
    # Limit if specified
    if max_conversations and max_conversations < len(json_files):
        json_files = json_files[:max_conversations]
        print(f"Limiting to {max_conversations} files")
    
    # Process each file
    for file_path in tqdm(json_files, desc="Processing conversations"):
        try:
            # Load and parse the conversation
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversation = parser.parse(data)
            
            # Store in database
            store_conversation(db, conversation, gemini)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Close database
    db.close()
    print(f"Indexed {len(json_files)} conversations")

def store_conversation(db, conversation, gemini):
    """Store a conversation and its embeddings in the database."""
    c = db.conn.cursor()
    
    # Store conversation
    c.execute('''
    INSERT OR REPLACE INTO conversations
    (id, title, timestamp, model, message_count, summary, quality_score, token_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        conversation.id,
        conversation.title,
        conversation.timestamp.timestamp() if conversation.timestamp else datetime.now().timestamp(),
        conversation.model,
        len(conversation.messages),
        "",  # Summary will be generated later
        None,  # Quality score to be determined
        0  # Token count to be calculated
    ))
    
    # Store messages
    for message in conversation.messages:
        message_id = str(uuid.uuid4())
        c.execute('''
        INSERT OR REPLACE INTO messages
        (id, conversation_id, role, content, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            message_id,
            conversation.id,
            message.role,
            message.content,
            message.timestamp.timestamp() if message.timestamp else datetime.now().timestamp()
        ))
    
    # Generate and store embedding
    try:
        # Combine messages for embedding
        content = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in conversation.messages])
        
        # Generate embedding
        embedding = gemini.generate_embeddings(content)
        
        # Store embedding
        db.store_embedding(conversation.id, embedding)
        
    except Exception as e:
        print(f"Error generating embedding for conversation {conversation.id}: {e}")
    
    db.conn.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index conversations for the memory system")
    parser.add_argument("--input", required=True, help="Input directory with conversation files")
    parser.add_argument("--db", default="memory.db", help="Output database path")
    parser.add_argument("--max", type=int, help="Maximum number of conversations to process")
    
    args = parser.parse_args()
    
    index_conversations(args.input, args.db, args.max)
```

### 4.2 Summary Generation Tool

```python
# tools/summarize.py

import argparse
from core.database import MemoryDatabase
from memory.summary import RollingSummaryProcessor
from utils.gemini import GeminiClient

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
```

### 4.3 Interactive Chat Tool

```python
# tools/interact.py

import argparse
import os
import time
import json
from datetime import datetime

from core.database import MemoryDatabase
from memory.retriever import MemoryRetriever
from memory.context import ContextAssembler
from memory.quality import MemoryQualityAssessor
from reflection.insights import InsightRepository, ResponseEvaluator
from chat.agent import MemoryEnhancedAgent
from utils.gemini import GeminiClient

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
            agent.conversation_history = []
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
```

## 5. Putting It All Together: Workflow

### 5.1 Setup

```bash
# Create environment directories
mkdir -p ~/.memory_ai/data

# Set up environment variables
export GOOGLE_API_KEY=your_key_here

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Process OpenAI Data

```bash
# Step 1: Index conversations
python -m tools.index --input /path/to/openai_export --db ~/.memory_ai/data/memory.db

# Step 2: Generate rolling summaries
python -m tools.summarize --db ~/.memory_ai/data/memory.db --batch-size 20

# Step 3: Apply forgetting mechanism (optional)
python -m tools.summarize --db ~/.memory_ai/data/memory.db --apply-forgetting
```

### 5.3 Start Chatting

```bash
# Start interactive chat
python -m tools.interact --db ~/.memory_ai/data/memory.db
```

## 6. Extension Possibilities

Once the core system is working, consider these extensions:

1. **Web Interface**: Create a simple web UI using Flask or FastAPI
2. **Scheduled Maintenance**: Set up cron jobs to periodically refresh summaries and apply forgetting
3. **Multi-User Support**: Extend the database to handle multiple user's conversation histories
4. **Feedback Collection**: Add explicit feedback mechanisms to improve quality assessment
5. **Cross-Conversation Links**: Create connections between related conversations

## 7. Conclusion

This implementation provides a streamlined yet powerful memory-enhanced AI system. The architecture focuses on four core capabilities:

1. **Comprehensive Memory**: Rolling summaries provide the big picture while vector retrieval supplies specific details
2. **Quality Control**: Memory filtering ensures only valuable context is used
3. **Self-Improvement**: The system learns from its own interactions to improve over time
4. **Practical Design**: The architecture emphasizes simplicity and modularity while maintaining powerful capabilities

By prioritizing chronological narrative understanding alongside specific retrieval and quality control, this system addresses the limitations of simple RAG approaches while remaining practical to implement. The self-reflection system adds the final piece - the ability to develop "wisdom" from experience and continuously improve its own performance.
