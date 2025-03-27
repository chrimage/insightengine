# Database operations with sqlite-vec

import sqlite3
import os
import sqlite_vec
import json
import numpy as np

class MemoryDatabase:
    """Database for storing conversations, embeddings, and summaries."""
    
    def __init__(self, db_path):
        """Initialize the database connection."""
        if os.path.dirname(db_path):  # Only try to create dirs if there's a dirname
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
            embedding BLOB,  -- Binary vector for sqlite-vec
            model TEXT,
            dimensions INTEGER
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS rolling_summaries (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            summary_text TEXT,
            conversation_range TEXT,  -- JSON array of conversation IDs
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
            evidence TEXT,  -- JSON array
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
        """Store a rolling summary with metadata."""
        c = self.conn.cursor()
        
        # Check if we need to add a metadata column
        c.execute("PRAGMA table_info(rolling_summaries)")
        columns = [info[1] for info in c.fetchall()]
        
        # Add metadata column if it doesn't exist
        if 'metadata' not in columns:
            try:
                c.execute('ALTER TABLE rolling_summaries ADD COLUMN metadata TEXT')
                print("Added metadata column to rolling_summaries table")
            except sqlite3.OperationalError as e:
                # Column may already exist in another session
                if "duplicate column name" not in str(e):
                    raise
        
        # Prepare metadata JSON if it exists
        metadata_json = json.dumps(summary_data.get('metadata', {})) if 'metadata' in summary_data else '{}'
        
        # Insert with metadata
        c.execute('''
        INSERT OR REPLACE INTO rolling_summaries
        (id, timestamp, summary_text, conversation_range, version, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            summary_data['id'],
            summary_data['timestamp'],
            summary_data['summary_text'],
            json.dumps(summary_data['conversation_range']),
            summary_data['version'],
            metadata_json
        ))
        
        self.conn.commit()
    
    def get_latest_summary(self):
        """Get the most recent rolling summary."""
        c = self.conn.cursor()
        
        # Check if metadata column exists
        c.execute("PRAGMA table_info(rolling_summaries)")
        columns = [info[1] for info in c.fetchall()]
        has_metadata = 'metadata' in columns
        
        if has_metadata:
            c.execute('''
            SELECT id, timestamp, summary_text, conversation_range, version, metadata
            FROM rolling_summaries
            ORDER BY timestamp DESC
            LIMIT 1
            ''')
        else:
            c.execute('''
            SELECT id, timestamp, summary_text, conversation_range, version
            FROM rolling_summaries
            ORDER BY timestamp DESC
            LIMIT 1
            ''')
        
        row = c.fetchone()
        if not row:
            return None
        
        result = {
            'id': row[0],
            'timestamp': row[1],
            'summary_text': row[2],
            'conversation_range': json.loads(row[3]),
            'version': row[4]
        }
        
        # Add metadata if it exists
        if has_metadata and row[5]:
            try:
                result['metadata'] = json.loads(row[5])
            except (json.JSONDecodeError, TypeError):
                result['metadata'] = {}
        
        return result
    
    def get_summaries_by_theme(self, theme_query, limit=5):
        """Get summaries matching a theme query."""
        c = self.conn.cursor()
        
        # Check if metadata column exists
        c.execute("PRAGMA table_info(rolling_summaries)")
        columns = [info[1] for info in c.fetchall()]
        if 'metadata' not in columns:
            return []
        
        # This is a simple implementation - in a real-world scenario
        # we would use full-text search or vector similarity
        c.execute('''
        SELECT id, timestamp, summary_text, version, metadata
        FROM rolling_summaries
        WHERE metadata IS NOT NULL
        ORDER BY timestamp DESC
        ''')
        
        matching_summaries = []
        for row in c.fetchall():
            try:
                metadata = json.loads(row[4])
                themes = metadata.get('themes', [])
                topics = metadata.get('topics', [])
                
                # Simple string matching search
                theme_query_lower = theme_query.lower()
                if any(theme_query_lower in theme.lower() for theme in themes) or \
                   any(theme_query_lower in topic.lower() for topic in topics):
                    matching_summaries.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'summary_text': row[2],
                        'version': row[3],
                        'metadata': metadata
                    })
            except (json.JSONDecodeError, TypeError):
                continue
        
        return matching_summaries[:limit]
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()