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
            success_rate REAL,
            embedding BLOB,  -- Binary vector for vector similarity search
            applications TEXT  -- JSON array of application history
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            query TEXT,
            evaluation TEXT,  -- JSON object with evaluation data
            structured BOOLEAN,
            meta_evaluation TEXT  -- JSON object with meta-evaluation data
        )
        ''')
        
        # Add metadata column to conversations if it doesn't exist
        self._add_column_if_not_exists("conversations", "metadata", "TEXT")
        
        # Create indices for better performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_source_id ON embeddings(source_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_insights_category ON insights(category)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp ON evaluations(timestamp)')
        
        self.conn.commit()
        
    def _add_column_if_not_exists(self, table, column, type):
        """Add a column to a table if it doesn't already exist."""
        c = self.conn.cursor()
        c.execute(f"PRAGMA table_info({table})")
        columns = [info[1] for info in c.fetchall()]
        
        if column not in columns:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type}")
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
        """Store a rolling summary with metadata and embeddings.
        
        Args:
            summary_data: Dictionary containing summary data including:
                - id: Unique ID for the summary
                - timestamp: Creation timestamp
                - summary_text: The actual summary content
                - conversation_range: List of conversation IDs included in this summary
                - version: Summary version number
                - metadata: Optional dictionary of metadata
                - embedding: Optional vector embedding of the summary
                - is_active: Whether this is the currently active global summary
        """
        c = self.conn.cursor()
        
        # Check if we need to add columns
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
                    
        # Add embedding column if it doesn't exist
        if 'embedding' not in columns:
            try:
                c.execute('ALTER TABLE rolling_summaries ADD COLUMN embedding BLOB')
                print("Added embedding column to rolling_summaries table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    raise
        
        # Add is_active column if it doesn't exist (for only keeping one active global summary)
        if 'is_active' not in columns:
            try:
                c.execute('ALTER TABLE rolling_summaries ADD COLUMN is_active BOOLEAN DEFAULT 0')
                print("Added is_active column to rolling_summaries table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    raise
        
        # Prepare metadata JSON if it exists
        metadata_json = json.dumps(summary_data.get('metadata', {})) if 'metadata' in summary_data else '{}'
        
        # Process embedding if provided
        embedding_blob = None
        if 'embedding' in summary_data and summary_data['embedding']:
            import sqlite_vec
            embedding_blob = sqlite_vec.serialize_float32(summary_data['embedding'])
        
        # Set is_active flag (only one global summary should be active)
        is_active = summary_data.get('is_active', True)
        
        # If this summary is marked as active, deactivate all others
        if is_active:
            c.execute("UPDATE rolling_summaries SET is_active = 0")
        
        # Insert with metadata and embedding
        c.execute('''
        INSERT OR REPLACE INTO rolling_summaries
        (id, timestamp, summary_text, conversation_range, version, metadata, embedding, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            summary_data['id'],
            summary_data['timestamp'],
            summary_data['summary_text'],
            json.dumps(summary_data['conversation_range']),
            summary_data['version'],
            metadata_json,
            embedding_blob,
            is_active
        ))
        
        self.conn.commit()
    
    def store_conversation_summary(self, conversation_id, summary_text, metadata=None, embedding=None):
        """Store a summary specific to a single conversation.
        
        Args:
            conversation_id: ID of the conversation
            summary_text: The summary text
            metadata: Optional metadata dictionary
            embedding: Optional vector embedding
        
        Returns:
            bool: True if successful, False otherwise
        """
        c = self.conn.cursor()
        
        try:
            # Check if summary column exists in conversations table
            c.execute("PRAGMA table_info(conversations)")
            columns = [info[1] for info in c.fetchall()]
            
            # Make sure all needed columns exist
            if 'summary' not in columns:
                c.execute("ALTER TABLE conversations ADD COLUMN summary TEXT")
            
            # Convert metadata to JSON if present
            metadata_json = None
            if metadata:
                metadata_json = json.dumps(metadata)
            
            # Process embedding if provided
            embedding_blob = None
            if embedding is not None:
                import sqlite_vec
                embedding_blob = sqlite_vec.serialize_float32(embedding)
            
            # First check if the conversation exists
            c.execute("SELECT COUNT(*) FROM conversations WHERE id = ?", (conversation_id,))
            if c.fetchone()[0] == 0:
                print(f"Warning: Conversation {conversation_id} does not exist")
                return False
            
            # Update the conversation with the summary
            if metadata_json:
                # Fetch existing metadata if any
                c.execute('SELECT metadata FROM conversations WHERE id = ?', (conversation_id,))
                row = c.fetchone()
                existing_metadata = {}
                
                if row and row[0]:
                    try:
                        existing_metadata = json.loads(row[0])
                    except (json.JSONDecodeError, TypeError):
                        existing_metadata = {}
                
                # Merge new metadata with existing
                if metadata:
                    existing_metadata.update(metadata)
                
                # Update with the merged metadata
                c.execute('UPDATE conversations SET summary = ?, metadata = ? WHERE id = ?', 
                         (summary_text, json.dumps(existing_metadata), conversation_id))
            else:
                # Just update summary
                c.execute('UPDATE conversations SET summary = ? WHERE id = ?', 
                        (summary_text, conversation_id))
            
            # Update embedding if provided
            if embedding_blob:
                # We'll store the conversation embedding in the embeddings table
                embedding_id = f"summary_{conversation_id}"
                self.store_embedding(embedding_id, embedding, model="summary-embedding")
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error storing conversation summary: {e}")
            if self.conn:
                self.conn.rollback()
            return False
            
    def get_active_summary(self):
        """Get the currently active global rolling summary."""
        c = self.conn.cursor()
        
        # Check if relevant columns exist
        c.execute("PRAGMA table_info(rolling_summaries)")
        columns = [info[1] for info in c.fetchall()]
        has_metadata = 'metadata' in columns
        has_embedding = 'embedding' in columns
        has_active = 'is_active' in columns
        
        # Build the query based on available columns
        query = '''
        SELECT id, timestamp, summary_text, conversation_range, version
        '''
        
        if has_metadata:
            query += ", metadata"
        if has_embedding:
            query += ", embedding"
            
        query += " FROM rolling_summaries "
        
        if has_active:
            query += "WHERE is_active = 1 "
            
        query += "ORDER BY timestamp DESC LIMIT 1"
        
        c.execute(query)
        row = c.fetchone()
        
        if not row:
            # If no active summary found, try getting the latest one
            return self.get_latest_summary()
        
        # Get basic fields
        result = {
            'id': row[0],
            'timestamp': row[1],
            'summary_text': row[2],
            'conversation_range': json.loads(row[3]),
            'version': row[4]
        }
        
        # Add additional fields if they exist
        idx = 5
        if has_metadata and len(row) > idx:
            try:
                if row[idx]:
                    result['metadata'] = json.loads(row[idx])
                idx += 1
            except (json.JSONDecodeError, TypeError):
                result['metadata'] = {}
                
        if has_embedding and len(row) > idx:
            if row[idx]:
                # The sqlite_vec module doesn't have deserialize_float32
                # We can load the embedding as a numpy array instead
                import numpy as np
                result['embedding'] = np.frombuffer(row[idx], dtype=np.float32)
        
        return result
        
    def get_latest_summary(self):
        """Get the most recent rolling summary."""
        c = self.conn.cursor()
        
        # Check if relevant columns exist
        c.execute("PRAGMA table_info(rolling_summaries)")
        columns = [info[1] for info in c.fetchall()]
        has_metadata = 'metadata' in columns
        has_embedding = 'embedding' in columns
        
        # Build the query based on available columns
        query = '''
        SELECT id, timestamp, summary_text, conversation_range, version
        '''
        
        if has_metadata:
            query += ", metadata"
        if has_embedding:
            query += ", embedding"
            
        query += " FROM rolling_summaries ORDER BY timestamp DESC LIMIT 1"
        
        c.execute(query)
        row = c.fetchone()
        
        if not row:
            return None
        
        # Get basic fields
        result = {
            'id': row[0],
            'timestamp': row[1],
            'summary_text': row[2],
            'conversation_range': json.loads(row[3]),
            'version': row[4]
        }
        
        # Add additional fields if they exist
        idx = 5
        if has_metadata and len(row) > idx:
            try:
                if row[idx]:
                    result['metadata'] = json.loads(row[idx])
                idx += 1
            except (json.JSONDecodeError, TypeError):
                result['metadata'] = {}
                
        if has_embedding and len(row) > idx:
            if row[idx]:
                # Convert embedding bytes to numpy array
                import numpy as np
                result['embedding'] = np.frombuffer(row[idx], dtype=np.float32)
        
        return result
        
    def get_conversation_summary(self, conversation_id):
        """Get the summary for a specific conversation.
        
        Args:
            conversation_id: The ID of the conversation
            
        Returns:
            dict: Summary data or None if not found
        """
        c = self.conn.cursor()
        
        # Check if the conversation has a summary
        c.execute('''
        SELECT summary, metadata FROM conversations 
        WHERE id = ? AND summary IS NOT NULL
        ''', (conversation_id,))
        
        row = c.fetchone()
        if not row or not row[0]:
            return None
            
        # Get the summary text
        summary_text = row[0]
        
        # Parse metadata if available
        metadata = {}
        if row[1]:
            try:
                metadata = json.loads(row[1])
            except (json.JSONDecodeError, TypeError):
                pass
                
        # Get embedding if available (from embeddings table)
        embedding = None
        embedding_id = f"summary_{conversation_id}"
        
        c.execute('''
        SELECT embedding FROM embeddings
        WHERE id = ?
        ''', (embedding_id,))
        
        embedding_row = c.fetchone()
        if embedding_row and embedding_row[0]:
            # Convert embedding bytes to numpy array
            import numpy as np
            embedding = np.frombuffer(embedding_row[0], dtype=np.float32)
            
        # Build and return summary data
        return {
            'conversation_id': conversation_id,
            'summary_text': summary_text,
            'metadata': metadata,
            'embedding': embedding
        }
    
    def get_summaries_by_theme(self, theme_query, limit=5, embedding=None):
        """Get summaries matching a theme query.
        
        Args:
            theme_query: Text query to search for
            limit: Maximum number of summaries to return
            embedding: Optional query embedding vector for semantic search
            
        Returns:
            list: Matching summaries with relevance scores
        """
        c = self.conn.cursor()
        
        # Check if columns exist
        c.execute("PRAGMA table_info(rolling_summaries)")
        columns = [info[1] for info in c.fetchall()]
        has_metadata = 'metadata' in columns
        has_embedding = 'embedding' in columns
        
        if not has_metadata:
            return []
            
        # If we have embedding data and a query embedding, use vector similarity
        if has_embedding and embedding is not None:
            # Convert embedding to sqlite-vec format
            import sqlite_vec
            embedding_blob = sqlite_vec.serialize_float32(embedding)
            
            # Use vector similarity search
            c.execute('''
            SELECT id, timestamp, summary_text, version, metadata, 
                   vec_distance_cosine(embedding, ?) AS distance
            FROM rolling_summaries
            WHERE embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT ?
            ''', (embedding_blob, limit))
            
            matching_summaries = []
            for row in c.fetchall():
                try:
                    metadata = json.loads(row[4]) if row[4] else {}
                    distance = row[5] if len(row) > 5 else 1.0
                    similarity = 1.0 - distance
                    
                    matching_summaries.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'summary_text': row[2],
                        'version': row[3],
                        'metadata': metadata,
                        'relevance': similarity
                    })
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Error processing summary metadata: {e}")
                    continue
            
            return matching_summaries
        
        # Fallback to metadata-based search if no embeddings
        # First try full-text-search if available
        try:
            # Check if FTS is available by creating a temporary table
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.fts_test USING fts5(content)")
            c.execute("DROP TABLE IF EXISTS temp.fts_test")
            
            # FTS is available, use it for better text search
            matching_summaries = []
            
            # Search in both summary text and metadata
            c.execute('''
            SELECT id, timestamp, summary_text, version, metadata
            FROM rolling_summaries
            WHERE metadata IS NOT NULL
            ORDER BY timestamp DESC
            ''')
            
            for row in c.fetchall():
                try:
                    metadata = json.loads(row[4]) if row[4] else {}
                    themes = metadata.get('themes', [])
                    topics = metadata.get('topics', [])
                    summary_text = row[2]
                    
                    # Create a searchable text combining summary and metadata
                    searchable_text = summary_text + " " + " ".join(themes) + " " + " ".join(topics)
                    
                    # Calculate relevance using term frequency
                    terms = theme_query.lower().split()
                    if not terms:
                        continue
                        
                    # Count how many query terms appear in the searchable text
                    term_count = sum(1 for term in terms if term.lower() in searchable_text.lower())
                    
                    # Calculate relevance score (0-1)
                    relevance = term_count / len(terms) if terms else 0
                    
                    # Only include if it has some relevance
                    if relevance > 0:
                        matching_summaries.append({
                            'id': row[0],
                            'timestamp': row[1],
                            'summary_text': row[2],
                            'version': row[3],
                            'metadata': metadata,
                            'relevance': relevance
                        })
                except (json.JSONDecodeError, TypeError):
                    continue
                    
            # Sort by relevance and limit results
            matching_summaries.sort(key=lambda x: x.get('relevance', 0), reverse=True)
            return matching_summaries[:limit]
            
        except sqlite3.OperationalError:
            # FTS not available, fall back to simple matching
            c.execute('''
            SELECT id, timestamp, summary_text, version, metadata
            FROM rolling_summaries
            WHERE metadata IS NOT NULL
            ORDER BY timestamp DESC
            ''')
            
            matching_summaries = []
            for row in c.fetchall():
                try:
                    metadata = json.loads(row[4]) if row[4] else {}
                    themes = metadata.get('themes', [])
                    topics = metadata.get('topics', [])
                    
                    # Simple string matching search
                    theme_query_lower = theme_query.lower()
                    
                    # Check for matches in themes or topics
                    theme_matches = [theme for theme in themes 
                                   if theme_query_lower in theme.lower()]
                    topic_matches = [topic for topic in topics 
                                   if theme_query_lower in topic.lower()]
                    
                    # Calculate a simple relevance score based on match count
                    relevance = (len(theme_matches) * 2 + len(topic_matches)) / (len(themes) + len(topics)) if themes or topics else 0
                    
                    if theme_matches or topic_matches:
                        matching_summaries.append({
                            'id': row[0],
                            'timestamp': row[1],
                            'summary_text': row[2],
                            'version': row[3],
                            'metadata': metadata,
                            'relevance': min(1.0, relevance)
                        })
                except (json.JSONDecodeError, TypeError):
                    continue
            
            return matching_summaries[:limit]
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
