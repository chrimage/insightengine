# Vector-based memory retrieval

from datetime import datetime

class MemoryRetriever:
    """Retrieves memories based on vector similarity."""
    
    def __init__(self, db, gemini_client):
        """Initialize the memory retriever."""
        self.db = db
        self.gemini_client = gemini_client
        self.quality_assessor = None
        
    def set_quality_assessor(self, quality_assessor):
        """Set the quality assessor for filtering results."""
        self.quality_assessor = quality_assessor
    
    def retrieve_memories(self, query, top_k=5):
        """Retrieve relevant memories based on the query."""
        # Generate embedding for the query
        query_embedding = self.gemini_client.generate_embeddings(query)
        
        # Search for similar memories
        similar_memories = self.db.search_similar(query_embedding, top_k=top_k*2)  # Get more than needed for filtering
        
        # Apply quality filtering if available
        if self.quality_assessor and similar_memories:
            filtered_memories = self.quality_assessor.filter_memories(similar_memories)
            return filtered_memories[:top_k]  # Limit to requested number
        
        return similar_memories[:top_k]
    
    def retrieve_by_date_range(self, start_date, end_date, limit=10):
        """Retrieve memories from a specific date range."""
        c = self.db.sqlite_conn.cursor()
        
        # Convert dates to timestamps
        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()
        
        # Query conversations in the date range
        c.execute('''
        SELECT id, title, timestamp, model
        FROM conversations
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (start_ts, end_ts, limit))
        
        results = []
        for row in c.fetchall():
            results.append({
                'source_id': row['id'],
                'title': row['title'],
                'timestamp': row['timestamp'],
                'model': row['model']
            })
        
        return results