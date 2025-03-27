# Wisdom extraction and application

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