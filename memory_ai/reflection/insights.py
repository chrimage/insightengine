# Wisdom extraction and application

import os
import uuid
import json
import time
from datetime import datetime

class InsightRepository:
    """
    Repository for storing and retrieving insights from self-reflection.
    
    This component:
    1. Stores insights extracted from response evaluations
    2. Tracks insight performance metrics over time
    3. Retrieves relevant insights using vector similarity
    4. Clusters and consolidates related insights
    5. Provides analytics on insight efficacy
    
    Insights are the key learning mechanism of the self-reflection system,
    allowing the AI to continuously improve based on its own experiences.
    """
    
    def __init__(self, db, gemini_client=None):
        """
        Initialize the insight repository.
        
        Args:
            db: Database connection
            gemini_client: Optional client for generating embeddings
        """
        self.db = db
        self.gemini_client = gemini_client
        
        # Configuration
        self.use_embeddings = (
            self.gemini_client is not None and
            os.environ.get('USE_EMBEDDINGS_FOR_INSIGHTS', '').lower() in ['true', '1', 'yes']
        )
        self.auto_consolidate = os.environ.get('AUTO_CONSOLIDATE_INSIGHTS', '').lower() in ['true', '1', 'yes']
        self.consolidation_threshold = float(os.environ.get('INSIGHT_CONSOLIDATION_THRESHOLD', '0.85'))
        self.debug_mode = os.environ.get('DEBUG_INSIGHTS', '').lower() in ['true', '1', 'yes']
        
        # Initialize embeddings cache
        self.embedding_cache = {}
        
        # Initialize insight metrics
        self._load_insight_metrics()
    
    def add_insight(self, text, category, confidence, evidence=None):
        """
        Add a new insight to the repository.
        
        Args:
            text: The insight text
            category: The insight category
            confidence: Initial confidence score (0.0-1.0)
            evidence: List of evidence dictionaries supporting this insight
            
        Returns:
            str: The ID of the new insight
        """
        insight_id = str(uuid.uuid4())
        now = datetime.now().timestamp()
        
        # Check for duplicates before adding
        if self._check_for_duplicates(text, category):
            if self.debug_mode:
                print(f"Skipping duplicate insight: {category} - {text[:50]}...")
            return None
            
        # Generate embedding if enabled
        embedding_blob = None
        if self.use_embeddings and self.gemini_client:
            try:
                embedding = self.gemini_client.generate_embeddings(text)
                if embedding:
                    import sqlite_vec
                    embedding_blob = sqlite_vec.serialize_float32(embedding)
                    # Cache the embedding
                    self.embedding_cache[insight_id] = embedding
            except Exception as e:
                if self.debug_mode:
                    print(f"Error generating embedding for insight: {e}")
        
        try:
            c = self.db.conn.cursor()
            c.execute('''
            INSERT INTO insights
            (id, text, category, confidence, created_at, updated_at, evidence, 
             application_count, success_rate, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                insight_id,
                text,
                category,
                confidence,
                now,
                now,
                json.dumps(evidence or []),
                0,
                0.0,
                embedding_blob
            ))
            
            self.db.conn.commit()
            
            # Add to metrics tracking
            self.insight_metrics.setdefault(category, {
                'count': 0,
                'avg_confidence': 0,
                'avg_success_rate': 0
            })
            self.insight_metrics[category]['count'] += 1
            
            # Check for consolidation opportunities
            if self.auto_consolidate and self.use_embeddings:
                self._check_for_consolidation(insight_id, text, category, embedding)
                
            return insight_id
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error adding insight: {e}")
            return None
    
    def get_relevant_insights(self, query, top_k=3, category=None, min_confidence=0.0, min_success_rate=0.0):
        """
        Get insights relevant to the query.
        
        Args:
            query: The query to find relevant insights for
            top_k: Maximum number of insights to return
            category: Optional category filter
            min_confidence: Minimum confidence threshold
            min_success_rate: Minimum success rate threshold
            
        Returns:
            list: Relevant insights with similarity scores
        """
        if self.use_embeddings and self.gemini_client:
            return self._get_insights_by_embedding(query, top_k, category, min_confidence, min_success_rate)
        else:
            return self._get_insights_by_keywords(query, top_k, category, min_confidence, min_success_rate)
            
    def _get_insights_by_embedding(self, query, top_k=3, category=None, min_confidence=0.0, min_success_rate=0.0):
        """
        Retrieve insights using vector similarity.
        
        This method uses embeddings to find semantically similar insights, which is
        much more powerful than simple keyword matching.
        """
        try:
            # Generate query embedding
            query_embedding = self.gemini_client.generate_embeddings(query)
            if not query_embedding:
                # Fallback to keyword matching if embedding generation fails
                if self.debug_mode:
                    print("Embedding generation failed, falling back to keyword matching")
                return self._get_insights_by_keywords(query, top_k, category, min_confidence, min_success_rate)
                
            # Prepare query parameters
            params = [min_confidence, min_success_rate]
            
            # Build SQL query with filters
            sql = '''
            SELECT id, text, category, confidence, application_count, success_rate,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM insights
            WHERE confidence >= ? AND (application_count = 0 OR success_rate >= ?)
            '''
            
            # Add category filter if specified
            if category:
                sql += ' AND category = ?'
                params.append(category)
                
            # Complete the query with sorting and limit
            sql += '''
            ORDER BY distance ASC
            LIMIT ?
            '''
            params.append(query_embedding)
            params.append(top_k)
            
            # Execute the query
            import sqlite_vec
            query_embedding_blob = sqlite_vec.serialize_float32(query_embedding)
            
            c = self.db.conn.cursor()
            c.execute(sql, [query_embedding_blob] + params)
            
            # Process results
            relevant_insights = []
            for row in c.fetchall():
                insight = dict(row)
                
                # Convert distance to similarity score (1 - distance)
                distance = insight.pop('distance', 0)
                similarity = 1 - distance
                
                # Add similarity score
                insight['relevance'] = similarity
                relevant_insights.append(insight)
                
            return relevant_insights
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error in vector search: {e}")
            # Fallback to keyword matching
            return self._get_insights_by_keywords(query, top_k, category, min_confidence, min_success_rate)
    
    def _get_insights_by_keywords(self, query, top_k=3, category=None, min_confidence=0.0, min_success_rate=0.0):
        """
        Retrieve insights using keyword matching.
        
        This is a fallback method when embeddings are not available or failed.
        """
        try:
            c = self.db.conn.cursor()
            
            # Build SQL with filters
            sql = '''
            SELECT id, text, category, confidence, application_count, success_rate
            FROM insights
            WHERE confidence >= ? AND (application_count = 0 OR success_rate >= ?)
            '''
            params = [min_confidence, min_success_rate]
            
            # Add category filter if specified
            if category:
                sql += ' AND category = ?'
                params.append(category)
                
            # Sort by confidence and success rate
            sql += '''
            ORDER BY confidence DESC, success_rate DESC
            '''
            
            # Execute query
            c.execute(sql, params)
            all_insights = [dict(row) for row in c.fetchall()]
            
            # Filter for relevance using keyword matching
            relevant_insights = []
            query_words = set(query.lower().split())
            
            for insight in all_insights:
                insight_words = set(insight['text'].lower().split())
                
                # Calculate relevance based on word overlap
                overlap = len(query_words.intersection(insight_words))
                if overlap > 0:
                    # Weight by word overlap relative to insight length
                    word_overlap_score = overlap / max(1, len(insight_words))
                    
                    # Full text similarity (crude but effective)
                    text_similarity = self._calculate_text_similarity(query, insight['text'])
                    
                    # Combined relevance score
                    relevance = (word_overlap_score * 0.4) + (text_similarity * 0.6)
                    insight['relevance'] = relevance
                    relevant_insights.append(insight)
            
            # Sort insights by combined score
            relevant_insights.sort(
                key=lambda x: (
                    x.get('relevance', 0) * 0.5 + 
                    x['confidence'] * 0.3 + 
                    (x.get('success_rate', 0) * 0.2)
                ),
                reverse=True
            )
            
            return relevant_insights[:top_k]
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error in keyword search: {e}")
            return []
            
    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate simple text similarity score.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        # Convert to lowercase for case-insensitive comparison
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Extract words, removing punctuation
        import re
        words1 = set(re.findall(r'\b\w+\b', text1))
        words2 = set(re.findall(r'\b\w+\b', text2))
        
        # Avoid division by zero
        if not words1 or not words2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union
        
    def _check_for_duplicates(self, text, category):
        """
        Check if a very similar insight already exists.
        
        Returns True if a duplicate is found, False otherwise.
        """
        # Use embeddings for semantic duplicate detection if available
        if self.use_embeddings and self.gemini_client:
            try:
                # Generate embedding for the new insight
                new_embedding = self.gemini_client.generate_embeddings(text)
                if not new_embedding:
                    return False
                    
                # Get existing insights in the same category
                c = self.db.conn.cursor()
                c.execute('''
                SELECT id, text, embedding
                FROM insights
                WHERE category = ?
                ''', (category,))
                
                # Check each existing insight for similarity
                import sqlite_vec
                threshold = 0.92  # High threshold for considering as duplicate
                
                for row in c.fetchall():
                    if row[2]:  # Has embedding
                        existing_embedding = sqlite_vec.deserialize_float32(row[2])
                        
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(new_embedding, existing_embedding)
                        
                        if similarity >= threshold:
                            return True
                
                return False
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Error checking duplicates: {e}")
                return False
        else:
            # Fallback to basic text similarity
            c = self.db.conn.cursor()
            c.execute('SELECT text FROM insights WHERE category = ?', (category,))
            
            for row in c.fetchall():
                similarity = self._calculate_text_similarity(text, row[0])
                if similarity > 0.8:  # High threshold for text similarity
                    return True
                    
            return False
    
    def update_insight_performance(self, insight_id, was_successful, feedback=None):
        """
        Update the performance metrics for an insight.
        
        Args:
            insight_id: The ID of the insight to update
            was_successful: Boolean indicating whether the insight led to success
            feedback: Optional feedback data to store with the update
        """
        c = self.db.conn.cursor()
        
        # Get current metrics and category
        c.execute('''
        SELECT application_count, success_rate, category, confidence
        FROM insights
        WHERE id = ?
        ''', (insight_id,))
        
        row = c.fetchone()
        if not row:
            if self.debug_mode:
                print(f"Cannot update performance for non-existent insight: {insight_id}")
            return
            
        application_count, success_rate, category, confidence = row
        
        # Update metrics
        new_count = application_count + 1
        success_count = (application_count * success_rate) + (1 if was_successful else 0)
        new_rate = success_count / new_count
        
        # Adjust confidence based on performance
        # Higher success rates lead to higher confidence
        confidence_adjustment = 0.0
        
        if new_count >= 3:  # Only adjust confidence after enough applications
            if new_rate > 0.8:
                # Very successful, boost confidence
                confidence_adjustment = 0.05
            elif new_rate < 0.3:
                # Mostly unsuccessful, decrease confidence
                confidence_adjustment = -0.05
            elif new_rate < 0.5:
                # Marginally unsuccessful, slight decrease
                confidence_adjustment = -0.02
        
        # Apply bounded adjustment
        new_confidence = max(0.1, min(1.0, confidence + confidence_adjustment))
        
        # Prepare feedback data
        update_data = {
            'timestamp': datetime.now().timestamp(),
            'was_successful': was_successful,
            'previous_rate': success_rate,
            'new_rate': new_rate,
            'application_count': new_count
        }
        
        if feedback:
            update_data['feedback'] = feedback
            
        # Get existing applications data
        c.execute('SELECT applications FROM insights WHERE id = ?', (insight_id,))
        row = c.fetchone()
        applications = []
        
        if row and row[0]:
            try:
                applications = json.loads(row[0])
            except json.JSONDecodeError:
                applications = []
        
        # Add new application data
        applications.append(update_data)
        
        # Save updated metrics
        c.execute('''
        UPDATE insights
        SET application_count = ?, 
            success_rate = ?, 
            confidence = ?,
            updated_at = ?,
            applications = ?
        WHERE id = ?
        ''', (
            new_count, 
            new_rate, 
            new_confidence,
            datetime.now().timestamp(), 
            json.dumps(applications),
            insight_id
        ))
        
        self.db.conn.commit()
        
        # Update metrics tracking
        if category in self.insight_metrics:
            cat_metrics = self.insight_metrics[category]
            
            # Recalculate category average success rate
            total_apps = cat_metrics.get('total_applications', 0) + 1
            cat_metrics['total_applications'] = total_apps
            
            # Update success count
            success_count = cat_metrics.get('success_count', 0)
            if was_successful:
                success_count += 1
            cat_metrics['success_count'] = success_count
            
            # Calculate new average
            cat_metrics['avg_success_rate'] = success_count / total_apps if total_apps > 0 else 0.0
            
        # Log if in debug mode
        if self.debug_mode:
            result = "successful" if was_successful else "unsuccessful"
            print(f"Updated insight {insight_id}: application {result}, " +
                  f"rate {success_rate:.2f} → {new_rate:.2f}, " +
                  f"confidence {confidence:.2f} → {new_confidence:.2f}")
    
    def _load_insight_metrics(self):
        """Load insight metrics from the database."""
        self.insight_metrics = {}
        
        try:
            c = self.db.conn.cursor()
            
            # Get metrics by category
            c.execute('''
            SELECT category, COUNT(*) as count, AVG(confidence) as avg_confidence, 
                   AVG(success_rate) as avg_success_rate,
                   SUM(application_count) as total_applications
            FROM insights
            GROUP BY category
            ''')
            
            for row in c.fetchall():
                category, count, avg_confidence, avg_success_rate, total_applications = row
                
                # Calculate total successes
                c.execute('''
                SELECT SUM(application_count * success_rate) 
                FROM insights 
                WHERE category = ?
                ''', (category,))
                
                success_result = c.fetchone()
                success_count = success_result[0] if success_result and success_result[0] is not None else 0
                
                self.insight_metrics[category] = {
                    'count': count,
                    'avg_confidence': avg_confidence or 0.0,
                    'avg_success_rate': avg_success_rate or 0.0,
                    'total_applications': total_applications or 0,
                    'success_count': success_count
                }
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error loading insight metrics: {e}")
            # Initialize with empty metrics
            self.insight_metrics = {}
    
    def _cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector as list or numpy array
            vec2: Second vector as list or numpy array
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        try:
            import numpy as np
            
            # Convert to numpy arrays if needed
            if not isinstance(vec1, np.ndarray):
                vec1 = np.array(vec1)
            if not isinstance(vec2, np.ndarray):
                vec2 = np.array(vec2)
                
            # Calculate dot product and magnitudes
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            # Return cosine similarity
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error calculating cosine similarity: {e}")
            return 0.0
            
    def _check_for_consolidation(self, new_insight_id, text, category, embedding):
        """
        Check if similar insights can be consolidated.
        
        This helps prevent the repository from becoming cluttered with
        many similar insights.
        """
        try:
            if not embedding:
                return
                
            # Find similar insights
            c = self.db.conn.cursor()
            c.execute('''
            SELECT id, text, confidence, application_count, success_rate, embedding
            FROM insights
            WHERE category = ? AND id != ?
            ''', (category, new_insight_id))
            
            consolidation_candidates = []
            
            for row in c.fetchall():
                if not row[5]:  # No embedding
                    continue
                    
                # Deserialize embedding
                import sqlite_vec
                existing_embedding = sqlite_vec.deserialize_float32(row[5])
                
                # Calculate similarity
                similarity = self._cosine_similarity(embedding, existing_embedding)
                
                # Check if above consolidation threshold
                if similarity >= self.consolidation_threshold:
                    consolidation_candidates.append({
                        'id': row[0],
                        'text': row[1],
                        'confidence': row[2],
                        'application_count': row[3],
                        'success_rate': row[4],
                        'similarity': similarity
                    })
            
            # If we have candidates, consolidate with the most similar one
            if consolidation_candidates:
                # Sort by similarity
                consolidation_candidates.sort(key=lambda x: x['similarity'], reverse=True)
                target = consolidation_candidates[0]
                
                if self.debug_mode:
                    print(f"Consolidating insight {new_insight_id} with {target['id']}")
                    print(f"Similarity: {target['similarity']:.3f}")
                    print(f"New: {text[:50]}...")
                    print(f"Existing: {target['text'][:50]}...")
                
                # Perform consolidation
                self._consolidate_insights(new_insight_id, target['id'])
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error checking for consolidation: {e}")
    
    def _consolidate_insights(self, source_id, target_id):
        """
        Consolidate two similar insights.
        
        Args:
            source_id: The ID of the insight to merge
            target_id: The ID of the insight to merge into
        """
        try:
            c = self.db.conn.cursor()
            
            # Get both insights
            c.execute('''
            SELECT id, text, confidence, evidence, application_count, success_rate
            FROM insights
            WHERE id IN (?, ?)
            ''', (source_id, target_id))
            
            insights = [dict(row) for row in c.fetchall()]
            if len(insights) != 2:
                return
                
            source = next(i for i in insights if i['id'] == source_id)
            target = next(i for i in insights if i['id'] == target_id)
            
            # Merge evidence
            try:
                source_evidence = json.loads(source['evidence']) if source['evidence'] else []
                target_evidence = json.loads(target['evidence']) if target['evidence'] else []
                merged_evidence = target_evidence + source_evidence
            except json.JSONDecodeError:
                merged_evidence = []
            
            # Calculate new metrics
            total_apps = source['application_count'] + target['application_count']
            
            if total_apps > 0:
                # Weighted success rate
                source_successes = source['application_count'] * source['success_rate']
                target_successes = target['application_count'] * target['success_rate']
                new_rate = (source_successes + target_successes) / total_apps
            else:
                new_rate = (source['success_rate'] + target['success_rate']) / 2
                
            # Choose higher confidence
            new_confidence = max(source['confidence'], target['confidence'])
            
            # Update target insight
            c.execute('''
            UPDATE insights
            SET evidence = ?,
                application_count = ?,
                success_rate = ?,
                confidence = ?,
                updated_at = ?
            WHERE id = ?
            ''', (
                json.dumps(merged_evidence),
                total_apps,
                new_rate,
                new_confidence,
                datetime.now().timestamp(),
                target_id
            ))
            
            # Delete source insight
            c.execute('DELETE FROM insights WHERE id = ?', (source_id,))
            
            self.db.conn.commit()
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error consolidating insights: {e}")
    
    def get_insight_metrics(self, category=None):
        """
        Get metrics about insights.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            dict: Metrics about insights
        """
        if category:
            return self.insight_metrics.get(category, {
                'count': 0,
                'avg_confidence': 0.0,
                'avg_success_rate': 0.0,
                'total_applications': 0,
                'success_count': 0
            })
        else:
            # Aggregate metrics across all categories
            total_count = sum(m['count'] for m in self.insight_metrics.values())
            total_apps = sum(m['total_applications'] for m in self.insight_metrics.values())
            success_count = sum(m['success_count'] for m in self.insight_metrics.values())
            
            # Calculate weighted averages
            if total_count > 0:
                avg_confidence = sum(m['count'] * m['avg_confidence'] for m in self.insight_metrics.values()) / total_count
            else:
                avg_confidence = 0.0
                
            if total_apps > 0:
                avg_success_rate = success_count / total_apps
            else:
                avg_success_rate = 0.0
                
            return {
                'count': total_count,
                'avg_confidence': avg_confidence,
                'avg_success_rate': avg_success_rate,
                'total_applications': total_apps,
                'success_count': success_count,
                'categories': {cat: metrics['count'] for cat, metrics in self.insight_metrics.items()}
            }
    
    def get_insights_by_category(self, category, limit=100, min_confidence=0.0):
        """
        Get insights filtered by category.
        
        Args:
            category: The category to filter by
            limit: Maximum number of insights to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            list: Matching insights
        """
        try:
            c = self.db.conn.cursor()
            c.execute('''
            SELECT id, text, confidence, application_count, success_rate, created_at
            FROM insights
            WHERE category = ? AND confidence >= ?
            ORDER BY confidence DESC, success_rate DESC
            LIMIT ?
            ''', (category, min_confidence, limit))
            
            return [dict(row) for row in c.fetchall()]
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error retrieving insights by category: {e}")
            return []
    
    def get_top_insights(self, limit=10, min_applications=3):
        """
        Get the most successful insights.
        
        Args:
            limit: Maximum number of insights to return
            min_applications: Minimum number of applications to be considered
            
        Returns:
            list: Top performing insights
        """
        try:
            c = self.db.conn.cursor()
            c.execute('''
            SELECT id, text, category, confidence, application_count, success_rate
            FROM insights
            WHERE application_count >= ?
            ORDER BY success_rate DESC, application_count DESC
            LIMIT ?
            ''', (min_applications, limit))
            
            return [dict(row) for row in c.fetchall()]
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error retrieving top insights: {e}")
            return []