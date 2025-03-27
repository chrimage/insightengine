# Memory quality assessment

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
        # Simple heuristic based on length and token count
        lines = content.strip().split('\n')
        word_count = sum(len(line.split()) for line in lines)
        unique_words = len(set(word.lower() for line in lines for word in line.split()))
        
        # More words and higher uniqueness ratio are better
        if word_count < 10:
            return 0.3  # Very short content
        
        uniqueness_ratio = unique_words / word_count if word_count > 0 else 0
        
        # Scale score - longer content with more unique words is better
        density_score = min(1.0, (word_count / 500) * 0.6 + uniqueness_ratio * 0.4)
        return max(0.1, density_score)  # Ensure minimum score
    
    def _calculate_coherence(self, content):
        """Calculate coherence score."""
        # For now, use a simplified approach based on structure
        lines = content.strip().split('\n')
        
        # Remove empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return 0.1
        
        # Heuristic: conversations with clear turns (user/assistant alternating) are better
        speaker_changes = 0
        current_speaker = None
        
        for line in non_empty_lines:
            if line.startswith('USER:') or line.startswith('ASSISTANT:'):
                speaker = line.split(':')[0]
                if current_speaker and speaker != current_speaker:
                    speaker_changes += 1
                current_speaker = speaker
        
        # More speaker changes indicate a coherent conversation
        coherence_score = min(1.0, speaker_changes / 10 * 0.8 + 0.2)
        return coherence_score
    
    def _calculate_specificity(self, content):
        """Calculate specificity score."""
        # Simple heuristic looking for specific markers in content
        specificity_markers = [
            r'\d{4}',  # Years
            r'\d+\s*(dollar|euro|\$|€)',  # Monetary values
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Dates
            r'\d{1,2}:\d{2}',  # Times
            r'https?://',  # URLs
            r'@[a-zA-Z0-9_]+',  # Handles
            r'[A-Z][a-z]+ [A-Z][a-z]+',  # Proper names (simplistic)
        ]
        
        # Check for presence of specificity markers
        marker_count = 0
        import re
        for marker in specificity_markers:
            matches = re.findall(marker, content)
            marker_count += len(matches)
        
        # Scale to a score
        specificity_score = min(1.0, marker_count / 10 * 0.7 + 0.3)
        return specificity_score
    
    def _estimate_factuality(self, content):
        """Estimate factuality score."""
        # This is a placeholder - in a real system we'd use more sophisticated methods
        # For now, return a moderate score
        return 0.7
    
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