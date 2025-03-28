# Memory quality assessment - Simplified version that relies primarily on LLM
import os
import json
import re

class MemoryQualityAssessor:
    """
    Assesses and filters memories based on quality metrics.
    
    This simplified implementation relies primarily on LLM evaluation
    with minimal fallback heuristics when LLM evaluation is not available.
    """
    
    def __init__(self, db, gemini_client, quality_threshold=0.6):
        """Initialize the quality assessor.
        
        Args:
            db: Database connection object
            gemini_client: Client for accessing Gemini API
            quality_threshold: Minimum quality score for memory inclusion (0.0-1.0)
        """
        self.db = db
        self.gemini_client = gemini_client
        self.quality_threshold = quality_threshold
        
        # Flag to use LLM for evaluation - defaults to TRUE in this simplified version
        self.use_llm_evaluation = os.environ.get('USE_LLM_QUALITY_EVALUATION', 'true').lower() in ['true', '1', 'yes']
        
        # Debug flag
        self.debug = os.environ.get('DEBUG_QUALITY_ASSESSMENT', '').lower() in ['true', '1', 'yes']
    
    def assess_memory_quality(self, memory_id, content):
        """Assess the quality of a memory primarily using LLM.
        
        Args:
            memory_id: The ID of the memory being assessed
            content: The text content to analyze
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        # Skip empty content
        if not content or not content.strip():
            return 0.1

        # Use LLM as primary method of evaluation (preferred)
        if self.use_llm_evaluation and self.gemini_client:
            llm_metrics = self._evaluate_with_llm(content)
            if llm_metrics:
                # Calculate average of all metrics
                metrics = {
                    'info_density': llm_metrics.get('info_density', 0.5),
                    'coherence': llm_metrics.get('coherence', 0.5),
                    'specificity': llm_metrics.get('specificity', 0.5),
                    'factuality': llm_metrics.get('factuality', 0.5)
                }
                quality_score = sum(metrics.values()) / len(metrics)
                
                # Store the quality score and metrics
                self._update_memory_quality(memory_id, quality_score, metrics)
                
                # Log if debug is enabled
                if self.debug:
                    print(f"LLM quality assessment for memory {memory_id}: {quality_score:.3f}")
                    if 'explanation' in llm_metrics:
                        print(f"Explanation: {llm_metrics['explanation']}")
                
                return quality_score
        
        # Fallback to simple heuristics if LLM evaluation fails or is disabled
        # This is much simpler than the original complex analysis
        return self._fallback_quality_assessment(content, memory_id)
    
    def _fallback_quality_assessment(self, content, memory_id):
        """Simple fallback quality assessment when LLM is not available.
        
        Args:
            content: The content to assess
            memory_id: The memory ID
            
        Returns:
            float: A simple quality score
        """
        if not content or not content.strip():
            return 0.1
            
        # Very basic heuristics for fallback
        words = re.findall(r'\b\w+\b', content)
        word_count = len(words)
        
        # Length-based quality (longer is typically better up to a point)
        if word_count < 20:
            quality_score = 0.3  # Very short content
        elif word_count < 100:
            quality_score = 0.5  # Short content
        elif word_count < 500:
            quality_score = 0.7  # Medium length content
        else:
            quality_score = 0.8  # Long content
            
        # Simple unique words ratio - a very basic information density metric
        unique_words = len(set(words))
        if word_count > 0:
            lexical_diversity = unique_words / word_count
            # Adjust quality based on lexical diversity
            quality_score += (lexical_diversity - 0.5) * 0.2
        
        # Ensure score is within bounds
        quality_score = max(0.1, min(1.0, quality_score))
        
        # Store the quality score
        self._update_memory_quality(memory_id, quality_score)
        
        if self.debug:
            print(f"Fallback quality assessment for memory {memory_id}: {quality_score:.3f}")
            
        return quality_score
    
    def _evaluate_with_llm(self, content):
        """Evaluate content quality dimensions using Gemini LLM.
        
        Args:
            content: The content to evaluate
            
        Returns:
            dict: Metrics from LLM evaluation or None if evaluation failed
        """
        # Prepare a sample of the content (to save tokens)
        words = re.findall(r'\b\w+\b', content)
        if len(words) > 1000:
            first_part = ' '.join(words[:500])
            last_part = ' '.join(words[-500:])
            sample = first_part + "\n\n[...content truncated...]\n\n" + last_part
        else:
            sample = content
            
        # Create evaluation prompt
        prompt = f"""
        Evaluate the quality of the following content on a scale from 0.0 to 1.0 for each dimension:
        
        1. INFORMATION DENSITY: How much useful information is contained in the text?
        2. COHERENCE: How well-structured and logical is the content?
        3. SPECIFICITY: How detailed and concrete is the information?
        4. FACTUALITY: How likely is the content to contain factual information?
        
        CONTENT TO EVALUATE:
        '''
        {sample}
        '''
        
        Provide your evaluation as a JSON object with the format:
        {{
            "info_density": <score>,
            "coherence": <score>,
            "specificity": <score>,
            "factuality": <score>,
            "explanation": "<brief reasoning for scores>"
        }}
        """
        
        try:
            # Make API call
            response = self.gemini_client.generate_content(
                prompt=prompt,
                max_output_tokens=1000,
                temperature=0.1
            )
            
            # Extract and parse JSON from response
            response_text = response.text.strip()
            
            # Clean up response if needed
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '')
                
            response_text = response_text.strip()
            
            try:
                metrics = json.loads(response_text)
                
                # Validate the required keys
                required_keys = ['info_density', 'coherence', 'specificity', 'factuality']
                if all(key in metrics for key in required_keys):
                    # Ensure all scores are within range
                    for key in required_keys:
                        metrics[key] = min(1.0, max(0.0, float(metrics[key])))
                    return metrics
                else:
                    if self.debug:
                        print(f"LLM evaluation missing required metrics: {required_keys}")
                    return None
                    
            except json.JSONDecodeError:
                if self.debug:
                    print(f"Failed to parse LLM evaluation response as JSON: {response_text[:100]}...")
                return None
                
        except Exception as e:
            if self.debug:
                print(f"Error during LLM quality evaluation: {e}")
            return None
    
    def _update_memory_quality(self, memory_id, quality_score, metrics=None):
        """Update the quality score and metrics in the database.
        
        Args:
            memory_id: The ID of the memory to update
            quality_score: The overall quality score
            metrics: Optional dictionary of individual quality dimension scores
        """
        c = self.db.sqlite_conn.cursor()
        
        # Store the overall quality score
        c.execute('UPDATE conversations SET quality_score = ? WHERE id = ?', 
                 (quality_score, memory_id))
        
        # Store detailed metrics if provided
        if metrics:
            # Get existing metadata if any
            c.execute('SELECT metadata_json FROM conversations WHERE id = ?', (memory_id,))
            row = c.fetchone()
            
            existing_metadata = {}
            if row and row['metadata_json']:
                try:
                    existing_metadata = json.loads(row['metadata_json'])
                except (json.JSONDecodeError, TypeError):
                    existing_metadata = {}
            
            # Update or add quality metrics to metadata
            if 'quality_metrics' not in existing_metadata:
                existing_metadata['quality_metrics'] = {}
                
            # Add individual metrics
            existing_metadata['quality_metrics'].update(metrics)
            
            # Store updated metadata
            c.execute('UPDATE conversations SET metadata_json = ? WHERE id = ?',
                     (json.dumps(existing_metadata), memory_id))
        
        self.db.sqlite_conn.commit()
    
    def filter_memories(self, memories):
        """Filter memories based on quality scores.
        
        Args:
            memories: List of memory objects with 'source_id' and 'similarity' fields
            
        Returns:
            list: Filtered and sorted memories with added 'quality_score' field
        """
        filtered_memories = []
        
        for memory in memories:
            memory_id = memory['source_id']
            
            # Get stored quality score
            c = self.db.sqlite_conn.cursor()
            c.execute('SELECT quality_score, metadata_json FROM conversations WHERE id = ?', (memory_id,))
            row = c.fetchone()
            
            if row and row['quality_score'] is not None:
                # Use stored quality score
                quality_score = row['quality_score']
                
                # Add detailed metrics if available
                if row['metadata_json']:
                    try:
                        metadata = json.loads(row['metadata_json'])
                        if 'quality_metrics' in metadata:
                            memory['quality_metrics'] = metadata['quality_metrics']
                    except (json.JSONDecodeError, TypeError):
                        pass
            else:
                # Get content to assess quality
                c.execute('''
                SELECT content FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
                ''', (memory_id,))
                
                messages = c.fetchall()
                content = "\n".join([msg['content'] for msg in messages])
                
                # Assess quality if not already scored
                quality_score = self.assess_memory_quality(memory_id, content)
            
            # Apply quality threshold
            if quality_score >= self.quality_threshold:
                memory['quality_score'] = quality_score
                filtered_memories.append(memory)
        
        # Sort by combination of similarity and quality
        similarity_weight = float(os.environ.get('QUALITY_SIMILARITY_WEIGHT', '0.7'))
        quality_weight = float(os.environ.get('QUALITY_SCORE_WEIGHT', '0.3'))
        
        # Normalize weights
        total_weight = similarity_weight + quality_weight
        if total_weight > 0:
            similarity_weight /= total_weight
            quality_weight /= total_weight
        else:
            similarity_weight = quality_weight = 0.5
            
        filtered_memories.sort(
            key=lambda x: (x['similarity'] * similarity_weight) + (x.get('quality_score', 0) * quality_weight), 
            reverse=True
        )
        
        # Limit the number of memories if specified
        max_memories = int(os.environ.get('MAX_FILTERED_MEMORIES', '10'))
        if max_memories > 0 and len(filtered_memories) > max_memories:
            filtered_memories = filtered_memories[:max_memories]
        
        return filtered_memories
    
    def update_quality_from_usage(self, memory_id, was_helpful, feedback=None):
        """Update quality score based on usage feedback.
        
        Args:
            memory_id: The ID of the memory to update
            was_helpful: Boolean indicating if the memory was helpful
            feedback: Optional dictionary with detailed feedback
        """
        # Get current score and metadata
        c = self.db.sqlite_conn.cursor()
        c.execute('SELECT quality_score, metadata_json FROM conversations WHERE id = ?', (memory_id,))
        row = c.fetchone()
        
        if not row:
            return
            
        current_score = row['quality_score'] or 0.5  # Default if NULL
        
        # Simple adjustment based on feedback
        adjustment = 0.05 if was_helpful else -0.05
            
        # Calculate new score with bounds
        new_score = max(0.1, min(1.0, current_score + adjustment))
        
        # Update metadata with feedback history
        metadata = {}
        if row['metadata_json']:
            try:
                metadata = json.loads(row['metadata_json'])
            except (json.JSONDecodeError, TypeError):
                metadata = {}
                
        # Add feedback to history
        if 'feedback_history' not in metadata:
            metadata['feedback_history'] = []
            
        # Add this feedback instance
        from datetime import datetime
        feedback_entry = {
            'timestamp': datetime.now().timestamp(),
            'was_helpful': was_helpful,
            'score_before': current_score,
            'score_after': new_score
        }
        
        # Add detailed feedback if provided
        if feedback:
            feedback_entry['details'] = feedback
            
        metadata['feedback_history'].append(feedback_entry)
        
        # Update the score and metadata
        c.execute('UPDATE conversations SET quality_score = ?, metadata_json = ? WHERE id = ?', 
                 (new_score, json.dumps(metadata), memory_id))
        self.db.sqlite_conn.commit()
        
        if self.debug:
            print(f"Updated quality score for memory {memory_id}: {current_score:.3f} -> {new_score:.3f} (helpful: {was_helpful})")
