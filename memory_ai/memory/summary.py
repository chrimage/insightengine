# Rolling summary implementation

import uuid
from datetime import datetime
import tiktoken
from memory_ai.core.database import MemoryDatabase
from memory_ai.utils.gemini import GeminiClient

class RollingSummaryProcessor:
    """Processes conversations to generate rolling summaries."""
    
    def __init__(self, db, gemini_client, batch_size=20, max_tokens=8000):
        """Initialize the rolling summary processor."""
        self.db = db
        self.gemini_client = gemini_client
        self.batch_size = batch_size  # Increased from 10 to 20 for better efficiency
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.metadata = {}
    
    def process_conversations(self):
        """Process all conversations to generate rolling summaries."""
        # Get conversations ordered by timestamp
        conversations = self._get_chronological_conversations()
        
        # Get the latest summary (if any)
        latest_summary = self.db.get_latest_summary()
        current_summary_text = latest_summary['summary_text'] if latest_summary else None
        processed_convs = set(latest_summary['conversation_range'] if latest_summary else [])
        
        # Count unprocessed conversations
        unprocessed_count = sum(1 for conv in conversations if conv['id'] not in processed_convs)
        if unprocessed_count > 0:
            print(f"Found {unprocessed_count} unprocessed conversations")
        else:
            print("All conversations have been processed already")
            return
        
        # Apply quality filtering to prioritize high-quality conversations
        quality_filtered_convs = self._apply_quality_filter(
            [conv for conv in conversations if conv['id'] not in processed_convs]
        )
        
        # Process conversations in batches
        batches = [quality_filtered_convs[i:i+self.batch_size] 
                   for i in range(0, len(quality_filtered_convs), self.batch_size)]
        
        total_batches = len(batches)
        for batch_idx, batch in enumerate(batches):
            print(f"Processing batch {batch_idx+1}/{total_batches} with {len(batch)} conversations")
            
            # Prepare batch content
            batch_content = self._format_batch_content(batch)
            
            # Generate updated summary
            new_summary = self._generate_summary(batch_content, current_summary_text)
            
            # Extract themes and topics using text analysis
            themes, topics = self._extract_themes_and_topics(new_summary)
            
            # Store the new summary
            summary_id = str(uuid.uuid4())
            version = (latest_summary['version'] + 1) if latest_summary else 1
            
            # Update processed conversations
            for conv in batch:
                processed_convs.add(conv['id'])
            
            # Create summary data with metadata
            summary_data = {
                'id': summary_id,
                'timestamp': datetime.now().timestamp(),
                'summary_text': new_summary,
                'conversation_range': list(processed_convs),
                'version': version,
                'metadata': {
                    'themes': themes,
                    'topics': topics,
                    'quality_score': self._calculate_summary_quality(new_summary),
                    'batch_size': len(batch),
                    'total_processed': len(processed_convs)
                }
            }
            
            self.db.store_rolling_summary(summary_data)
            
            # Update current summary for next iteration
            current_summary_text = new_summary
            latest_summary = summary_data
            
            print(f"Created summary version {version} covering {len(processed_convs)} conversations")
    
    def _apply_quality_filter(self, conversations):
        """Filter and prioritize conversations based on quality while preserving chronological ordering."""
        # If we have less than 100 conversations, don't filter
        if len(conversations) < 100:
            return conversations
        
        # Calculate quality score for each conversation
        for conv in conversations:
            # Default quality score of 0.5 if not set
            quality = conv.get('quality_score') or 0.5
            # Recency factor (0-1) based on timestamp
            max_time = max(c['timestamp'] for c in conversations)
            min_time = min(c['timestamp'] for c in conversations)
            time_range = max(1, max_time - min_time)  # Avoid division by zero
            recency = (conv['timestamp'] - min_time) / time_range
            # Combined score with heavier weight on quality
            conv['_quality_combined_score'] = (quality * 0.7) + (recency * 0.3)
        
        # Sort by combined quality score to identify top conversations
        quality_sorted = sorted(conversations, key=lambda c: c.get('_quality_combined_score', 0.5), reverse=True)
        
        # Select top 80% of conversations by quality
        cutoff = int(len(quality_sorted) * 0.8)
        top_quality_convs = quality_sorted[:cutoff]
        
        # Get IDs of top quality conversations
        top_ids = {conv['id'] for conv in top_quality_convs}
        
        # Filter original conversations to keep only high quality ones, preserving chronological order
        filtered_chronological = [conv for conv in conversations if conv['id'] in top_ids]
        
        # Sort by timestamp to ensure chronological order
        return sorted(filtered_chronological, key=lambda c: c['timestamp'])
    
    def _extract_themes_and_topics(self, summary_text):
        """Extract themes and topics from the generated summary."""
        themes = []
        topics = []
        
        # Simple keyword extraction for themes
        theme_indicators = ["CORE INTERESTS", "TECHNICAL EXPERTISE", "RECURRING TOPICS", 
                           "THEME:", "INTEREST:", "EXPERTISE:"]
        
        for line in summary_text.split('\n'):
            line = line.strip()
            # Check for theme sections
            for indicator in theme_indicators:
                if indicator in line:
                    cleaned = line.replace(indicator, "").strip().strip(':').strip()
                    if cleaned and len(cleaned) > 3:  # Avoid empty or too short themes
                        themes.append(cleaned)
            
            # Check for bullet points which likely contain topics
            if line.startswith('- ') or line.startswith('* '):
                topic = line[2:].split(':')[0].strip()
                if topic and len(topic) > 3 and not any(c in topic for c in ['(', ')', '[', ']']):
                    topics.append(topic)
        
        # Deduplicate and limit to reasonable number
        themes = list(set(themes))[:10]
        topics = list(set(topics))[:20]
        
        return themes, topics
    
    def _calculate_summary_quality(self, summary_text):
        """Calculate a quality score for the summary."""
        # Basic metrics for summary quality
        length_score = min(1.0, len(summary_text) / 5000)  # Length up to 5000 chars
        
        # Structure score - check for expected sections
        expected_sections = ["CORE INTERESTS", "TECHNICAL EXPERTISE", "RECURRING TOPICS"]
        section_score = sum(1 for section in expected_sections if section in summary_text) / len(expected_sections)
        
        # Format score - check for bullet points
        bullet_points = summary_text.count('\n- ') + summary_text.count('\n* ')
        format_score = min(1.0, bullet_points / 20)  # At least 20 bullet points is ideal
        
        # Combined score
        return (length_score * 0.3) + (section_score * 0.4) + (format_score * 0.3)
    
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
            You are creating an insightful, structured memory system for an AI assistant. Your task is to maintain a comprehensive yet usable summary of all user conversations.

            Below is the previous summary followed by new conversations to integrate:

            <<<PREVIOUS_SUMMARY>>>
            {previous_summary}
            <<<END_PREVIOUS_SUMMARY>>>

            <<<NEW_CONVERSATIONS>>>
            {batch_content}
            <<<END_NEW_CONVERSATIONS>>>

            Generate an updated summary with these characteristics:
            
            1. DO NOT include any of the prompt markers (like "PREVIOUS_SUMMARY" or "END_NEW_CONVERSATIONS") in your response
            2. DO NOT preface your response with phrases like "Here's the updated summary" or "Summary:"
            3. Focus on identifying KEY INSIGHTS and PATTERNS, not detailed recaps of every conversation
            4. Begin with a "CORE INTERESTS" section listing 3-7 main areas of interest shown across conversations
            5. Group information by THEMES (technical subjects, creative projects, personal interests) rather than chronology
            6. For each theme, provide bullet points of relevant insights, preferences, and expertise
            7. Add a "TECHNICAL EXPERTISE" section identifying areas where the user has demonstrated knowledge
            8. Include a "RECURRING TOPICS" section for subjects that appear multiple times
            9. When information is updated or contradicted, retain only the latest version
            10. For important timestamps (project deadlines, significant events), use the format YYYY-MM-DD

            The summary should be immediately useful for an AI to understand this user's core interests, expertise, and patterns without needing to process all historical conversations again.
            """
        else:
            prompt = f"""
            You are creating an insightful, structured memory system for an AI assistant. Your task is to create the initial summary for a set of user conversations.

            <<<CONVERSATIONS>>>
            {batch_content}
            <<<END_CONVERSATIONS>>>

            Generate a structured summary with these characteristics:
            
            1. DO NOT include any of the prompt markers (like "CONVERSATIONS" or "END_CONVERSATIONS") in your response
            2. DO NOT preface your response with phrases like "Here's the summary" or "Summary:"
            3. Focus on identifying KEY INSIGHTS and PATTERNS, not detailed recaps of every conversation
            4. Begin with a "CORE INTERESTS" section listing 3-7 main areas of interest shown across conversations
            5. Group information by THEMES (technical subjects, creative projects, personal interests) 
            6. For each theme, provide bullet points of relevant insights, preferences, and expertise
            7. Add a "TECHNICAL EXPERTISE" section identifying areas where the user has demonstrated knowledge
            8. Include a "RECURRING TOPICS" section for subjects that appear multiple times
            9. For important timestamps (project deadlines, significant events), use the format YYYY-MM-DD

            The summary should be immediately useful for an AI to understand this user's core interests, expertise, and patterns without needing to process all historical conversations again.
            """
        
        print(f"Generating summary for batch with {batch_content.count('CONVERSATION:')} conversations...")
        
        try:
            # Generate summary with Gemini
            response = self.gemini_client.generate_content(
                prompt=prompt,
                max_output_tokens=self.max_tokens,
                temperature=0.1  # Reduced temperature for more consistent formatting
            )
            
            # If response is None or doesn't have text attribute
            if not response or not hasattr(response, 'text'):
                print(f"WARNING: Invalid response from Gemini API: {response}")
                raise ValueError("Invalid response from Gemini API")
            
            # Process the response to ensure it's properly formatted
            summary_text = response.text.strip()
            
            # Remove any lingering prompt artifacts that might have been included
            for marker in ["<<<PREVIOUS_SUMMARY>>>", "<<<END_PREVIOUS_SUMMARY>>>", 
                          "<<<NEW_CONVERSATIONS>>>", "<<<END_NEW_CONVERSATIONS>>>",
                          "<<<CONVERSATIONS>>>", "<<<END_CONVERSATIONS>>>",
                          "Here's the updated summary:", "Summary:", "Updated Summary:"]:
                summary_text = summary_text.replace(marker, "")
            
            # Remove extra line breaks and leading/trailing whitespace
            while "\n\n\n" in summary_text:
                summary_text = summary_text.replace("\n\n\n", "\n\n")
            
            summary_text = summary_text.strip()
                
            return summary_text
            
        except Exception as e:
            print(f"ERROR generating summary: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a fallback summary to avoid crashing the process
            return f"""
            FALLBACK SUMMARY (due to API error: {str(e)})
            
            This is a computer-generated placeholder summary because there was an error generating 
            the actual summary. The system processed {batch_content.count('CONVERSATION:')} conversations.
            
            Please try running the summarization process again.
            """
    
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