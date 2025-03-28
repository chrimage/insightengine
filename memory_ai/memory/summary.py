# Rolling and conversation-specific summary implementation

import uuid
import os
import json
from datetime import datetime
import tiktoken
import numpy as np
from memory_ai.core.database import MemoryDatabase
from memory_ai.utils.gemini import GeminiClient
from memory_ai.reflection.evaluator import StructuredEvaluator
from concurrent.futures import ThreadPoolExecutor

class RollingSummaryProcessor:
    """Processes conversations to generate rolling and conversation-specific summaries."""
    
    def __init__(self, db, gemini_client, batch_size=20, max_tokens=8000):
        """Initialize the rolling summary processor."""
        self.db = db
        self.gemini_client = gemini_client
        self.batch_size = batch_size  # Increased from 10 to 20 for better efficiency
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.metadata = {}
        
        # Configuration for embedding generation
        self.generate_embeddings = os.environ.get("GENERATE_SUMMARY_EMBEDDINGS", "true").lower() == "true"
        self.evaluator = StructuredEvaluator(gemini_client) if os.environ.get("EVALUATE_SUMMARIES", "false").lower() == "true" else None
        
        # Maximum threads for parallel processing
        self.max_workers = int(os.environ.get("SUMMARY_MAX_WORKERS", "4"))
    
    def process_conversations(self):
        """Process all conversations to generate rolling summaries."""
        # Get conversations ordered by timestamp
        conversations = self._get_chronological_conversations()
        
        # Get the latest summary (if any)
        latest_summary = self.db.get_active_summary()
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
            
            # Generate embedding for the summary if configured
            embedding = None
            if self.generate_embeddings:
                try:
                    embedding = self.gemini_client.generate_embeddings(new_summary)
                except Exception as e:
                    print(f"Error generating embedding for rolling summary: {e}")
            
            # Create summary data with metadata
            summary_data = {
                'id': summary_id,
                'timestamp': datetime.now().timestamp(),
                'summary_text': new_summary,
                'conversation_range': list(processed_convs),
                'version': version,
                'embedding': embedding,
                'is_active': True,  # Make this the active summary
                'metadata': {
                    'themes': themes,
                    'topics': topics,
                    'quality_score': self._calculate_summary_quality(new_summary),
                    'batch_size': len(batch),
                    'total_processed': len(processed_convs)
                }
            }
            
            # Evaluate the summary quality if evaluator is available
            if self.evaluator:
                try:
                    # Create a sampling of the input content for evaluation
                    sample_content = self._get_sample_content(batch, 10)  # Get up to 10 conversations
                    
                    # Evaluate the summary
                    evaluation = self.evaluator.evaluate_summary(
                        summary_text=new_summary,
                        original_content=sample_content
                    )
                    
                    # Add evaluation to metadata
                    if isinstance(evaluation, dict):
                        summary_data['metadata']['evaluation'] = evaluation
                        # Use the evaluator's quality score if available
                        if 'quality_score' in evaluation:
                            summary_data['metadata']['quality_score'] = evaluation['quality_score']
                except Exception as e:
                    print(f"Error evaluating summary: {e}")
            
            # Store the summary
            self.db.store_rolling_summary(summary_data)
            
            # Update current summary for next iteration
            current_summary_text = new_summary
            latest_summary = summary_data
            
            print(f"Created summary version {version} covering {len(processed_convs)} conversations")
            
            # Generate conversation-specific summaries sequentially for this batch
            for conv in batch:
                # Only process if it doesn't already have a summary
                try:
                    conv_has_summary = False
                    with self.db.sqlite_conn as conn:
                        c = conn.cursor()
                        c.execute("SELECT 1 FROM conversations WHERE id = ? AND summary IS NOT NULL AND summary != ''", 
                                (conv['id'],))
                        if c.fetchone():
                            conv_has_summary = True
                    
                    if not conv_has_summary:
                        success = self.generate_conversation_summary(conv)
                        if success:
                            print(f"Generated summary for conversation {conv['id']} - {conv['title']}")
                except Exception as e:
                    print(f"Error generating conversation-specific summary: {e}")
                    import traceback
                    traceback.print_exc()
    
    def _get_sample_content(self, conversations, max_count=10):
        """Get a representative sample of content from conversations for evaluation.
        
        Args:
            conversations: List of conversation dictionaries
            max_count: Maximum number of conversations to include
            
        Returns:
            str: Sample content
        """
        # Select a subset of conversations if needed
        sample = conversations[:max_count] if len(conversations) > max_count else conversations
        
        # Format the content
        content = ""
        for conv in sample:
            content += f"## CONVERSATION: {conv['title']}\n"
            content += f"Date: {datetime.fromtimestamp(conv['timestamp']).strftime('%Y-%m-%d')}\n\n"
            
            # Get a sample of messages from this conversation
            with self.db.sqlite_conn as conn:
                c = conn.cursor()
                c.execute('''
                SELECT role, content FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT 10
                ''', (conv['id'],))
                
                messages = c.fetchall()
            
            # Process messages outside of the database cursor context
            for msg in messages:
                msg_content = msg[1]
                # Truncate long messages
                if len(msg_content) > 200:
                    msg_content = msg_content[:200] + "..."
                content += f"{msg[0].upper()}: {msg_content}\n\n"
            
            content += "\n---\n"
            
        return content
    
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
        c = self.db.sqlite_conn.cursor()
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
            c = self.db.sqlite_conn.cursor()
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
    
    def generate_embeddings_for_summaries(self):
        """Generate embeddings for existing summaries that don't have them yet.
        
        Returns:
            tuple: (processed_count_global, processed_count_conversation)
        """
        if not self.generate_embeddings:
            print("Embedding generation is disabled")
            return (0, 0)
            
        c = self.db.sqlite_conn.cursor()
        processed_global = 0
        processed_conversation = 0
        
        # Process global rolling summaries first
        c.execute('''
        SELECT rs.id, rs.summary_text 
        FROM rolling_summaries rs
        LEFT JOIN embeddings e ON e.source_id = rs.id
        WHERE e.id IS NULL
        ''')
        
        global_summaries = c.fetchall()
        if global_summaries:
            print(f"Found {len(global_summaries)} global summaries without embeddings")
            
            for summary in global_summaries:
                summary_id = summary[0]
                summary_text = summary[1]
                
                try:
                    # Generate and store embedding
                    embedding = self.gemini_client.generate_embeddings(summary_text)
                    
                    # Update the summary with the embedding
                    c.execute('''
                    UPDATE rolling_summaries
                    SET embedding = ?
                    WHERE id = ?
                    ''', (np.array(embedding, dtype=np.float32).tobytes(), summary_id))
                    
                    self.db.sqlite_conn.commit()
                    processed_global += 1
                    print(f"Generated embedding for global summary {summary_id}")
                    
                except Exception as e:
                    print(f"Error generating embedding for global summary {summary_id}: {e}")
        
        # Process conversation-specific summaries
        c.execute('''
        SELECT c.id, c.summary FROM conversations c
        LEFT JOIN embeddings e ON e.id = 'summary_' || c.id
        WHERE c.summary IS NOT NULL AND e.id IS NULL
        ''')
        
        conversation_summaries = c.fetchall()
        if conversation_summaries:
            print(f"Found {len(conversation_summaries)} conversation summaries without embeddings")
            
            for summary in conversation_summaries:
                conversation_id = summary[0]
                summary_text = summary[1]
                
                try:
                    # Generate and store embedding
                    embedding = self.gemini_client.generate_embeddings(summary_text)
                    
                    # Store in the embeddings table
                    embedding_id = f"summary_{conversation_id}"
                    self.db.store_embedding(embedding_id, embedding, model="summary-embedding")
                    
                    processed_conversation += 1
                    print(f"Generated embedding for conversation summary {conversation_id}")
                    
                except Exception as e:
                    print(f"Error generating embedding for conversation summary {conversation_id}: {e}")
        
        print(f"Generated embeddings for {processed_global} global summaries and {processed_conversation} conversation summaries")
        return (processed_global, processed_conversation)
    
    def _generate_summary(self, batch_content, previous_summary=None):
        """Generate a new summary incorporating the previous summary and new content."""
        if previous_summary:
            prompt = f"""
            You are creating an insightful, structured memory system for an AI assistant. Your task is to maintain a comprehensive yet **highly focused** summary of all user conversations.

            Below is the previous summary followed by new conversations to integrate:

            <<<PREVIOUS_SUMMARY>>>
            {previous_summary}
            <<<END_PREVIOUS_SUMMARY>>>

            <<<NEW_CONVERSATIONS>>>
            {batch_content}
            <<<END_NEW_CONVERSATIONS>>>

            Generate an updated summary with these characteristics:
            
            1. DO NOT include any of the prompt markers (like "PREVIOUS_SUMMARY" or "END_NEW_CONVERSATIONS") in your response.
            2. DO NOT preface your response with phrases like "Here's the updated summary" or "Summary:".
            3. Focus **strictly** on identifying **significant, recurring KEY INSIGHTS and PATTERNS**. Avoid detailed recaps or isolated facts from individual conversations.
            4. Begin with a "CORE INTERESTS" section listing 3-7 main areas of interest demonstrated **repeatedly** across conversations.
            5. Group information by **major THEMES** (technical subjects, creative projects, personal interests) rather than chronology.
            6. For each theme, provide **concise** bullet points highlighting **only the most relevant and enduring** insights, preferences, and expertise. **Exclude trivial details:** one-off mentions, minor examples, specific commands/tools used infrequently unless they illustrate a core skill or recurring pattern.
            7. Add a "TECHNICAL EXPERTISE" section identifying areas where the user has **consistently** demonstrated knowledge or significant experience.
            8. Include a "RECURRING TOPICS" section for subjects that appear **multiple times with substance**.
            9. When information is updated or contradicted, retain **only the latest, most accurate version**.
            10. For important timestamps (project deadlines, significant events), use the format YYYY-MM-DD.
            11. **DO NOT create exhaustive lists.** For example, instead of listing every Linux command mentioned, summarize the user's proficiency in Linux command-line usage. Instead of listing every song mentioned for image generation, summarize the interest in generating images for song lyrics, perhaps noting a few key examples if they represent a broader style or theme.

            The summary should be immediately useful for an AI to understand this user's **core, long-term** interests, expertise, and patterns without being cluttered by minor details. Prioritize information that reflects established preferences, skills, or goals.
            """
        else:
            prompt = f"""
            You are creating an insightful, structured memory system for an AI assistant. Your task is to create the initial **highly focused** summary for a set of user conversations.

            <<<CONVERSATIONS>>>
            {batch_content}
            <<<END_CONVERSATIONS>>>

            Generate a structured summary with these characteristics:
            
            1. DO NOT include any of the prompt markers (like "CONVERSATIONS" or "END_CONVERSATIONS") in your response.
            2. DO NOT preface your response with phrases like "Here's the summary" or "Summary:".
            3. Focus **strictly** on identifying **significant, recurring KEY INSIGHTS and PATTERNS**. Avoid detailed recaps or isolated facts from individual conversations.
            4. Begin with a "CORE INTERESTS" section listing 3-7 main areas of interest demonstrated **repeatedly** across conversations.
            5. Group information by **major THEMES** (technical subjects, creative projects, personal interests).
            6. For each theme, provide **concise** bullet points highlighting **only the most relevant and enduring** insights, preferences, and expertise. **Exclude trivial details:** one-off mentions, minor examples, specific commands/tools used infrequently unless they illustrate a core skill or recurring pattern.
            7. Add a "TECHNICAL EXPERTISE" section identifying areas where the user has **consistently** demonstrated knowledge or significant experience.
            8. Include a "RECURRING TOPICS" section for subjects that appear **multiple times with substance**.
            9. For important timestamps (project deadlines, significant events), use the format YYYY-MM-DD.
            10. **DO NOT create exhaustive lists.** For example, instead of listing every Linux command mentioned, summarize the user's proficiency in Linux command-line usage. Instead of listing every song mentioned for image generation, summarize the interest in generating images for song lyrics, perhaps noting a few key examples if they represent a broader style or theme.

            The summary should be immediately useful for an AI to understand this user's **core, long-term** interests, expertise, and patterns without being cluttered by minor details. Prioritize information that reflects established preferences, skills, or goals.
            """
        
        print(f"Generating summary for batch with {batch_content.count('CONVERSATION:')} conversations...")
        
        try:
            # Generate summary with Gemini
            response = self.gemini_client.generate_text(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=0.1  # Reduced temperature for more consistent formatting
            )
            
            # If response is None
            if not response:
                print(f"WARNING: Invalid response from Gemini API: {response}")
                raise ValueError("Invalid response from Gemini API")
            
            # Process the response to ensure it's properly formatted
            summary_text = response.strip()
            
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
    
    def process_conversation_specific_summaries(self, limit=None):
        """Generate summaries for individual conversations that don't have them yet.
        
        Args:
            limit: Maximum number of conversations to process (default: all)
            
        Returns:
            int: Number of summaries generated
        """
        with self.db.sqlite_conn as conn:
            c = conn.cursor()
            
            # Find conversations without summaries
            c.execute('''
            SELECT id, title, timestamp, model, message_count
            FROM conversations 
            WHERE summary IS NULL OR summary = ''
            ORDER BY timestamp DESC
            ''')
            
            if limit:
                conversations = c.fetchmany(limit)
            else:
                conversations = c.fetchall()
        
        # Convert to list of dictionaries (outside of the cursor context)
        conv_data = [dict(conv) for conv in conversations]
            
        if not conv_data:
            print("No conversations found without summaries")
            return 0
        
        print(f"Found {len(conv_data)} conversations without summaries")
        
        # Process each conversation sequentially instead of using threads
        # This avoids the SQLite thread safety issues
        processed_count = 0
        
        for conv in conv_data:
            try:
                # Process one at a time
                success = self.generate_conversation_summary(conv)
                if success:
                    processed_count += 1
                    print(f"Generated summary for conversation {conv['id']} - {conv['title']}")
                else:
                    print(f"Failed to generate summary for conversation {conv['id']}")
            except Exception as e:
                print(f"Error processing conversation {conv['id']}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Successfully generated {processed_count} conversation summaries")
        return processed_count
    
    def generate_conversation_summary(self, conversation):
        """Generate a summary for a specific conversation.
        
        Args:
            conversation: Dictionary with conversation data
            
        Returns:
            bool: True if successful, False otherwise
        """
        conversation_id = conversation['id']
        
        # Get messages for this conversation using a with statement to ensure proper cleanup
        with self.db.sqlite_conn as conn:
            c = conn.cursor()
            c.execute('''
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            ''', (conversation_id,))
            
            messages = c.fetchall()
        
        if not messages:
            print(f"No messages found for conversation {conversation_id}")
            return False
        
        # Format messages for the prompt
        formatted_content = f"## CONVERSATION: {conversation['title']}\n"
        formatted_content += f"Date: {datetime.fromtimestamp(conversation['timestamp']).strftime('%Y-%m-%d')}\n\n"
        
        for msg in messages:
            formatted_content += f"{msg[0].upper()}: {msg[1]}\n\n"
        
        # Generate the summary
        prompt = f"""
        You are creating a comprehensive summary of a single conversation. Your task is to extract the most important information, themes, and insights.

        <<<CONVERSATION>>>
        {formatted_content}
        <<<END_CONVERSATION>>>

        Create a JSON-formatted summary with the following structure:
        ```json
        {{
            "summary": "Concise 2-3 paragraph summary of the conversation",
            "key_points": ["List of 3-5 key points as bullet points"],
            "themes": ["List of main themes or topics discussed"],
            "entities": ["List of important entities mentioned (people, technologies, concepts)"],
            "questions": ["Any unresolved questions raised"],
            "action_items": ["Any clear action items or next steps"],
            "sentiment": "Overall sentiment of the conversation (positive, negative, neutral)",
            "technical_level": "Estimated technical expertise level in the conversation (basic, intermediate, advanced)"
        }}
        ```

        Focus on capturing:
        1. The main purpose and outcome of the conversation
        2. Important factual information exchanged
        3. User preferences, interests, and expertise revealed
        4. Any decisions made or conclusions reached
        5. Relevant technical details when present

        Only include the JSON object in your response, with no preamble or explanation.
        """
        
        try:
            # Generate summary with Gemini
            response = self.gemini_client.generate_text(
                prompt=prompt,
                max_tokens=4096,
                temperature=0.1  # Low temperature for consistency
            )
            
            # Extract and validate JSON
            try:
                # Clean up the response to extract only the JSON part
                summary_text = response.strip()
                
                # Extract JSON if it's enclosed in backticks
                if "```json" in summary_text and "```" in summary_text:
                    json_part = summary_text.split("```json")[1].split("```")[0].strip()
                elif "```" in summary_text:
                    json_part = summary_text.split("```")[1].split("```")[0].strip()
                else:
                    json_part = summary_text
                
                # Parse JSON
                summary_data = json.loads(json_part)
                
                # For backward compatibility, extract the summary text
                summary_text = summary_data["summary"]
                
                # Generate embedding for the summary if configured
                embedding = None
                if self.generate_embeddings:
                    try:
                        # Get embedding for the complete JSON to enable semantic search
                        full_text = json.dumps(summary_data)
                        embedding = self.gemini_client.generate_embeddings(full_text)
                    except Exception as e:
                        print(f"Error generating embedding for conversation summary: {e}")
                
                # Evaluate the summary if configured
                quality_score = 0.5  # Default score
                if self.evaluator:
                    try:
                        evaluation = self.evaluator.evaluate_summary(
                            summary_text=summary_text,
                            original_content=formatted_content
                        )
                        if isinstance(evaluation, dict) and 'quality_score' in evaluation:
                            quality_score = evaluation['quality_score']
                        summary_data['evaluation'] = evaluation
                    except Exception as e:
                        print(f"Error evaluating summary: {e}")
                
                # Store the summary in the database
                return self.db.store_conversation_summary(
                    conversation_id=conversation_id,
                    summary_text=summary_text,
                    metadata=summary_data,
                    embedding=embedding
                )
                
            except json.JSONDecodeError:
                # Fallback for non-JSON output
                print(f"Failed to parse JSON for conversation {conversation_id}, using plain text")
                summary_text = response.strip()
                
                return self.db.store_conversation_summary(
                    conversation_id=conversation_id,
                    summary_text=summary_text
                )
                
        except Exception as e:
            print(f"Error generating summary for conversation {conversation_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def implement_forgetting(self, days_threshold=180):
        """Apply a forgetting mechanism to gradually remove outdated information."""
        latest_summary = self.db.get_active_summary()
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
        
        response = self.gemini_client.generate_text(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=0.2
        )
        
        # Store the updated summary
        summary_id = str(uuid.uuid4())
        
        # Generate embedding for the updated summary if configured
        embedding = None
        if self.generate_embeddings:
            try:
                embedding = self.gemini_client.generate_embeddings(response)
            except Exception as e:
                print(f"Error generating embedding for forgotten summary: {e}")
        
        summary_data = {
            'id': summary_id,
            'timestamp': datetime.now().timestamp(),
            'summary_text': response,
            'conversation_range': latest_summary['conversation_range'],
            'version': latest_summary['version'] + 1,
            'embedding': embedding,
            'is_active': True,  # This will be the new active summary
            'metadata': {
                'forgotten_from': latest_summary.get('id'),
                'days_threshold': days_threshold,
                'themes': self._extract_themes_and_topics(response)[0],
                'topics': self._extract_themes_and_topics(response)[1],
                'quality_score': self._calculate_summary_quality(response)
            }
        }
        
        self.db.store_rolling_summary(summary_data)
        
        return summary_data
