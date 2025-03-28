# Context assembly from multiple sources

import tiktoken
from datetime import datetime

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
        summary = self.db.get_active_summary()
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
                c = self.db.sqlite_conn.cursor()
                c.execute('''
                SELECT title, timestamp FROM conversations
                WHERE id = ?
                ''', (memory['source_id'],))
                
                conv = c.fetchone()
                if conv:
                    memory_text += f"Title: {conv['title']}\n"
                    memory_text += f"Date: {datetime.fromtimestamp(conv['timestamp']).strftime('%Y-%m-%d')}\n"
                
                # Get content
                c.execute('''
                SELECT role, content FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT 20  -- Limit number of messages
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