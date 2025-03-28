# Main chat agent implementation

from memory_ai.chat.prompts import PromptTemplates

class MemoryEnhancedAgent:
    """Chat agent with enhanced memory capabilities."""
    
    def __init__(self, db, gemini_client, context_assembler, evaluator):
        """Initialize the memory-enhanced chat agent."""
        self.db = db
        self.gemini_client = gemini_client
        self.context_assembler = context_assembler
        self.evaluator = evaluator
        self.conversation_history = []
    
    def chat(self, message):
        """Process a message and generate a response."""
        # Add message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Assemble context
        context, metadata = self.context_assembler.assemble_context(
            query=message,
            conversation_history=self.conversation_history
        )
        
        # Build prompt using templates
        prompt = PromptTemplates.format_with_context(context, metadata)
        
        # Generate response
        response_text = self.gemini_client.generate_content(
            prompt=prompt,
            max_output_tokens=8000,
            temperature=0.7
        )
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Evaluate response in the background
        # This could be run asynchronously or in a separate thread
        self.evaluator.evaluate_response(
            query=message,
            response=response_text,
            conversation_history=self.conversation_history[:-2]  # Exclude current interaction
        )
        
        # Update memory quality scores based on usage
        if "memory_ids" in metadata:
            # This would ideally be done based on feedback, but we'll assume 
            # memories were helpful for now
            for memory_id in metadata["memory_ids"]:
                self.db.sqlite_conn.execute('''
                UPDATE conversations 
                SET quality_score = quality_score + 0.01
                WHERE id = ? AND quality_score < 1.0
                ''', (memory_id,))
                self.db.sqlite_conn.commit()
        
        # Update insight performance if insights were used
        if "insight_ids" in metadata:
            for insight_id in metadata["insight_ids"]:
                # We would ideally determine this based on actual outcome
                was_successful = True  # Placeholder
                self.evaluator.insight_repository.update_insight_performance(
                    insight_id, was_successful
                )
        
        return response_text
    
    def clear_history(self):
        """Clear the current conversation history."""
        self.conversation_history = []
        return "Conversation history cleared."