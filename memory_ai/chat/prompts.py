# Templated prompts for different scenarios

class PromptTemplates:
    """Collection of prompt templates for different scenarios."""
    
    SYSTEM_BASE = """
    You are an AI assistant with an enhanced memory system that allows you to reference 
    past conversations and learned insights. Use this context to provide personalized, 
    helpful responses. Incorporate relevant information from your memory when appropriate, 
    but do not explicitly mention the memory system unless asked directly about your capabilities.
    """
    
    SYSTEM_WITH_MEMORY = SYSTEM_BASE + """
    Refer to the conversation history summary for overall context about the user.
    """
    
    SYSTEM_WITH_SPECIFIC_MEMORIES = SYSTEM_BASE + """
    Incorporate specific details from relevant memories when they directly relate to the query.
    """
    
    SYSTEM_WITH_INSIGHTS = SYSTEM_BASE + """
    Apply the response guidance to improve your answer quality.
    """
    
    SYSTEM_COMPREHENSIVE = SYSTEM_BASE + """
    Refer to the conversation history summary for overall context about the user.
    Incorporate specific details from relevant memories when they directly relate to the query.
    Apply the response guidance to improve your answer quality.
    """
    
    @classmethod
    def get_prompt(cls, context_metadata):
        """Get appropriate prompt based on available context components."""
        components = context_metadata.get("components", [])
        
        if "rolling_summary" in components and "relevant_memories" in components and "insights" in components:
            return cls.SYSTEM_COMPREHENSIVE
        elif "rolling_summary" in components:
            return cls.SYSTEM_WITH_MEMORY
        elif "relevant_memories" in components:
            return cls.SYSTEM_WITH_SPECIFIC_MEMORIES
        elif "insights" in components:
            return cls.SYSTEM_WITH_INSIGHTS
        else:
            return cls.SYSTEM_BASE
    
    @classmethod
    def format_with_context(cls, context, context_metadata):
        """Format a complete prompt with system instructions and context."""
        system_prompt = cls.get_prompt(context_metadata)
        return f"{system_prompt.strip()}\n\n{context}\n\nProvide a helpful, thorough response based on the information above."
