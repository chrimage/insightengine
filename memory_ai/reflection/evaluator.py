# Response quality evaluation

from datetime import datetime

class ResponseEvaluator:
    """Evaluates AI responses to extract insights and improve future responses."""
    
    def __init__(self, db, gemini_client, insight_repository):
        """Initialize the response evaluator."""
        self.db = db
        self.gemini_client = gemini_client
        self.insight_repository = insight_repository
    
    def evaluate_response(self, query, response, conversation_history=None):
        """Evaluate a response and extract insights."""
        # Format the conversation for evaluation
        conv_text = self._format_conversation(query, response, conversation_history)
        
        evaluation_prompt = f"""
        Analyze this AI interaction objectively:
        
        {conv_text}
        
        Evaluate the AI's response on these dimensions:
        1. Relevance to the query (1-10)
        2. Accuracy of information (1-10)
        3. Helpfulness to the user (1-10)
        4. Coherence and clarity (1-10)
        5. Overall quality (1-10)
        
        Then identify:
        - What specifically worked well in this response
        - What could be improved
        - One specific lesson that could be applied to future responses
        
        Format your evaluation as a structured assessment with scores and detailed feedback.
        Include ONE clear, actionable insight in the format:
        INSIGHT: [Category] - [Specific lesson learned]
        """
        
        try:
            eval_response = self.gemini_client.generate_content(
                prompt=evaluation_prompt,
                max_output_tokens=1000,
                temperature=0.2
            )
            
            # Extract insight from evaluation
            insight_text = None
            category = "response_quality"
            
            # Simple extraction with string operations
            if "INSIGHT:" in eval_response.text:
                insight_section = eval_response.text.split("INSIGHT:")[1].strip()
                if "-" in insight_section:
                    parts = insight_section.split("-", 1)
                    category = parts[0].strip()
                    insight_text = parts[1].strip()
                else:
                    insight_text = insight_section
            
            # Add insight if found
            if insight_text:
                evidence = [{
                    "query": query,
                    "response": response,
                    "evaluation": eval_response.text
                }]
                
                self.insight_repository.add_insight(
                    text=insight_text,
                    category=category,
                    confidence=0.7,  # Start with moderate confidence
                    evidence=evidence
                )
            
            return {
                "evaluation": eval_response.text,
                "insight_extracted": bool(insight_text)
            }
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {
                "evaluation": "Evaluation failed",
                "insight_extracted": False
            }
    
    def _format_conversation(self, query, response, history=None):
        """Format the conversation for evaluation."""
        formatted = ""
        
        # Add history if available
        if history:
            formatted += "PREVIOUS CONVERSATION:\n"
            for msg in history:
                role = "USER" if msg.get('role') == 'user' else "AI"
                formatted += f"{role}: {msg.get('content', '')}\n\n"
        
        # Add the current interaction
        formatted += "CURRENT INTERACTION:\n"
        formatted += f"USER: {query}\n\n"
        formatted += f"AI: {response}\n"
        
        return formatted