# Response quality evaluation

import os
import json
import time
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class StructuredEvaluator:
    """
    Evaluator for structured content like summaries, using a consistent evaluation framework.
    This lighter-weight evaluator works directly with the LLM without requiring database access.
    """
    
    def __init__(self, llm_client):
        """
        Initialize the structured evaluator.
        
        Args:
            llm_client: Client for LLM API access (Gemini, etc.)
        """
        self.llm_client = llm_client
        self.debug_mode = os.environ.get('DEBUG_EVALUATION', '').lower() in ['true', '1', 'yes']
    
    def evaluate_summary(self, summary_text, original_content=None):
        """
        Evaluate a summary against the original content.
        
        Args:
            summary_text: The summary to evaluate
            original_content: Optional original content that was summarized
            
        Returns:
            dict: Evaluation results including scores and feedback
        """
        # Construct evaluation prompt based on what's available
        if original_content:
            prompt = f"""
            You are evaluating the quality of a summary. Review both the original content and the summary, then provide a structured evaluation.
            
            ORIGINAL CONTENT:
            ```
            {original_content[:10000]}  # Truncate if too long
            ```
            
            SUMMARY:
            ```
            {summary_text}
            ```
            
            Evaluate the summary on these dimensions (score each from 0.0 to 1.0):
            1. COMPLETENESS: Does the summary capture the key information from the original?
            2. ACCURACY: Is the information in the summary factually correct?
            3. CONCISENESS: Does the summary avoid unnecessary details while preserving meaning?
            4. COHERENCE: Is the summary well-structured and logically organized?
            5. USEFULNESS: How effective would this summary be for understanding the original content?
            
            Return your evaluation in JSON format:
            ```json
            {{
                "scores": {{
                    "completeness": float,
                    "accuracy": float,
                    "conciseness": float,
                    "coherence": float,
                    "usefulness": float,
                    "overall": float
                }},
                "strengths": [
                    "Specific strength 1",
                    "Specific strength 2"
                ],
                "weaknesses": [
                    "Specific weakness 1",
                    "Specific weakness 2"
                ],
                "quality_score": float,  // Single combined quality score from 0.0 to 1.0
                "improvement_suggestions": [
                    "Specific suggestion 1",
                    "Specific suggestion 2"
                ]
            }}
            ```
            
            Only return the JSON with no additional text.
            """
        else:
            # Evaluate the summary in isolation if no original content is provided
            prompt = f"""
            You are evaluating the quality of a summary. Review the summary and provide a structured evaluation.
            
            SUMMARY:
            ```
            {summary_text}
            ```
            
            Evaluate the summary on these dimensions (score each from 0.0 to 1.0):
            1. COHERENCE: Is the summary well-structured and logically organized?
            2. INFORMATIVENESS: Does the summary contain substantial, useful information?
            3. CLARITY: Is the summary clear and easy to understand?
            4. STRUCTURE: Is the summary well-formatted with appropriate organization?
            5. USEFULNESS: How useful would this summary be as a memory aid?
            
            Return your evaluation in JSON format:
            ```json
            {{
                "scores": {{
                    "coherence": float,
                    "informativeness": float,
                    "clarity": float,
                    "structure": float,
                    "usefulness": float,
                    "overall": float
                }},
                "strengths": [
                    "Specific strength 1",
                    "Specific strength 2"
                ],
                "weaknesses": [
                    "Specific weakness 1",
                    "Specific weakness 2"
                ],
                "quality_score": float,  // Single combined quality score from 0.0 to 1.0
                "improvement_suggestions": [
                    "Specific suggestion 1",
                    "Specific suggestion 2"
                ]
            }}
            ```
            
            Only return the JSON with no additional text.
            """
        
        try:
            # Generate evaluation using LLM
            response = self.llm_client.generate_content(
                prompt=prompt,
                max_output_tokens=2000,
                temperature=0.1  # Low temperature for consistency
            )
            
            # Extract and parse JSON
            response_text = response.text.strip()
            
            # Clean up response to extract JSON
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                json_text = response_text
            
            try:
                evaluation_data = json.loads(json_text)
                
                # Ensure quality_score exists
                if "quality_score" not in evaluation_data:
                    # Calculate from overall if available
                    if "scores" in evaluation_data and "overall" in evaluation_data["scores"]:
                        evaluation_data["quality_score"] = evaluation_data["scores"]["overall"]
                    else:
                        # Average the available scores
                        scores = evaluation_data.get("scores", {})
                        if scores:
                            evaluation_data["quality_score"] = sum(scores.values()) / len(scores)
                        else:
                            evaluation_data["quality_score"] = 0.5  # Default
                
                # Add timestamp
                evaluation_data["timestamp"] = datetime.now().timestamp()
                
                return evaluation_data
                
            except json.JSONDecodeError as e:
                if self.debug_mode:
                    print(f"Failed to parse summary evaluation as JSON: {e}")
                    print(f"Response text: {response_text[:200]}...")
                
                # Return basic evaluation
                return {
                    "quality_score": 0.5,  # Default score
                    "error": "Could not parse evaluation",
                    "strengths": [],
                    "weaknesses": ["Could not automatically evaluate summary"],
                    "raw_response": response_text
                }
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error evaluating summary: {e}")
                
            # Return error info
            return {
                "quality_score": 0.5,  # Default score
                "error": str(e),
                "strengths": [],
                "weaknesses": ["Error during summary evaluation"],
            }

class ResponseEvaluator:
    """
    Evaluates AI responses to extract insights and improve future responses.
    
    This system implements a sophisticated self-reflection capability that:
    1. Evaluates responses on multiple quality dimensions
    2. Identifies specific strengths and weaknesses
    3. Extracts actionable insights for future improvement
    4. Tracks success metrics for continuous learning
    5. Provides structured data for analytics and reporting
    
    The evaluation process can run asynchronously to avoid slowing down
    the chat interaction, with insights being applied in future conversations.
    """
    
    # Define evaluation dimensions and their descriptions
    EVALUATION_DIMENSIONS = {
        "relevance": "How well the response addresses the specific query",
        "accuracy": "Factual correctness and absence of errors",
        "helpfulness": "Practical utility and value to the user",
        "coherence": "Logical flow, clarity, and readability",
        "completeness": "Comprehensive coverage of the topic",
        "conciseness": "Appropriate length and focus without unnecessary content",
        "tone": "Appropriateness of tone, formality, and empathy",
        "creativity": "Novel approaches and insights where appropriate"
    }
    
    # Define insight categories
    INSIGHT_CATEGORIES = [
        "response_structure",
        "information_presentation",
        "user_engagement",
        "technical_accuracy",
        "tone_management",
        "question_understanding",
        "context_utilization",
        "explanation_technique",
        "error_recovery",
        "general_improvement"
    ]
    
    def __init__(self, db, gemini_client, insight_repository):
        """
        Initialize the response evaluator.
        
        Args:
            db: Database connection
            gemini_client: Client for Gemini API access
            insight_repository: Repository for storing and retrieving insights
        """
        self.db = db
        self.gemini_client = gemini_client
        self.insight_repository = insight_repository
        
        # Configuration
        self.async_evaluation = os.environ.get('ASYNC_EVALUATION', '').lower() in ['true', '1', 'yes']
        self.evaluation_frequency = float(os.environ.get('EVALUATION_FREQUENCY', '1.0'))
        self.default_confidence = float(os.environ.get('DEFAULT_INSIGHT_CONFIDENCE', '0.7'))
        self.perform_meta_evaluation = os.environ.get('PERFORM_META_EVALUATION', '').lower() in ['true', '1', 'yes']
        
        # Create thread pool for async evaluation
        if self.async_evaluation:
            self.executor = ThreadPoolExecutor(max_workers=2)
            
        # Track evaluation history
        self.evaluation_history = []
        
        # Debug mode
        self.debug_mode = os.environ.get('DEBUG_EVALUATION', '').lower() in ['true', '1', 'yes']
    
    def evaluate_response(self, query, response, conversation_history=None, user_feedback=None):
        """
        Evaluate a response and extract insights.
        
        This method performs a comprehensive evaluation of an AI response, extracting
        structured data and actionable insights. It can run either synchronously or
        asynchronously based on configuration.
        
        Args:
            query: The user's query
            response: The AI's response
            conversation_history: Optional history of the conversation
            user_feedback: Optional explicit feedback from the user
            
        Returns:
            dict: Evaluation results including scores, insights, and metadata
        """
        # Apply evaluation frequency - randomly skip some evaluations if configured
        if self.evaluation_frequency < 1.0 and not self._should_evaluate():
            if self.debug_mode:
                print(f"Skipping evaluation based on frequency setting ({self.evaluation_frequency})")
            return {
                "evaluation": "Evaluation skipped",
                "insight_extracted": False,
                "skipped": True
            }
            
        # Run asynchronously if configured
        if self.async_evaluation:
            if self.debug_mode:
                print("Starting asynchronous evaluation")
            
            # Submit the evaluation task to the thread pool
            future = self.executor.submit(
                self._perform_evaluation,
                query, 
                response, 
                conversation_history,
                user_feedback
            )
            
            # Return immediately without waiting for results
            return {
                "evaluation": "Evaluation started asynchronously",
                "insight_extracted": False,
                "async": True,
                "task_id": str(id(future))  # Simple task ID
            }
        else:
            # Run synchronously and return results
            return self._perform_evaluation(query, response, conversation_history, user_feedback)
        
    def _should_evaluate(self):
        """Determine if this interaction should be evaluated based on frequency setting."""
        import random
        return random.random() < self.evaluation_frequency
            
    def _perform_evaluation(self, query, response, conversation_history=None, user_feedback=None):
        """
        Perform the actual evaluation (internal method).
        
        This separates the evaluation logic from the async/sync decision, allowing the same
        implementation to be used in both modes.
        """
        evaluation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Format the conversation for structured evaluation
            conv_text = self._format_conversation(query, response, conversation_history)
            
            # First pass: Structured evaluation with JSON output
            structured_evaluation = self._evaluate_structured(query, response, conv_text, user_feedback)
            
            # Extract and validate the structured evaluation
            if not structured_evaluation:
                # Fallback to unstructured evaluation
                unstructured_evaluation = self._evaluate_unstructured(query, response, conv_text)
                evaluation_data = {
                    "evaluation_id": evaluation_id,
                    "structured": False,
                    "scores": None,
                    "strengths": None,
                    "weaknesses": None,
                    "insights": None,
                    "raw_evaluation": unstructured_evaluation,
                    "timestamp": datetime.now().timestamp(),
                    "query": query
                }
                
                # Try to extract insights from unstructured evaluation
                insights = self._extract_insights_from_text(unstructured_evaluation)
            else:
                # Process structured evaluation
                evaluation_data = structured_evaluation
                evaluation_data["evaluation_id"] = evaluation_id
                evaluation_data["structured"] = True
                evaluation_data["timestamp"] = datetime.now().timestamp()
                evaluation_data["query"] = query
                
                # Extract insights directly
                insights = evaluation_data.get("insights", [])
                
            # Store the evaluation in the database
            self._store_evaluation(evaluation_data)
            
            # Process extracted insights
            if insights:
                for insight in insights:
                    if not isinstance(insight, dict):
                        # Skip invalid insights
                        continue
                        
                    text = insight.get("text")
                    category = insight.get("category", "general_improvement")
                    confidence = insight.get("confidence", self.default_confidence)
                    
                    if not text:
                        continue
                        
                    evidence = [{
                        "query": query,
                        "response": response,
                        "evaluation_id": evaluation_id,
                        "scores": evaluation_data.get("scores")
                    }]
                    
                    # Add user feedback if available
                    if user_feedback:
                        evidence[0]["user_feedback"] = user_feedback
                    
                    # Add insight to repository
                    insight_id = self.insight_repository.add_insight(
                        text=text,
                        category=category,
                        confidence=confidence,
                        evidence=evidence
                    )
                    
                    # Log if in debug mode
                    if self.debug_mode:
                        print(f"Added insight {insight_id}: {category} - {text}")
            
            # Track in history (truncate if needed)
            self.evaluation_history.append(evaluation_data)
            if len(self.evaluation_history) > 100:
                self.evaluation_history = self.evaluation_history[-100:]
                
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Optional meta-evaluation (evaluate the evaluator)
            if self.perform_meta_evaluation and structured_evaluation:
                self._run_meta_evaluation(evaluation_data)
            
            # Return results
            return {
                "evaluation_id": evaluation_id,
                "structured": bool(structured_evaluation),
                "scores": evaluation_data.get("scores"),
                "strengths": evaluation_data.get("strengths"),
                "weaknesses": evaluation_data.get("weaknesses"),
                "insights": insights,
                "processing_time": processing_time,
                "insight_extracted": bool(insights)
            }
            
        except Exception as e:
            error_message = f"Error in evaluation: {str(e)}"
            if self.debug_mode:
                import traceback
                error_message = f"Error in evaluation: {str(e)}\n{traceback.format_exc()}"
                print(error_message)
                
            # Return error info
            return {
                "evaluation_id": evaluation_id,
                "evaluation": "Evaluation failed",
                "error": str(e),
                "insight_extracted": False
            }
            
    def _evaluate_structured(self, query, response, conv_text, user_feedback=None):
        """
        Perform structured evaluation with JSON output.
        
        This method prompts the LLM to produce structured evaluation data in JSON format,
        which makes it easier to parse and use programmatically.
        
        Args:
            query: The user query
            response: The AI response
            conv_text: Formatted conversation text
            user_feedback: Optional feedback from the user
            
        Returns:
            dict: Structured evaluation data or None if evaluation failed
        """
        # Create list of dimensions for prompt
        dimensions_text = "\n".join([
            f"{i+1}. {dim.upper()}: {desc}" 
            for i, (dim, desc) in enumerate(self.EVALUATION_DIMENSIONS.items())
        ])
        
        # Include user feedback in prompt if available
        feedback_text = ""
        if user_feedback:
            feedback_text = f"""
            USER FEEDBACK ON THIS RESPONSE:
            {user_feedback}
            
            Take this feedback into account in your evaluation.
            """
            
        # Construct the evaluation prompt
        evaluation_prompt = f"""
        You are an objective evaluator analyzing AI assistant responses. Your job is to provide detailed feedback to help the AI improve.
        
        CONVERSATION TO EVALUATE:
        {conv_text}
        {feedback_text}
        
        Evaluate the AI's response on these dimensions (score each from 0.0 to 1.0):
        {dimensions_text}
        
        Analyze what worked well and what could be improved, then formulate 1-3 specific, actionable insights that would help improve future responses.
        
        GUIDELINES FOR INSIGHTS:
        - Focus on concrete techniques, not vague suggestions
        - Be specific and actionable
        - Consider context utilization, explanation techniques, and response structure
        - Insights should be applicable to similar future queries
        
        FORMAT YOUR RESPONSE AS JSON:
        ```json
        {{
            "scores": {{
                "relevance": float,
                "accuracy": float,
                "helpfulness": float,
                "coherence": float,
                "completeness": float,
                "conciseness": float,
                "tone": float,
                "creativity": float,
                "overall": float
            }},
            "strengths": [
                "Specific strength 1",
                "Specific strength 2"
            ],
            "weaknesses": [
                "Specific weakness 1",
                "Specific weakness 2"
            ],
            "insights": [
                {{
                    "category": string,  // One of: {", ".join(self.INSIGHT_CATEGORIES)}
                    "text": "Detailed, actionable insight",
                    "confidence": float  // How confident you are in this insight (0.0-1.0)
                }}
            ],
            "explanation": "Brief overall assessment of the response's quality"
        }}
        ```
        
        BE OBJECTIVE AND CONSTRUCTIVE. The purpose is to help the AI improve, not to criticize.
        RETURN ONLY THE JSON WITH NO OTHER TEXT.
        """
        
        try:
            # Make LLM call with shorter timeout for evaluation
            eval_response = self.gemini_client.generate_content(
                prompt=evaluation_prompt,
                max_output_tokens=2000,
                temperature=0.1  # Low temperature for consistency
            )
            
            # Extract JSON from response
            response_text = eval_response.text.strip()
            
            # Clean up response text to extract JSON
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                json_text = response_text
                
            # Parse JSON
            try:
                evaluation_data = json.loads(json_text)
                
                # Validate required fields
                required_fields = ["scores", "strengths", "weaknesses", "insights"]
                if not all(field in evaluation_data for field in required_fields):
                    if self.debug_mode:
                        missing = [f for f in required_fields if f not in evaluation_data]
                        print(f"Missing required fields in evaluation: {missing}")
                    return None
                    
                # Validate scores
                scores = evaluation_data.get("scores", {})
                if not isinstance(scores, dict) or len(scores) < 5:
                    if self.debug_mode:
                        print(f"Invalid scores in evaluation: {scores}")
                    return None
                    
                # Ensure scores are in 0-1 range
                for dimension, score in scores.items():
                    if not isinstance(score, (int, float)) or score < 0 or score > 1:
                        scores[dimension] = max(0.0, min(1.0, float(score) / 10.0 if score > 1 else float(score)))
                        
                # Add raw evaluation text for reference
                evaluation_data["raw_evaluation"] = response_text
                
                return evaluation_data
                
            except json.JSONDecodeError as e:
                if self.debug_mode:
                    print(f"Failed to parse evaluation as JSON: {e}")
                    print(f"JSON text: {json_text[:100]}...")
                return None
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error in structured evaluation: {e}")
            return None

    def _evaluate_unstructured(self, query, response, conv_text):
        """
        Fallback to unstructured evaluation when structured evaluation fails.
        
        This provides a simpler, text-based evaluation that can still be processed
        to extract insights.
        """
        # Simpler evaluation prompt without JSON structure requirement
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
        - Three specific lessons that could be applied to future responses
        
        Format each lesson as:
        INSIGHT: [Category] - [Specific lesson learned]
        
        Categories should be one of: {", ".join(self.INSIGHT_CATEGORIES)}
        """
        
        try:
            eval_response = self.gemini_client.generate_content(
                prompt=evaluation_prompt,
                max_output_tokens=1000,
                temperature=0.2
            )
            
            return eval_response.text
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error in unstructured evaluation: {e}")
            return f"Evaluation failed: {str(e)}"
            
    def _extract_insights_from_text(self, evaluation_text):
        """
        Extract insights from unstructured evaluation text.
        
        This is a fallback method used when the structured JSON evaluation fails.
        It uses pattern matching to find insights in the text.
        """
        insights = []
        
        if not evaluation_text:
            return insights
            
        # Extract insights using INSIGHT: pattern
        if "INSIGHT:" in evaluation_text:
            insight_sections = evaluation_text.split("INSIGHT:")[1:]
            
            for section in insight_sections:
                # Find the end of the insight (newline or next INSIGHT: tag)
                end_pos = section.find("\n\n")
                if end_pos == -1:
                    end_pos = len(section)
                    
                # Extract the insight text
                insight_text = section[:end_pos].strip()
                
                # Parse category and text
                category = "general_improvement"
                text = insight_text
                
                if "-" in insight_text:
                    parts = insight_text.split("-", 1)
                    potential_category = parts[0].strip().lower()
                    
                    # Validate category
                    if potential_category in [cat.lower() for cat in self.INSIGHT_CATEGORIES]:
                        category = next(cat for cat in self.INSIGHT_CATEGORIES if cat.lower() == potential_category)
                        text = parts[1].strip()
                
                insights.append({
                    "category": category,
                    "text": text,
                    "confidence": self.default_confidence
                })
        
        return insights
        
    def _store_evaluation(self, evaluation_data):
        """
        Store the evaluation in the database for future reference.
        """
        try:
            c = self.db.conn.cursor()
            
            # Convert to JSON for storage
            evaluation_json = json.dumps(evaluation_data)
            
            # Store in evaluations table
            c.execute('''
            INSERT INTO evaluations
            (id, timestamp, query, evaluation, structured)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                evaluation_data["evaluation_id"],
                evaluation_data["timestamp"],
                evaluation_data["query"],
                evaluation_json,
                evaluation_data["structured"]
            ))
            
            self.db.conn.commit()
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error storing evaluation: {e}")
                
    def _run_meta_evaluation(self, evaluation_data):
        """
        Perform meta-evaluation to assess and improve the evaluator itself.
        
        This is an advanced feature that uses the LLM to evaluate the quality
        of its own evaluations, helping to improve the evaluation process over time.
        """
        try:
            # Get the raw evaluation and structured result
            raw_eval = evaluation_data.get("raw_evaluation", "")
            structured = evaluation_data.get("structured", False)
            
            meta_prompt = f"""
            You're conducting a meta-evaluation - assessing the quality of an AI evaluator's analysis.
            
            THE EVALUATION TO ASSESS:
            ```
            {raw_eval}
            ```
            
            Assess this evaluation on:
            1. OBJECTIVITY: Is the evaluation fair and unbiased?
            2. SPECIFICITY: Are the insights specific and actionable?
            3. COMPREHENSIVENESS: Does it cover important aspects of the response?
            4. CONSTRUCTIVITY: Is the feedback helpful for improvement?
            
            Provide 1-2 specific suggestions on how to improve the evaluation process itself.
            Format your response as JSON:
            
            ```json
            {{
                "meta_scores": {{
                    "objectivity": float,
                    "specificity": float,
                    "comprehensiveness": float,
                    "constructivity": float
                }},
                "improvement_suggestions": [
                    "Specific suggestion for improving the evaluation process"
                ]
            }}
            ```
            
            ONLY RETURN THE JSON WITH NO OTHER TEXT.
            """
            
            # Run meta-evaluation with lower priority
            self.executor.submit(self._process_meta_evaluation, meta_prompt, evaluation_data["evaluation_id"])
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error starting meta-evaluation: {e}")
    
    def _process_meta_evaluation(self, meta_prompt, evaluation_id):
        """Process meta-evaluation in background thread."""
        try:
            # Make LLM call for meta-evaluation
            meta_response = self.gemini_client.generate_content(
                prompt=meta_prompt,
                max_output_tokens=1000,
                temperature=0.2
            )
            
            # Try to parse JSON
            response_text = meta_response.text.strip()
            
            # Clean up response text to extract JSON
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                json_text = response_text
                
            try:
                meta_data = json.loads(json_text)
                
                # Store meta-evaluation results
                c = self.db.conn.cursor()
                c.execute('''
                UPDATE evaluations 
                SET meta_evaluation = ? 
                WHERE id = ?
                ''', (json.dumps(meta_data), evaluation_id))
                
                self.db.conn.commit()
                
                if self.debug_mode:
                    print(f"Meta-evaluation completed for evaluation {evaluation_id}")
                    
            except json.JSONDecodeError:
                if self.debug_mode:
                    print(f"Failed to parse meta-evaluation as JSON")
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error in meta-evaluation processing: {e}")
    
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