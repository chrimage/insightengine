"""
LLM client interface for Google Gemini models.

This module provides a standardized interface for interacting with
Google's Gemini models for embeddings and text generation.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union, Callable
import os
import random
import json
from collections import deque
from datetime import datetime
from functools import lru_cache

import google.generativeai as genai
import numpy as np
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    before_sleep_log
)
from dotenv import load_dotenv

from memory_ai.core.config import get_settings

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Google's Gemini models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client.
        
        Args:
            api_key: Google API key (optional, falls back to config)
        """
        settings = get_settings()
        self.api_key = api_key or settings.llm.google_api_key
        
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini client")
            
        # Configure the client
        genai.configure(api_key=self.api_key)
        
        # Store model names from config
        self.model_name = settings.llm.gemini_model
        self.embedding_model = settings.llm.embedding_model
        
        # Debug settings
        self.verbose = settings.llm.verbose_embeddings
        self.debug = settings.llm.debug_embeddings
        
        # Rate limiting configuration
        self.max_requests_per_minute = 60
        self.request_timestamps = deque(maxlen=self.max_requests_per_minute)
        self.max_retries = 5
        self.base_retry_delay = 2
    
    def _handle_rate_limiting(self):
        """Handle rate limiting logic."""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
        
        # If we're at 90% of the limit, start adding small delays
        if len(self.request_timestamps) >= (self.max_requests_per_minute * 0.9):
            # Add a small delay based on how close we are to the limit
            ratio = len(self.request_timestamps) / self.max_requests_per_minute
            small_delay = ratio * 1.5
            if self.debug:
                logger.debug(f"Approaching rate limit, adding {small_delay:.2f}s delay")
            time.sleep(small_delay)
        
        # If we're at the limit, wait with buffer
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            # Wait until we're under the limit
            wait_time = 62 - (current_time - self.request_timestamps[0])  # 60s + 2s buffer
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # Record this request
        self.request_timestamps.append(time.time())
    
    def _is_rate_limit_error(self, error):
        """Check if an error is related to rate limiting."""
        error_str = str(error).lower()
        return any(term in error_str for term in 
                 ["429", "resource exhausted", "quota", "rate limit"])
    
    @retry(
        retry=retry_if_exception_type((genai.types.BlockedPromptException, genai.types.InternalServerError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def generate_embeddings(self, text: str, chunk_size: int = 1900) -> List[float]:
        """Generate embeddings for a text using Gemini.
        
        Args:
            text: Text to generate embeddings for
            chunk_size: Size of text chunks for long inputs
            
        Returns:
            List[float]: Embedding vector
        """
        if self.verbose:
            logger.info(f"Generating embeddings for text of length {len(text)}")
            
        if self.debug:
            logger.debug(f"Text for embedding: {text[:100]}...")
        
        # If text is empty, return a zero embedding
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for embedding. Returning zero embedding.")
            return [0.0] * 768  # Return zero embedding
            
        # Check if text needs to be chunked (rough estimate: 4 chars ~ 1 token)
        if len(text) > chunk_size * 4:
            if self.verbose:
                logger.info(f"Text too long ({len(text)} chars), splitting into chunks...")
            
            # Simple chunking by character count
            chunks = [text[i:i + chunk_size * 4] for i in range(0, len(text), chunk_size * 4)]
            
            # Limit chunks to 5 to stay within reasonable limits
            if len(chunks) > 5:
                if self.verbose:
                    logger.info(f"Too many chunks ({len(chunks)}), limiting to 5 representative chunks")
                # Take first, last, and 3 evenly spaced chunks from the middle
                if len(chunks) >= 3:
                    step = len(chunks) // 4
                    indices = [0, step, 2*step, 3*step, len(chunks)-1]
                    chunks = [chunks[i] for i in indices]
                else:
                    chunks = chunks[:5]
            
            # Get embeddings for each chunk
            all_embeddings = []
            for i, chunk in enumerate(chunks):
                if self.verbose:
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
                
                # Handle rate limiting
                self._handle_rate_limiting()
                
                try:
                    # Generate embedding for this chunk
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=chunk,
                        task_type="retrieval_document"
                    )
                    
                    embedding = result["embedding"]
                    if embedding and len(embedding) > 0:
                        all_embeddings.append(embedding)
                    else:
                        logger.warning(f"Empty embedding returned for chunk {i+1}")
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    # Continue with other chunks
                
                # Add short delay between chunk processing to avoid rate limits
                if i < len(chunks) - 1:
                    time.sleep(0.5)
            
            # If we have any embeddings, average them
            if all_embeddings:
                # Make sure all embeddings are converted to numpy arrays
                np_embeddings = [np.array(emb) for emb in all_embeddings]
                avg_embedding = np.mean(np_embeddings, axis=0).tolist()
                
                if self.verbose:
                    logger.info(f"Successfully averaged {len(all_embeddings)} chunk embeddings")
                return avg_embedding
            else:
                logger.warning("Could not extract embeddings from any chunk. Returning zero embedding.")
                return [0.0] * 768  # Return zero embedding
        else:
            # Text is small enough, process as a single chunk
            # Handle rate limiting
            self._handle_rate_limiting()
            
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                
                embedding = result["embedding"]
                if embedding and len(embedding) > 0:
                    return embedding
                else:
                    logger.warning("Empty embedding returned. Returning zero embedding.")
                    return [0.0] * 768  # Return zero embedding
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                return [0.0] * 768  # Return zero embedding
    
    @retry(
        retry=retry_if_exception_type((genai.types.BlockedPromptException, genai.types.InternalServerError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def generate_text(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        top_k: int = 64,
        json_mode: bool = False,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text using Gemini.
        
        Args:
            prompt: Text prompt to send to Gemini
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            json_mode: Whether to request JSON output
            system_prompt: Optional system prompt
            
        Returns:
            str: Generated text
        """
        # Handle rate limiting
        self._handle_rate_limiting()
        
        # Create generation config
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
            
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
        
        # Create the model instance
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
        
        # Build chat session
        chat_session = model.start_chat()
        
        # Add system prompt if provided
        if system_prompt:
            chat_session.send_message(
                genai.types.content.Content(
                    parts=[genai.types.content.Part.from_text(system_prompt)],
                    role="model"
                )
            )
        
        # Send the user prompt and get the response
        response = chat_session.send_message(prompt)
        return response.text
    
    def generate_structured_response(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """Generate a structured JSON response.
        
        Args:
            prompt: Text prompt to send to Gemini
            output_schema: Schema for the expected output
            system_prompt: Optional system prompt
            temperature: Temperature parameter for generation
            
        Returns:
            Dict[str, Any]: Structured response
        """
        # Build a prompt that requests JSON output according to the schema
        schema_str = json.dumps(output_schema, indent=2)
        
        json_prompt = f"""{prompt}

Please respond with a valid JSON object that follows this schema:
```json
{schema_str}
```

Ensure your response contains ONLY the JSON object and nothing else."""

        # Generate with JSON mode
        response_text = self.generate_text(
            prompt=json_prompt,
            temperature=temperature,
            json_mode=True,
            system_prompt=system_prompt
        )
        
        # Parse the response
        try:
            # Try to parse the response as JSON
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            # If parsing fails, try to extract JSON from the text
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # Look for JSON between triple backticks
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    logger.error("Failed to parse extracted JSON")
            
            # Last resort: try to fix common JSON issues
            try:
                import ast
                # Replace single quotes with double quotes
                fixed_text = response_text.replace("'", '"')
                # Try to parse with ast first to catch Python dict syntax
                parsed = ast.literal_eval(fixed_text)
                return parsed if isinstance(parsed, dict) else {"error": "Failed to parse response"}
            except (SyntaxError, ValueError):
                logger.error("All JSON parsing attempts failed")
                return {"error": "Failed to parse model response"}
    
    def stream_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        """Stream generated text from Gemini.
        
        Args:
            prompt: Text prompt to send to Gemini
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            system_prompt: Optional system prompt
            
        Yields:
            str: Chunks of generated text
        """
        # Handle rate limiting
        self._handle_rate_limiting()
        
        # Create generation config
        generation_config = {
            "temperature": temperature,
        }
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        # Create the model instance
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
        
        # Build chat session
        chat_session = model.start_chat()
        
        # Add system prompt if provided
        if system_prompt:
            chat_session.send_message(
                genai.types.content.Content(
                    parts=[genai.types.content.Part.from_text(system_prompt)],
                    role="model"
                )
            )
        
        # Send the user prompt and stream the response
        response = chat_session.send_message(
            prompt,
            stream=True
        )
        
        for chunk in response:
            yield chunk.text