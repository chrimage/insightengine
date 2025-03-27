# Gemini API wrapper with optimizations

import os
import time
import random
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

import hashlib
import random

class GeminiClient:
    """Optimized client for interacting with Gemini API."""
    
    def __init__(self, api_key=None, model="gemini-2.0-flash", embedding_model="models/text-embedding-004", 
                 use_dummy_embeddings=False):
        """Initialize the Gemini client.
        
        Args:
            api_key: API key for Google Generative AI
            model: Text generation model name
            embedding_model: Embedding model name
            use_dummy_embeddings: If True, use deterministic dummy embeddings instead of calling the API
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key and not use_dummy_embeddings:
            raise ValueError("No API key provided and GOOGLE_API_KEY not set")
        
        self.model = model
        self.embedding_model = embedding_model
        self.use_dummy_embeddings = use_dummy_embeddings or os.environ.get("USE_DUMMY_EMBEDDINGS", "").lower() in ["true", "1", "yes"]
        
        if self.use_dummy_embeddings:
            print("WARNING: Using dummy embeddings since use_dummy_embeddings=True")
        else:
            genai.configure(api_key=self.api_key)
        
        # Rate limiting parameters for text-embedding-005
        self.max_requests_per_minute = 60  # More generous limit
        self.request_timestamps = deque(maxlen=self.max_requests_per_minute)
        self.max_retries = 10
        self.base_retry_delay = 10
        
        # Just for caution, still track daily embedding usage but with high limits
        self.embedding_daily_limit = 10000  # Much more generous limit
        self.embedding_daily_count = 0
        self.embedding_day = datetime.now().date()
    
    def generate_deterministic_embedding(self, text, dimension=768):
        """Generate a deterministic dummy embedding based on the text content.
        
        This creates a reproducible embedding for a given text, which is useful for testing
        or when the embedding API is not working.
        
        Args:
            text: The text to generate an embedding for
            dimension: The embedding dimension (default 768)
            
        Returns:
            A list of floats representing the embedding
        """
        if not text:
            return [0.0] * dimension
            
        # Create a deterministic seed from the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed_val = int(text_hash, 16) % (2**32)
        
        # Set the seed for reproducibility
        random.seed(seed_val)
        
        # Generate a random embedding with values between -1 and 1
        embedding = [random.uniform(-1, 1) for _ in range(dimension)]
        
        # Normalize the embedding to have a norm of 1
        import numpy as np
        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm
            
        return embedding_np.tolist()
    
    def generate_content(self, prompt, max_output_tokens=8000, temperature=0.7, 
                        top_p=0.95, top_k=64):
        """Generate content with automatic rate limiting and retries."""
        retry_count = 0
        backoff_time = self.base_retry_delay
        
        while retry_count <= self.max_retries:
            # Apply rate limiting
            self._handle_rate_limiting()
            
            try:
                # Make the API call using generativeai
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "top_p": top_p,
                    "top_k": top_k
                }
                
                model = genai.GenerativeModel(model_name=self.model, generation_config=generation_config)
                response = model.generate_content(prompt)
                return response
                
            except Exception as e:
                if self._is_rate_limit_error(e):
                    retry_count += 1
                    
                    if retry_count <= self.max_retries:
                        # Apply exponential backoff with jitter
                        jitter = random.uniform(0.8, 1.2)
                        actual_delay = backoff_time * jitter
                        
                        print(f"Rate limit exceeded. Retrying in {actual_delay:.1f}s ({retry_count}/{self.max_retries})")
                        time.sleep(actual_delay)
                        
                        # Increase backoff for next attempt
                        backoff_time = min(backoff_time * 2, 60)
                    else:
                        print(f"Max retries exceeded.")
                        raise
                else:
                    # Not a rate limit error
                    raise
        
        raise Exception("Failed after maximum retries")
    
    def generate_embedding(self, text):
        """Alias for generate_embeddings, for API compatibility."""
        return self.generate_embeddings(text)
        
    def generate_embeddings(self, text, chunk_size=1900):  # Keep below 2048 token limit
        """Generate embeddings with rate limiting and retries.
        
        For text-embedding-004, max input is 2048 tokens.
        If text is longer than chunk_size tokens, it will be split into chunks
        and the embeddings will be averaged.
        
        If use_dummy_embeddings is set to True (either in the constructor or via the
        USE_DUMMY_EMBEDDINGS environment variable), this will generate deterministic
        pseudorandom embeddings instead of calling the API.
        """
        # If using dummy embeddings, use our deterministic generator
        if self.use_dummy_embeddings:
            print(f"Generating deterministic dummy embedding for text ({len(text)} chars)")
            return self.generate_deterministic_embedding(text)
        
        # Check daily limit
        today = datetime.now().date()
        if today > self.embedding_day:
            # Reset counter for new day
            self.embedding_day = today
            self.embedding_daily_count = 0
        
        # Check if we're approaching daily limit
        if self.embedding_daily_count >= self.embedding_daily_limit:
            print(f"WARNING: Daily embedding limit approached ({self.embedding_daily_count}/{self.embedding_daily_limit})")
            print("Returning deterministic embedding to avoid exceeding quota")
            return self.generate_deterministic_embedding(text)
        
        # If text is empty, return a dummy embedding
        if not text or len(text.strip()) == 0:
            print("WARNING: Empty text provided for embedding. Returning zero embedding.")
            return [0.0] * 768

        # Check if text needs to be chunked (rough estimate: 4 chars ~ 1 token)
        if len(text) > chunk_size * 4:
            # Only print detailed chunking info in verbose mode
            verbose = os.environ.get("VERBOSE_EMBEDDINGS", "").lower() in ["true", "1", "yes"]
            if verbose:
                print(f"Text too long ({len(text)} chars), splitting into chunks...")
            
            # Simple chunking by character count - aim for 2000 tokens max per chunk
            chunks = [text[i:i + chunk_size * 4] for i in range(0, len(text), chunk_size * 4)]
            
            # Limit chunks to 5 to stay within reasonable limits
            if len(chunks) > 5:
                if verbose:
                    print(f"Too many chunks ({len(chunks)}), limiting to 5 representative chunks")
                # Take first, last, and 3 evenly spaced chunks from the middle
                if len(chunks) >= 3:
                    step = len(chunks) // 4
                    indices = [0, step, 2*step, 3*step, len(chunks)-1]
                    chunks = [chunks[i] for i in indices]
                else:
                    chunks = chunks[:5]
            
            # Get embeddings for each chunk and average them
            all_embeddings = []
            for i, chunk in enumerate(chunks):
                if verbose:
                    print(f"Processing chunk {i+1}/{len(chunks)}...")
                try:
                    chunk_embedding = self._get_single_embedding(chunk)
                    if chunk_embedding and len(chunk_embedding) > 0:
                        all_embeddings.append(chunk_embedding)
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {str(e)}")
                    # Try fallback to deterministic embedding for this chunk
                    print(f"Generating deterministic embedding for chunk {i+1} as fallback")
                    all_embeddings.append(self.generate_deterministic_embedding(chunk))
                
                # Add short delay between chunk processing to avoid rate limits
                if i < len(chunks) - 1 and not self.use_dummy_embeddings:
                    time.sleep(2)
            
            # If we have any embeddings, average them
            if all_embeddings:
                import numpy as np
                # Make sure all embeddings are converted to numpy arrays
                np_embeddings = [np.array(emb) for emb in all_embeddings]
                avg_embedding = np.mean(np_embeddings, axis=0).tolist()
                # Verify the embedding is valid
                if avg_embedding and len(avg_embedding) > 0:
                    if os.environ.get("VERBOSE_EMBEDDINGS", "").lower() in ["true", "1", "yes"]:
                        print(f"Successfully averaged {len(all_embeddings)} chunk embeddings into a vector of size {len(avg_embedding)}")
                    return avg_embedding
                else:
                    print(f"Warning: Averaging produced an invalid embedding. Using deterministic embedding instead.")
                    return self.generate_deterministic_embedding(text)
            else:
                # Return deterministic embedding if all chunks failed
                print(f"Warning: Could not extract embedding from any chunk. Using deterministic embedding instead.")
                return self.generate_deterministic_embedding(text)
        else:
            # Text is small enough, process as a single chunk
            try:
                embedding = self._get_single_embedding(text)
                if embedding and len(embedding) > 0:
                    return embedding
                else:
                    print(f"Warning: Generated an invalid embedding. Using deterministic embedding instead.")
                    return self.generate_deterministic_embedding(text)
            except Exception as e:
                print(f"Error generating embedding: {str(e)}")
                print(f"Falling back to deterministic embedding")
                return self.generate_deterministic_embedding(text)
    
    def _get_single_embedding(self, text):
        """Get embedding for a single text chunk."""
        retry_count = 0
        backoff_time = self.base_retry_delay
        
        # Check if we've hit the daily limit
        if self.embedding_daily_count >= self.embedding_daily_limit:
            print(f"Daily limit reached ({self.embedding_daily_count}/{self.embedding_daily_limit})")
            return [0.0] * 768  # Return dummy embedding
        
        while retry_count <= self.max_retries:
            # Apply rate limiting
            self._handle_rate_limiting()
            
            try:
                # Make the API call using the generativeai API
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text
                )
                
                # Increment daily usage counter
                self.embedding_daily_count += 1
                
                # Get the embeddings from the result
                # Only print debug info if DEBUG_EMBEDDINGS env var is set
                if os.environ.get("DEBUG_EMBEDDINGS", "").lower() in ["true", "1", "yes"]:
                    print("\nDEBUG - EMBEDDING RESPONSE DETAILS:")
                    print(f"Result type: {type(result)}")
                    print(f"Result attributes: {dir(result)}")
                    print(f"Raw result: {repr(result)}")
                
                # Try to extract embedding based on API documentation (Google may have changed the API again)
                # See: https://ai.google.dev/api/python/google/generativeai/embed_content
                
                # Approach 1: Direct access to embedding (latest API format)
                if hasattr(result, 'embedding'):
                    print("Found 'embedding' attribute.")
                    if isinstance(result.embedding, list):
                        print(f"embedding is a list with {len(result.embedding)} elements")
                        return result.embedding
                    else:
                        print(f"embedding is not a list, type: {type(result.embedding)}")
                        try:
                            # Try to convert whatever it is to a list
                            embedding_list = list(result.embedding)
                            print(f"Converted to list with {len(embedding_list)} elements")
                            return embedding_list
                        except Exception as e:
                            print(f"Error converting to list: {str(e)}")
                            
                # Approach 2: Access through embeddings array (older API format)
                if hasattr(result, 'embeddings') and result.embeddings:
                    print("Found 'embeddings' attribute.")
                    embedding = result.embeddings[0]
                    print(f"First embedding type: {type(embedding)}")
                    
                    if hasattr(embedding, 'values'):
                        print("Found 'values' attribute in the embedding.")
                        return embedding.values
                        
                    elif isinstance(embedding, list):
                        print(f"Embedding is a list with {len(embedding)} elements")
                        return embedding
                        
                    else:
                        try:
                            embedding_list = list(embedding)
                            print(f"Converted to list with {len(embedding_list)} elements")
                            return embedding_list
                        except Exception as e:
                            print(f"Error converting to list: {str(e)}")
                
                # Approach 3: Access values directly 
                if hasattr(result, 'values'):
                    if callable(result.values):
                        # It's a method, not an attribute
                        try:
                            values_result = result.values()
                            
                            # If it's dict_values, convert to list
                            if isinstance(values_result, type({}.values())):
                                values_list = list(values_result)
                                
                                # Check if it's a list with a single embedding list inside (this is the current format)
                                if len(values_list) == 1 and isinstance(values_list[0], list):
                                    inner_list = values_list[0]
                                    if len(inner_list) > 0 and all(isinstance(x, (int, float)) for x in inner_list[:5]):
                                        # This is the valid embedding format for the latest API version
                                        return inner_list
                                
                                # Check if the values list itself is a numeric list (could be the embedding)
                                if len(values_list) > 0 and all(isinstance(x, (int, float)) for x in values_list[:5]):
                                    return values_list
                            
                            return values_result
                        except Exception as e:
                            print(f"Error calling values() method: {str(e)}")
                    else:
                        # It's an attribute
                        print("Found 'values' attribute.")
                        return result.values
                    
                # Additional attempts for other potential attributes
                potential_attrs = ['vector', 'vectors', 'value', 'data']
                for attr in potential_attrs:
                    if hasattr(result, attr):
                        print(f"Found alternative attribute '{attr}'")
                        try:
                            value = getattr(result, attr)
                            if isinstance(value, list):
                                print(f"Attribute is a list with {len(value)} elements")
                                return value
                            else:
                                print(f"Attribute type: {type(value)}")
                                try:
                                    return list(value)
                                except:
                                    print(f"Couldn't convert to list")
                        except Exception as e:
                            print(f"Error accessing {attr}: {str(e)}")
                
                # Try accessing 'values' as a method (not an attribute)
                if hasattr(result, 'values') and callable(result.values):
                    try:
                        values_result = result.values()
                        print(f"Called result.values() method successfully: {type(values_result)}")
                        
                        # If values_result is a dict_values object, convert to list
                        if isinstance(values_result, type({}.values())):
                            values_list = list(values_result)
                            print(f"Converted dict_values to list with {len(values_list)} items")
                            
                            # Check if it's a list of embeddings
                            if len(values_list) > 0 and all(isinstance(x, (int, float)) for x in values_list[:5]):
                                print(f"Values list looks like a valid embedding vector")
                                return values_list
                            
                            # Check if it's a list with a single embedding list
                            if len(values_list) == 1 and isinstance(values_list[0], list):
                                inner_list = values_list[0]
                                if len(inner_list) > 0 and all(isinstance(x, (int, float)) for x in inner_list[:5]):
                                    print(f"Found embedding list inside values list")
                                    return inner_list
                        
                        # Try to directly extract embeddings if values_result is a dict
                        if isinstance(values_result, dict):
                            # Check for common embedding attributes
                            for key in ['embedding', 'embeddings', 'vector', 'values']:
                                if key in values_result:
                                    embedding_data = values_result[key]
                                    print(f"Found key '{key}' in values_result")
                                    
                                    # Check if it's a valid embedding
                                    if isinstance(embedding_data, list) and len(embedding_data) > 0:
                                        if all(isinstance(x, (int, float)) for x in embedding_data[:5]):
                                            print(f"Found valid embedding under key '{key}'")
                                            return embedding_data
                        
                        # If values_result is a list
                        if isinstance(values_result, list):
                            print(f"Values method returned a list with {len(values_result)} elements")
                            if len(values_result) > 0 and all(isinstance(x, (int, float)) for x in values_result[:5]):
                                return values_result
                        
                        # If values_result is iterable but not a list
                        elif hasattr(values_result, '__iter__') and not isinstance(values_result, str):
                            values_list = list(values_result)
                            print(f"Converted values() result to list with {len(values_list)} elements")
                            if len(values_list) > 0 and all(isinstance(x, (int, float)) for x in values_list[:5]):
                                return values_list
                    except Exception as e:
                        print(f"Error processing values() method result: {str(e)}")
                
                # If result itself is list-like, try that
                try:
                    if hasattr(result, '__iter__') and not isinstance(result, str):
                        result_list = list(result)
                        if result_list and len(result_list) > 0:
                            print(f"Converted result directly to list with {len(result_list)} elements")
                            return result_list
                except Exception as e:
                    print(f"Error converting result to list: {str(e)}")
                
                # Last desperate attempt: if result has __getitem__, try direct indexing
                try:
                    if hasattr(result, '__getitem__'):
                        first_item = result[0]
                        if isinstance(first_item, (int, float)):
                            # Looks like it might be directly indexable as a vector
                            vector = [result[i] for i in range(768)]  # Standard embedding size
                            print(f"Extracted vector by direct indexing, length: {len(vector)}")
                            return vector
                except Exception as e:
                    print(f"Error with direct indexing: {str(e)}")
                
                # If all else fails
                print("\nWARNING: All embedding extraction attempts failed.")
                print("Please check the Google Generative AI library documentation for updates.")
                print("Consider using a different embedding model or provider.")
                return [0.0] * 768  # Standard embedding size
                    
            except Exception as e:
                if self._is_rate_limit_error(e):
                    retry_count += 1
                    
                    if retry_count <= self.max_retries:
                        # Apply exponential backoff with jitter
                        jitter = random.uniform(0.8, 1.2)
                        actual_delay = backoff_time * jitter
                        
                        print(f"Rate limit exceeded. Retrying in {actual_delay:.1f}s ({retry_count}/{self.max_retries})")
                        time.sleep(actual_delay)
                        
                        # Increase backoff for next attempt
                        backoff_time = min(backoff_time * 2, 60)
                    else:
                        print(f"Max retries exceeded.")
                        raise
                else:
                    # Not a rate limit error
                    raise
        
        raise Exception("Failed after maximum retries")
    
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
            small_delay = ratio * 3  # Smaller delay for the more generous limit
            time.sleep(small_delay)
        
        # If we're at the limit, wait with buffer
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            # Wait until we're under the limit
            wait_time = 65 - (current_time - self.request_timestamps[0])  # 60s + 5s buffer
            print(f"Rate limit approaching. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        # Record this request
        self.request_timestamps.append(time.time())
    
    def _is_rate_limit_error(self, error):
        """Check if an error is related to rate limiting."""
        error_str = str(error).lower()
        return any(term in error_str for term in 
                  ["429", "resource exhausted", "quota", "rate limit"])