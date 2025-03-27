# Token counting and management

import tiktoken

def count_tokens(text, encoding_name="cl100k_base"):
    """Count the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

def truncate_to_token_limit(text, max_tokens, encoding_name="cl100k_base"):
    """Truncate text to stay within a token limit."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens) + "\n[Text truncated to fit token limit...]"

def estimate_tokens_from_characters(char_count):
    """Roughly estimate tokens from character count."""
    # A very rough estimate - typically for English, tokens are around 4 characters
    return char_count // 4

def truncate_conversation_history(history, max_tokens, encoding_name="cl100k_base"):
    """Truncate conversation history to fit within a token limit."""
    encoding = tiktoken.get_encoding(encoding_name)
    
    # Estimate the serialized format
    serialized = []
    for msg in history:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        serialized.append(f"{role.upper()}: {content}\n\n")
    
    serialized_text = "".join(serialized)
    tokens = encoding.encode(serialized_text)
    
    if len(tokens) <= max_tokens:
        return history
    
    # Start removing from the beginning of history until we're within the limit
    while len(history) > 2 and len(tokens) > max_tokens:  # Keep at least last user and assistant message
        history.pop(0)  # Remove oldest message
        
        # Recalculate tokens
        serialized = []
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            serialized.append(f"{role.upper()}: {content}\n\n")
        
        serialized_text = "".join(serialized)
        tokens = encoding.encode(serialized_text)
    
    return history