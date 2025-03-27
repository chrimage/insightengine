#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("No API key found in GOOGLE_API_KEY environment variable")

genai.configure(api_key=api_key)

# List all available models
print("Available models:")
for model in genai.list_models():
    print(f"- {model.name}")
    if hasattr(model, 'supported_generation_methods'):
        print(f"  Supported methods: {model.supported_generation_methods}")
    print()

# Check for embedding models specifically
print("\nEmbedding models:")
embedding_models = [model for model in genai.list_models() if "embed" in model.name.lower() or "embedding" in model.name.lower()]
for model in embedding_models:
    print(f"- {model.name}")
    if hasattr(model, 'supported_generation_methods'):
        print(f"  Supported methods: {model.supported_generation_methods}")
    print()