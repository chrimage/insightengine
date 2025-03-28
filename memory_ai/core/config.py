"""
Application configuration using Pydantic settings.

This module provides a strongly-typed configuration system with validation,
environment variable loading, and sensible defaults.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Set
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Settings for LLM providers."""
    
    # Google API settings
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    gemini_model: str = Field("gemini-2.0-flash", env="LLM_MODEL")
    embedding_model: str = Field("models/text-embedding-004", env="EMBEDDING_MODEL")
    
    # OpenAI API settings (optional)
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    
    # General LLM settings
    default_provider: str = Field("gemini", env="DEFAULT_LLM_PROVIDER")
    max_tokens: int = Field(8000, env="MAX_TOKENS")
    temperature: float = Field(0.7, env="TEMPERATURE")
    
    # Debug settings
    verbose_embeddings: bool = Field(False, env="VERBOSE_EMBEDDINGS")
    debug_embeddings: bool = Field(False, env="DEBUG_EMBEDDINGS")
    
    @field_validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v
    
    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "LLMSettings":
        """Validate that credentials exist for the selected provider."""
        if self.default_provider == "gemini" and not self.google_api_key:
            raise ValueError("Google API key is required when using Gemini as the default provider")
        if self.default_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI as the default provider")
        return self
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class DatabaseSettings(BaseSettings):
    """Settings for database connections."""
    
    # Main database settings
    db_path: str = Field("memory.db", env="DB_PATH")
    vector_db_type: str = Field("chroma", env="VECTOR_DB_TYPE")
    vector_db_path: str = Field("vector_db", env="VECTOR_DB_PATH")
    
    # Vector search parameters
    embedding_dimension: int = Field(768, env="EMBEDDING_DIMENSION")
    vector_distance_metric: str = Field("cosine", env="VECTOR_DISTANCE_METRIC")
    
    # Database performance
    connection_pool_size: int = Field(5, env="DB_POOL_SIZE")
    
    @field_validator("vector_db_type")
    def validate_vector_db_type(cls, v: str) -> str:
        """Validate vector database type."""
        allowed_types = {"chroma", "faiss", "hnswlib"}
        if v.lower() not in allowed_types:
            raise ValueError(f"Vector DB type must be one of: {', '.join(allowed_types)}")
        return v.lower()
    
    @field_validator("vector_distance_metric")
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric."""
        allowed_metrics = {"cosine", "l2", "ip"}
        if v.lower() not in allowed_metrics:
            raise ValueError(f"Distance metric must be one of: {', '.join(allowed_metrics)}")
        return v.lower()
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class MemorySettings(BaseSettings):
    """Settings for memory management."""
    
    # Context window management
    max_context_tokens: int = Field(8000, env="MAX_CONTEXT_TOKENS")
    
    # Memory quality thresholds
    quality_threshold: float = Field(0.6, env="QUALITY_THRESHOLD")
    days_threshold: int = Field(180, env="DAYS_THRESHOLD")
    
    # Chunking settings
    chunk_size: int = Field(3, env="CHUNK_SIZE")
    chunk_overlap: int = Field(1, env="CHUNK_OVERLAP")
    
    # Token budgets for different memory types (percentages)
    rolling_summary_budget: float = Field(0.3, env="ROLLING_SUMMARY_BUDGET")
    specific_memory_budget: float = Field(0.4, env="SPECIFIC_MEMORY_BUDGET")
    insight_budget: float = Field(0.1, env="INSIGHT_BUDGET")
    conversation_history_budget: float = Field(0.2, env="CONVERSATION_HISTORY_BUDGET")
    
    @field_validator("quality_threshold")
    def validate_quality_threshold(cls, v: float) -> float:
        """Validate quality threshold is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Quality threshold must be between 0 and 1")
        return v
    
    @model_validator(mode="after")
    def validate_budget_sum(self) -> "MemorySettings":
        """Validate that token budgets sum to 1."""
        budget_sum = (
            self.rolling_summary_budget +
            self.specific_memory_budget +
            self.insight_budget +
            self.conversation_history_budget
        )
        if abs(budget_sum - 1.0) > 0.001:  # Allow for small floating point errors
            raise ValueError(f"Token budgets must sum to 1.0, got {budget_sum}")
        return self
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class AppSettings(BaseSettings):
    """Application-wide settings."""
    
    # Processing settings
    batch_size: int = Field(10, env="BATCH_SIZE")
    processing_threads: int = Field(4, env="PROCESSING_THREADS")
    
    # Logging settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    
    # Application paths
    data_dir: str = Field("data", env="DATA_DIR")
    temp_dir: str = Field("tmp", env="TEMP_DIR")
    
    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {', '.join(allowed_levels)}")
        return v.upper()
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class Settings(BaseSettings):
    """Root settings container."""
    
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    app: AppSettings = Field(default_factory=AppSettings)
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Create a singleton instance for global access
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings.
    
    Returns:
        Settings: The application settings.
    """
    return settings