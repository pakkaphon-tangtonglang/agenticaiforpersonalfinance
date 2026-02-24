"""
Configuration management using Pydantic settings.

Loads settings from environment variables and .env file with multi-provider LLM support.
"""

from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings with multi-provider LLM support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Personal Finance AI", description="Application name")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=True, description="Debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    # Database
    db_url: str = Field(default="sqlite:///./finance_ai.db", description="Database connection URL")

    # LLM Provider Selection (Google Gemini or OLLAMA/THALLE only)
    llm_provider: Literal["google", "ollama"] = Field(
        default="google", description="LLM provider to use (google or ollama)"
    )

    # Google Gemini Configuration
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    google_model: str = Field(default="gemini-pro", description="Google model name")

    # OLLAMA Configuration (for THALLE and other local models)
    ollama_base_url: str = Field(default="http://localhost:11434", description="OLLAMA base URL")
    ollama_model: str = Field(default="THALLE", description="OLLAMA model name")

    # General LLM Settings
    llm_max_tokens: int = Field(default=4000, ge=1, le=8000, description="Max tokens")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="LLM temperature")

    # RAG Configuration
    rag_embedding_provider: Literal["google", "sentence_transformers"] = Field(
        default="google", description="Embedding provider (google or sentence_transformers)"
    )
    rag_embedding_model: str = Field(
        default="models/embedding-001", description="Google embedding model name"
    )
    rag_sentence_transformer_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Sentence-transformers model name",
    )
    rag_chroma_persist_directory: str = Field(
        default="./data/chroma_db", description="ChromaDB persistence directory"
    )
    rag_chroma_collection_name: str = Field(
        default="finance_knowledge", description="ChromaDB collection name"
    )
    rag_chunk_size: int = Field(
        default=500, ge=100, le=2000, description="Text chunk size in characters"
    )
    rag_chunk_overlap: int = Field(
        default=50, ge=0, le=500, description="Overlap between chunks in characters"
    )
    rag_search_top_k: int = Field(
        default=3, ge=1, le=10, description="Number of results to return from retrieval"
    )
    rag_knowledge_base_directory: str = Field(
        default="docs/knowledge_base", description="Directory containing knowledge base documents"
    )

    # API
    api_host: str = Field(default="127.0.0.1", description="API host")  # nosec B104
    api_port: int = Field(default=8000, ge=1024, le=65535, description="API port")


def get_settings() -> Settings:
    """
    Get application settings instance.

    Returns:
        Settings: Configured settings object

    Raises:
        ValueError: If required settings are missing or invalid

    Example:
        >>> settings = get_settings()
        >>> print(settings.llm_provider)
        'google'
    """
    return Settings()
