"""
Centralized configuration for Savarkar GPT.

ALL settings are loaded from environment variables via .env file.
This is the SINGLE SOURCE OF TRUTH — no hardcoded values anywhere else.
Change any parameter in .env without touching a single line of code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)


class Settings:
    """Application settings loaded from environment variables."""

    # --- Paths ---
    PROJECT_ROOT: Path = _PROJECT_ROOT
    DATA_DIR: Path = _PROJECT_ROOT / os.getenv("DATA_DIR", "json_output")

    # --- Pinecone ---
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "savarkar-gpt")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    PINECONE_METRIC: str = os.getenv("PINECONE_METRIC", "cosine")
    PINECONE_UPSERT_BATCH_SIZE: int = int(os.getenv("PINECONE_UPSERT_BATCH_SIZE", "100"))
    PINECONE_METADATA_TEXT_LIMIT: int = int(os.getenv("PINECONE_METADATA_TEXT_LIMIT", "1000"))

    # --- Google Gemini ---
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))

    # --- Embedding ---
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2"
    )
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

    # --- Chunking ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    CHUNK_SEPARATORS: list[str] = os.getenv(
        "CHUNK_SEPARATORS", r"\n\n,\n,. , ,"
    ).split(",")

    # --- RAG ---
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    @classmethod
    def _parse_separators(cls) -> list[str]:
        """Parse chunk separators, converting escape sequences."""
        raw = os.getenv("CHUNK_SEPARATORS", r"\n\n,\n,. , ,")
        parts = raw.split(",")
        parsed = []
        for part in parts:
            part = part.replace(r"\n", "\n")
            parsed.append(part)
        return parsed

    @classmethod
    def get_chunk_separators(cls) -> list[str]:
        """Return parsed chunk separator list."""
        return cls._parse_separators()

    @classmethod
    def validate(cls) -> list[str]:
        """Validate that all required settings are present. Returns list of errors."""
        errors = []
        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY is not set in .env")
        if not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is not set in .env")
        if not cls.DATA_DIR.exists():
            errors.append(f"DATA_DIR does not exist: {cls.DATA_DIR}")
        return errors

    @classmethod
    def print_config(cls):
        """Print current configuration (masks sensitive keys)."""

        def mask(key: str) -> str:
            if not key:
                return "(NOT SET)"
            return key[:4] + "****" + key[-4:] if len(key) > 8 else "****"

        print("=" * 50)
        print("Savarkar GPT Configuration")
        print("=" * 50)
        print(f"  Project Root:          {cls.PROJECT_ROOT}")
        print(f"  Data Directory:        {cls.DATA_DIR}")
        print()
        print(f"  Pinecone API Key:      {mask(cls.PINECONE_API_KEY)}")
        print(f"  Pinecone Index:        {cls.PINECONE_INDEX_NAME}")
        print(f"  Pinecone Cloud:        {cls.PINECONE_CLOUD}")
        print(f"  Pinecone Region:       {cls.PINECONE_REGION}")
        print(f"  Pinecone Metric:       {cls.PINECONE_METRIC}")
        print(f"  Pinecone Upsert Batch: {cls.PINECONE_UPSERT_BATCH_SIZE}")
        print(f"  Pinecone Text Limit:   {cls.PINECONE_METADATA_TEXT_LIMIT}")
        print()
        print(f"  Google API Key:        {mask(cls.GOOGLE_API_KEY)}")
        print(f"  LLM Model:             {cls.LLM_MODEL_NAME}")
        print(f"  LLM Temperature:       {cls.LLM_TEMPERATURE}")
        print(f"  LLM Max Tokens:        {cls.LLM_MAX_OUTPUT_TOKENS}")
        print()
        print(f"  Embedding Model:       {cls.EMBEDDING_MODEL_NAME}")
        print(f"  Embedding Dimension:   {cls.EMBEDDING_DIMENSION}")
        print(f"  Embedding Batch Size:  {cls.EMBEDDING_BATCH_SIZE}")
        print()
        print(f"  Chunk Size:            {cls.CHUNK_SIZE}")
        print(f"  Chunk Overlap:         {cls.CHUNK_OVERLAP}")
        print(f"  Chunk Separators:      {cls.get_chunk_separators()}")
        print()
        print(f"  Top-K Results:         {cls.TOP_K_RESULTS}")
        print("=" * 50)


settings = Settings()
