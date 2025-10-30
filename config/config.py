"""
Configuration settings for the RAG system.
"""
from typing import Dict
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    
    @staticmethod
    def get_config() -> Dict[str, str]:
        """Return configuration as a dictionary."""
        return {
            "OPENAI_API_KEY": Config.OPENAI_API_KEY,
            "EMBEDDING_MODEL": Config.EMBEDDING_MODEL,
            "LLM_MODEL": Config.LLM_MODEL,
            "VECTOR_STORE_PATH": Config.VECTOR_STORE_PATH
        }