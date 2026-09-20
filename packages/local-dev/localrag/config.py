from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    ollama_host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.1:8b"
    lancedb_uri: str = "./data/lancedb"
    docs_dir: str = "./docs"

    class Config:
        env_prefix = ""


settings = Settings()
Path(settings.docs_dir).mkdir(parents=True, exist_ok=True)
Path(settings.lancedb_uri).mkdir(parents=True, exist_ok=True)