import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class LangChainConfig:
    SERVICE_NAME: str = "python-langchain-service"
    VERSION: str = "1.0.0"
    HOST: str = os.getenv("LANGCHAIN_SERVICE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("LANGCHAIN_SERVICE_PORT", "8003"))

    LLM_API_KEY: str = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "sk-dummy"
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL") or os.getenv("OPENROUTER_URL") or "https://api.deepseek.com"
    LLM_MODEL: str = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL") or "deepseek-flash"

config = LangChainConfig()
