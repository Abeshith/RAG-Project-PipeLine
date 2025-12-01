import os
from typing import Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from project.utils.config_loader import load_config
from project.logger.logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    def __init__(self, config_path: str = None):
        load_dotenv()
        self.config = load_config(config_path)
        self._load_api_keys()
        logger.info("ModelLoader initialized")
    
    def _load_api_keys(self):
        groq_key = os.getenv('GROQ_API_KEY')
        
        if groq_key:
            os.environ['GROQ_API_KEY'] = groq_key
            logger.info("GROQ API key loaded")
        
    
    def load_llm(self) -> Any:
        llm_config = self.config.get('llm', {})
        provider = llm_config.get('provider', 'langchain_groq')
        
        try:
            if provider == 'langchain_groq':
                model = ChatGroq(
                    model=llm_config.get('model', 'openai/gpt-oss-20b'),
                    temperature=llm_config.get('temperature', 0.1),
                    max_tokens=llm_config.get('max_tokens', 2048)
                )
                logger.info(f"Loaded Groq LLM: {llm_config.get('model')}")
                return model
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            raise
    
    def load_embeddings(self) -> Any:
        embed_config = self.config.get('embedding_model', {})
        provider = embed_config.get('provider', 'fastembedding')
        
        try:
            if provider == 'fastembedding':
                embeddings = FastEmbedEmbeddings(
                    model_name=embed_config.get('model_name', 'BAAI/bge-small-en-v1.5')
                )
                logger.info(f"Loaded FastEmbed: {embed_config.get('model_name')}")
                return embeddings
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            raise
