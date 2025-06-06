from typing import Optional, List
import asyncio
from config.config import CONFIG
from utils.logging_config_helper import get_configured_logger, LogLevel

from embedding.azure_oai_embedding import (
    get_azure_embedding,
    get_azure_batch_embeddings,
)

logger = get_configured_logger("embedding_wrapper")

async def get_embedding(
    text: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 30,
) -> List[float]:
    provider = provider or CONFIG.preferred_embedding_provider
    if provider != "azure_openai":
        raise ValueError(f"Unsupported embedding provider '{provider}'")

    provider_config = CONFIG.get_embedding_provider(provider)
    model_id = model or (provider_config.model if provider_config else None)
    if not model_id:
        raise ValueError("No embedding model specified")

    try:
        return await asyncio.wait_for(
            get_azure_embedding(text, model=model_id), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error("Embedding request timed out")
        raise

async def batch_get_embeddings(
    texts: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 60,
) -> List[List[float]]:
    provider = provider or CONFIG.preferred_embedding_provider
    if provider != "azure_openai":
        raise ValueError(f"Unsupported embedding provider '{provider}'")

    provider_config = CONFIG.get_embedding_provider(provider)
    model_id = model or (provider_config.model if provider_config else None)
    if not model_id:
        raise ValueError("No embedding model specified")

    try:
        return await asyncio.wait_for(
            get_azure_batch_embeddings(texts, model=model_id), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error("Batch embedding request timed out")
        raise
