# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Very simple wrapper around the various LLM providers.  

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.

"""

from typing import Optional, Dict, Any
from config.config import CONFIG
import asyncio

from llm.azure_oai import provider as azure_openai_provider

from utils.logging_config_helper import get_configured_logger, LogLevel
logger = get_configured_logger("llm_wrapper")


async def ask_llm(
    prompt: str,
    schema: Dict[str, Any],
    provider: Optional[str] = None,
    level: str = "low",
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Route an LLM request to the Azure OpenAI endpoint.
    
    Args:
        prompt: The text prompt to send to the LLM
        schema: JSON schema that the response should conform to
        provider: The LLM endpoint to use (if None, use preferred endpoint from config)
        level: The model tier to use ('low' or 'high')
        timeout: Request timeout in seconds
        
    Returns:
        Parsed JSON response from the LLM
        
    Raises:
        ValueError: If the endpoint is unknown or response cannot be parsed
        TimeoutError: If the request times out
    """
    provider_name = provider or CONFIG.preferred_llm_endpoint
    logger.debug(f"Initiating LLM request with provider: {provider_name}, level: {level}")
    logger.debug(f"Prompt preview: {prompt[:100]}...")
    logger.debug(f"Schema: {schema}")

    if provider_name != "azure_openai":
        error_msg = f"Unsupported provider '{provider_name}'"
        logger.error(error_msg)
        raise ValueError(error_msg)

    provider_config = CONFIG.get_llm_provider("azure_openai")
    if not provider_config or not provider_config.models:
        error_msg = "Missing Azure OpenAI model configuration"
        logger.error(error_msg)
        raise ValueError(error_msg)

    model_id = getattr(provider_config.models, level)
    logger.debug(f"Using model: {model_id}")

    try:
        logger.debug("Calling Azure OpenAI provider")
        result = await asyncio.wait_for(
            azure_openai_provider.get_completion(prompt, schema, model=model_id),
            timeout=timeout
        )
        logger.debug(f"{provider_name} response received, size: {len(str(result))} chars")
        return result
        
    except asyncio.TimeoutError:
        logger.error(f"LLM call timed out after {timeout}s with provider {provider_name}")
        raise
    except Exception as e:
        error_msg = f"LLM call failed: {type(e).__name__}: {str(e)}"
        logger.error(f"Error with provider {provider_name}: {error_msg}")
        print(f"LLM Error ({provider_name}): {type(e).__name__}: {str(e)}")
        logger.log_with_context(
            LogLevel.ERROR,
            "LLM call failed",
            {
                "endpoint": provider_name,
                "llm_type": "azure_openai",
                "model": model_id,
                "level": level,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )

        raise
