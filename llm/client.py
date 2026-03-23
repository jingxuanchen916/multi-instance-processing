"""
OpenRouter API Client
Handles LLM API calls with retry logic and error handling.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """
    Async-only client for interacting with OpenRouter API.
    Includes retry logic and error handling.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: API key (if None, loads from .env)
            base_url: OpenAI-compatible API base URL (if None, loads from .env)
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (exponential backoff)
            max_delay: Maximum delay between retries
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key. Prefer provider-neutral API_KEY and keep legacy fallback.
        self.api_key = api_key or os.getenv('API_KEY') or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key not found. "
                "Set API_KEY in .env file or pass as argument."
            )

        # Get API endpoint (OpenRouter by default).
        # Prefer provider-neutral BASE_URL and keep legacy fallbacks.
        self.base_url = base_url or os.getenv(
            'BASE_URL'
        ) or os.getenv(
            'OPENAI_BASE_URL'
        ) or os.getenv(
            'OPENROUTER_BASE_URL',
            'https://openrouter.ai/api/v1'
        )
        
        # Initialize async client
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        # Retry configuration
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        logger.info(f"OpenRouter async client initialized with base URL: {self.base_url}")
    
    async def async_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 20000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send an async chat completion request with retry logic.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4o")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate (default: 20000)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response as dictionary
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                # Prepare request parameters
                request_params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    **kwargs
                }
                
                if max_tokens:
                    request_params["max_tokens"] = max_tokens
                
                # Make async API call
                logger.debug(f"Async attempt {attempt + 1}/{self.max_retries} for model {model}")
                response = await self.async_client.chat.completions.create(**request_params)
                
                # Convert response to dict for easier handling
                result = {
                    "id": response.id,
                    "model": response.model,
                    "created": response.created,
                    "choices": [
                        {
                            "index": choice.index,
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content,
                            },
                            "finish_reason": choice.finish_reason
                        }
                        for choice in response.choices
                    ],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if response.usage else None
                }
                
                logger.debug(f"Successfully got async response from {model}")
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(phrase in error_str for phrase in [
                    'rate limit', 'rate_limit', 'ratelimit', 
                    'too many requests', '429', 'quota exceeded'
                ])
                
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"All {self.max_retries} async attempts failed for {model}: {str(e)}"
                    )
                else:
                    logger.debug(
                        f"Async attempt {attempt + 1}/{self.max_retries} failed for {model}: {str(e)}"
                    )
                
                # If this was the last attempt, raise the exception
                if attempt == self.max_retries - 1:
                    raise
                
                # Calculate exponential backoff delay
                # Use longer delays for rate limit errors
                if is_rate_limit:
                    # Rate limit errors: start with 5s and go up to 120s
                    delay = min(5.0 * (2 ** attempt), 120.0)
                    logger.warning(
                        f"Rate limit hit for {model}. Waiting {delay:.1f}s before retry..."
                    )
                else:
                    # Other errors: use standard backoff
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                
                logger.debug(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
        
        # Should not reach here, but just in case
        raise Exception(f"Failed after {self.max_retries} attempts")
