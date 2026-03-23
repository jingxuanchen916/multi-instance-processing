"""
Client Manager for LLM API Clients
Manages a pool of OpenRouter clients with concurrency control.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from .client import OpenRouterClient

logger = logging.getLogger(__name__)


class ClientManager:
    """
    Manages a pool of LLM clients with async task queue and concurrency control.
    Uses async-only approach: sequential execution = async with concurrency=1.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_concurrent: int = 5,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        """
        Initialize ClientManager with concurrency control.
        
        Args:
            api_key: API key (if None, loads from .env)
            max_concurrent: Maximum concurrent API requests (1 = sequential)
            max_retries: Maximum retry attempts per request
            base_delay: Initial delay for exponential backoff
            max_delay: Maximum delay between retries
        """
        self.max_concurrent = max_concurrent
        
        # Create a single client (we'll use semaphore for concurrency control)
        self.client = OpenRouterClient(
            api_key=api_key,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay
        )
        
        # Semaphore to limit concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(
            f"ClientManager initialized with max_concurrent={max_concurrent}"
        )
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 20000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request with concurrency control.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4o")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate (default: 20000)
            **kwargs: Additional parameters
            
        Returns:
            API response as dictionary
        """
        async with self.semaphore:
            return await self.client.async_chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
    
    async def execute_batch(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of chat completion requests concurrently.
        
        Args:
            tasks: List of task dicts, each containing parameters for chat_completion
                   Example: {"model": "...", "messages": [...], "temperature": 0}
        
        Returns:
            List of responses in same order as tasks
        """
        async def execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a single task with error handling."""
            try:
                return await self.chat_completion(**task)
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")
                raise
        
        # Execute all tasks concurrently (semaphore controls actual concurrency)
        results = await asyncio.gather(
            *[execute_task(task) for task in tasks],
            return_exceptions=True
        )
        
        return results
    
    def get_concurrency_level(self) -> int:
        """Get current concurrency level."""
        return self.max_concurrent
    
    def get_available_slots(self) -> int:
        """Get number of available execution slots."""
        return self.semaphore._value
