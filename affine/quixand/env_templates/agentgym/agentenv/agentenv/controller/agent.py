from .types import APIConversationMessage

import time
import json
import logging
from typing import Tuple, Optional, Dict, Any, List
import httpx

logger = logging.getLogger(__name__)


class APIAgent:    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 0,
        temperature: float = 1,
        top_p: float = 1,
        timeout: float = 300.0,
    ) -> None:
        """
        Initialize the API agent with configuration parameters.

        Args:
            api_key: API key for authentication
            base_url: Base URL of the API endpoint
            model: Model identifier to use
            max_tokens: Maximum tokens in the response
            temperature: Temperature parameter for generation
            top_p: Top-p parameter for generation
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout
        )

    def __del__(self):
        """Ensure the HTTP client is properly closed."""
        if hasattr(self, 'client'):
            self.client.close()

    def _build_request_payload(
        self,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Build the request payload for the chat completion API.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Request payload dictionary
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if self.max_tokens and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens

        return payload

    def _parse_response(self, response_data: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """
        Parse the API response to extract content and reasoning.
        
        Args:
            response_data: Response JSON from the API
            
        Returns:
            Tuple of (content, reasoning_content)
        """
        try:
            choice = response_data["choices"][0]
            message = choice["message"]
            
            content = message.get("content", "")
            reasoning_content = message.get("reasoning_content", None)
            
            return content, reasoning_content
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse response: {e}")
            raise ValueError(f"Invalid response format: {e}") from e

    def generate(
        self,
        conversation: List[APIConversationMessage],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Tuple[str, Optional[str]]:
        """
        Generate a response from the API based on the conversation history.
        
        Args:
            conversation: List of conversation messages
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Tuple of (response_content, reasoning_content)
        """
        # Convert conversation messages to the expected format
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation
        ]
        
        # Build the request payload
        payload = self._build_request_payload(messages)
        
        # Construct the full endpoint URL
        endpoint = f"{self.base_url}/chat/completions"
        
        # Retry logic for resilience
        for attempt in range(max_retries):
            try:
                # Make the API request
                response = self.client.post(
                    endpoint,
                    json=payload,
                )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse and return the response
                response_data = response.json()
                return self._parse_response(response_data)
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if e.response.status_code == 429:  # Rate limit
                    # Exponential backoff for rate limits
                    wait_time = retry_delay * (2 ** attempt)
                    logger.info(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif attempt == max_retries - 1:
                    raise
                else:
                    time.sleep(retry_delay)
                    
            except httpx.RequestError as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
        
        # This should never be reached due to the raise in the loop
        raise RuntimeError(f"Failed after {max_retries} attempts")
