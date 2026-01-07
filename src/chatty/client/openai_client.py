"""Async HTTP client for OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
import httpx_sse

from chatty.client.http import build_async_client

if TYPE_CHECKING:
    from chatty.config import Config


# Retry configuration
MAX_RETRIES = 2
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
BASE_BACKOFF_SECONDS = 1.0


class ChattyClientError(Exception):
    """Base client error with actionable message."""

    pass


class AuthenticationError(ChattyClientError):
    """401/403 - Invalid API key."""

    pass


class RateLimitError(ChattyClientError):
    """429 - Rate limited (after retries exhausted)."""

    pass


class APIError(ChattyClientError):
    """Other API errors."""

    pass


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class AssistantMessage:
    """Response from the LLM."""

    content: str
    usage: dict[str, int] | None = None


class OpenAIClient:
    """Async client for OpenAI-compatible chat completions API.

    Features:
    - Exponential backoff retry on 429/5xx (max 2 retries)
    - SSE streaming with proper cancellation
    - Actionable error messages
    """

    def __init__(self, config: Config) -> None:
        """Initialize the client with configuration."""
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = build_async_client(self.config, timeout=float(self.config.timeout_s))
        return self._client

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.config.get_api_key()}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[Message], stream: bool = False) -> dict[str, Any]:
        """Build the request payload."""
        return {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.config.temperature,
            "stream": stream,
        }

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make request with exponential backoff retry.

        Retries on:
        - 429 Too Many Requests
        - 5xx Server Errors
        - Connection errors

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments passed to httpx

        Returns:
            httpx.Response on success

        Raises:
            RateLimitError: After retries exhausted on 429
            APIError: After retries exhausted on 5xx
            ChattyClientError: On connection errors after retries
        """
        client = await self._get_client()
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.request(method, url, **kwargs)

                # Handle specific error codes
                if response.status_code == 401:
                    raise AuthenticationError(
                        "Invalid API key. Check OPENAI_API_KEY or api_key_file."
                    )
                if response.status_code == 403:
                    raise AuthenticationError("API key does not have access to this endpoint.")
                if response.status_code == 404:
                    raise APIError(
                        f"Endpoint not found: {url}. "
                        "Check that base_url points to an OpenAI-compatible API."
                    )

                # Success - return response
                if response.status_code < 400:
                    return response

                # Retryable errors
                if response.status_code in RETRY_STATUS_CODES:
                    # Check Retry-After header
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = float(retry_after)
                    else:
                        wait_time = BASE_BACKOFF_SECONDS * (2**attempt)

                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(wait_time)
                        continue

                    # Retries exhausted
                    if response.status_code == 429:
                        raise RateLimitError(
                            f"Rate limited after {MAX_RETRIES + 1} attempts. " "Try again later."
                        )
                    raise APIError(
                        f"Server error {response.status_code} after " f"{MAX_RETRIES + 1} attempts."
                    )

                # Non-retryable error
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", response.text[:200])
                except Exception:
                    error_msg = response.text[:200]

                raise APIError(f"API error {response.status_code}: {error_msg}")

            except httpx.ConnectError as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF_SECONDS * (2**attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise ChattyClientError(
                    f"Connection failed after {MAX_RETRIES + 1} attempts: {e}"
                ) from e

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    wait_time = BASE_BACKOFF_SECONDS * (2**attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise ChattyClientError(
                    f"Request timed out after {MAX_RETRIES + 1} attempts. "
                    f"Try increasing timeout_s (currently {self.config.timeout_s}s)."
                ) from e

        # Should not reach here, but just in case
        raise ChattyClientError(f"Request failed: {last_error}")

    async def chat(
        self,
        messages: list[Message],
        stream: bool = True,
    ) -> AssistantMessage | AsyncIterator[str]:
        """Send a chat completion request.

        Args:
            messages: List of messages in the conversation
            stream: Whether to stream the response

        Returns:
            If stream=False: AssistantMessage with complete response
            If stream=True: AsyncIterator yielding tokens
        """
        if stream:
            return self._chat_stream(messages)
        return await self._chat_non_stream(messages)

    async def _chat_non_stream(self, messages: list[Message]) -> AssistantMessage:
        """Non-streaming chat completion with retry."""
        url = f"{self.config.base_url}/chat/completions"
        payload = self._build_payload(messages, stream=False)
        headers = self._get_headers()

        response = await self._request_with_retry("POST", url, json=payload, headers=headers)
        data = response.json()

        return AssistantMessage(
            content=data["choices"][0]["message"]["content"],
            usage=data.get("usage"),
        )

    async def _chat_stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Streaming chat completion with proper cancellation.

        Uses httpx-sse for SSE parsing. The async context manager ensures
        proper cleanup on cancellation (e.g., user presses Esc).

        Yields:
            Token strings as they arrive
        """
        client = await self._get_client()
        url = f"{self.config.base_url}/chat/completions"
        payload = self._build_payload(messages, stream=True)
        headers = self._get_headers()

        async with httpx_sse.aconnect_sse(
            client, "POST", url, json=payload, headers=headers
        ) as event_source:
            # Check for errors before streaming
            if event_source.response.status_code == 401:
                raise AuthenticationError("Invalid API key. Check OPENAI_API_KEY or api_key_file.")
            if event_source.response.status_code == 403:
                raise AuthenticationError("API key does not have access to this endpoint.")
            if event_source.response.status_code >= 400:
                # Read error body
                await event_source.response.aread()
                try:
                    error_data = event_source.response.json()
                    error_msg = error_data.get("error", {}).get(
                        "message", event_source.response.text[:200]
                    )
                except Exception:
                    error_msg = event_source.response.text[:200]
                raise APIError(f"API error {event_source.response.status_code}: {error_msg}")

            # Stream SSE events
            async for sse in event_source.aiter_sse():
                if sse.data == "[DONE]":
                    break

                try:
                    chunk = json.loads(sse.data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    # Skip malformed chunks
                    continue

    async def models(self) -> list[str]:
        """List available models."""
        url = f"{self.config.base_url}/models"
        headers = self._get_headers()

        response = await self._request_with_retry("GET", url, headers=headers)
        data = response.json()

        return [model["id"] for model in data.get("data", [])]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
