"""Async HTTP client for OpenAI-compatible APIs."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from chatty.config import Config


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
    """Async client for OpenAI-compatible chat completions API."""

    def __init__(self, config: "Config") -> None:
        """Initialize the client with configuration."""
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> httpx.AsyncClient:
        """Build httpx client with institutional network settings."""
        # CA bundle: path to PEM file, or True for system default
        verify: bool | str = True
        if self.config.ca_bundle:
            verify = self.config.ca_bundle
        if not self.config.verify_tls:
            verify = False

        # Proxy configuration
        proxy = self.config.http_proxy if self.config.http_proxy else None

        return httpx.AsyncClient(
            verify=verify,
            proxy=proxy,
            timeout=httpx.Timeout(self.config.timeout_s),
        )

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
        """Non-streaming chat completion."""
        client = await self._get_client()
        url = f"{self.config.base_url}/chat/completions"

        payload = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.config.temperature,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.config.get_api_key()}",
            "Content-Type": "application/json",
        }

        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        return AssistantMessage(
            content=data["choices"][0]["message"]["content"],
            usage=data.get("usage"),
        )

    async def _chat_stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Streaming chat completion."""
        # TODO: Implement SSE streaming with httpx-sse
        _ = messages  # Will be used when streaming is implemented
        raise NotImplementedError("Streaming not yet implemented")
        yield ""  # Make this a generator

    async def models(self) -> list[str]:
        """List available models."""
        client = await self._get_client()
        url = f"{self.config.base_url}/models"

        headers = {
            "Authorization": f"Bearer {self.config.get_api_key()}",
        }

        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        return [model["id"] for model in data.get("data", [])]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
