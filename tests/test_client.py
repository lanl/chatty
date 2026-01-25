"""Tests for OpenAI-compatible async client."""

import json
from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from pydantic import SecretStr

from chatty.client.openai_client import (
    APIError,
    AssistantMessage,
    AuthenticationError,
    Message,
    OpenAIClient,
    RateLimitError,
)
from chatty.config import Config


@pytest.fixture
def mock_config() -> Config:
    """Create a mock config for testing."""
    return Config(
        base_url="https://api.test.com/v1",
        api_key=SecretStr("test-key"),
        model="gpt-4",
        temperature=0.7,
        timeout_s=30,
    )


@pytest.fixture
def client(mock_config: Config) -> OpenAIClient:
    """Create a client instance for testing."""
    return OpenAIClient(mock_config)


@pytest.fixture
def sample_messages() -> list[Message]:
    """Sample messages for testing."""
    return [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello!"),
    ]


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self) -> None:
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message_creation(self) -> None:
        """Test creating an assistant message."""
        msg = AssistantMessage(content="Hi there!", usage={"total_tokens": 10})
        assert msg.content == "Hi there!"
        assert msg.usage == {"total_tokens": 10}

    def test_assistant_message_no_usage(self) -> None:
        """Test assistant message without usage."""
        msg = AssistantMessage(content="Response")
        assert msg.usage is None


class TestOpenAIClientNonStreaming:
    """Tests for non-streaming chat completion."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_non_streaming_success(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test successful non-streaming chat."""
        respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Hello back!"}}],
                    "usage": {"total_tokens": 15},
                },
            )
        )

        result = await client.chat(sample_messages, stream=False)

        assert isinstance(result, AssistantMessage)
        assert result.content == "Hello back!"
        assert result.usage == {"total_tokens": 15}

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_request_format(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test that request is formatted correctly."""
        route = respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "OK"}}]},
            )
        )

        await client.chat(sample_messages, stream=False)

        assert route.called
        request = route.calls.last.request
        body = json.loads(request.content)

        assert body["model"] == "gpt-4"
        assert body["temperature"] == 0.7
        assert body["stream"] is False
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_authentication_header(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test that auth header is set correctly."""
        route = respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "OK"}}]},
            )
        )

        await client.chat(sample_messages, stream=False)

        request = route.calls.last.request
        assert request.headers["Authorization"] == "Bearer test-key"

        await client.close()


class TestOpenAIClientStreaming:
    """Tests for streaming chat completion."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_streaming_success(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test successful streaming chat."""
        sse_data = (
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            b"data: [DONE]\n\n"
        )

        respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=sse_data,
                headers={"content-type": "text/event-stream"},
            )
        )

        result = await client.chat(sample_messages, stream=True)
        stream = cast(AsyncIterator[str], result)
        tokens = [token async for token in stream]

        assert tokens == ["Hello", " world"]

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_streaming_empty_deltas(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test streaming handles empty deltas."""
        sse_data = (
            b'data: {"choices":[{"delta":{}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
            b"data: [DONE]\n\n"
        )

        respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=sse_data,
                headers={"content-type": "text/event-stream"},
            )
        )

        result = await client.chat(sample_messages, stream=True)
        stream = cast(AsyncIterator[str], result)
        tokens = [token async for token in stream]

        assert tokens == ["Hi"]

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_streaming_malformed_json(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test streaming skips malformed JSON."""
        sse_data = (
            b"data: not-json\n\n"
            b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
            b"data: [DONE]\n\n"
        )

        respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=sse_data,
                headers={"content-type": "text/event-stream"},
            )
        )

        result = await client.chat(sample_messages, stream=True)
        stream = cast(AsyncIterator[str], result)
        tokens = [token async for token in stream]

        assert tokens == ["OK"]

        await client.close()


class TestOpenAIClientRetry:
    """Tests for retry logic."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_429(self, client: OpenAIClient, sample_messages: list[Message]) -> None:
        """Test client retries on rate limit."""
        call_count = 0

        def rate_limit_then_succeed(_request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return httpx.Response(429, headers={"Retry-After": "0.01"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

        respx.post("https://api.test.com/v1/chat/completions").mock(
            side_effect=rate_limit_then_succeed
        )

        result = await client.chat(sample_messages, stream=False)

        assert isinstance(result, AssistantMessage)
        assert result.content == "OK"
        assert call_count == 2

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_500(self, client: OpenAIClient, sample_messages: list[Message]) -> None:
        """Test client retries on server error."""
        call_count = 0

        def server_error_then_succeed(_request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return httpx.Response(500, text="Internal Server Error")
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

        respx.post("https://api.test.com/v1/chat/completions").mock(
            side_effect=server_error_then_succeed
        )

        # Patch sleep to speed up test
        with patch("chatty.client.openai_client.asyncio.sleep", new=AsyncMock()):
            result = await client.chat(sample_messages, stream=False)

        assert isinstance(result, AssistantMessage)
        assert call_count == 2

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_exhausted(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test RateLimitError after retries exhausted."""
        respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "0.01"})
        )

        with (
            patch("chatty.client.openai_client.asyncio.sleep", new=AsyncMock()),
            pytest.raises(RateLimitError) as exc_info,
        ):
            await client.chat(sample_messages, stream=False)

        assert "Rate limited after 3 attempts" in str(exc_info.value)

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_with_http_date_retry_after(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test retry handles HTTP-date format Retry-After header.

        Some servers return Retry-After as an HTTP-date instead of seconds.
        The client should fall back to exponential backoff when it can't
        parse the value as a float.
        """
        call_count = 0

        def http_date_then_succeed(_request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                # HTTP-date format (RFC 7231)
                return httpx.Response(
                    429,
                    headers={"Retry-After": "Sat, 25 Jan 2026 12:00:00 GMT"},
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "OK"}}]},
            )

        respx.post("https://api.test.com/v1/chat/completions").mock(
            side_effect=http_date_then_succeed
        )

        # Patch sleep to speed up test and verify it was called
        with patch("chatty.client.openai_client.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await client.chat(sample_messages, stream=False)

        assert isinstance(result, AssistantMessage)
        assert result.content == "OK"
        assert call_count == 2
        # Should have slept (exponential backoff fallback)
        mock_sleep.assert_called()

        await client.close()


class TestOpenAIClientErrors:
    """Tests for error handling."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_authentication_error_401(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test 401 raises AuthenticationError."""
        respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid key"}})
        )

        with pytest.raises(AuthenticationError) as exc_info:
            await client.chat(sample_messages, stream=False)

        assert "Invalid API key" in str(exc_info.value)

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_authentication_error_403(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test 403 raises AuthenticationError."""
        respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(403, json={"error": {"message": "Forbidden"}})
        )

        with pytest.raises(AuthenticationError) as exc_info:
            await client.chat(sample_messages, stream=False)

        assert "does not have access" in str(exc_info.value)

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_api_error_404(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test 404 raises APIError."""
        respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(404)
        )

        with pytest.raises(APIError) as exc_info:
            await client.chat(sample_messages, stream=False)

        assert "Endpoint not found" in str(exc_info.value)

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_streaming_auth_error(
        self, client: OpenAIClient, sample_messages: list[Message]
    ) -> None:
        """Test streaming handles auth errors."""
        respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Bad key"}})
        )

        result = await client.chat(sample_messages, stream=True)
        stream = cast(AsyncIterator[str], result)

        with pytest.raises(AuthenticationError):
            async for _ in stream:
                pass

        await client.close()


class TestOpenAIClientModels:
    """Tests for models endpoint."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models(self, client: OpenAIClient) -> None:
        """Test listing available models."""
        respx.get("https://api.test.com/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "gpt-4"},
                        {"id": "gpt-3.5-turbo"},
                    ]
                },
            )
        )

        models = await client.models()

        assert models == ["gpt-4", "gpt-3.5-turbo"]

        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_empty(self, client: OpenAIClient) -> None:
        """Test listing models when none available."""
        respx.get("https://api.test.com/v1/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        models = await client.models()

        assert models == []

        await client.close()
