"""Tests for OpenAI-compatible async client."""

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
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
            b'data: not-json\n\ndata: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'
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
    @pytest.mark.parametrize(
        "header, expected",
        [
            ("0", 0),
            ("0.25", 0.25),
            ("60", 60),
            ("nan", 1),
            ("inf", 1),
            ("-inf", 1),
            ("-3", 1),
            ("invalid date", 1),
            ("Sat, 25 Jan 2020 12:00:00 GMT", 0),
        ],
    )
    async def test_retry_delay(self, client: OpenAIClient, header: str, expected: float) -> None:
        route = respx.get("https://api.test.com/v1/models").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": header}),
                httpx.Response(200, json={"data": []}),
            ]
        )
        with patch("chatty.client.openai_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
            assert await client.models() == []
        sleep.assert_awaited_once_with(expected)
        assert route.call_count == 2
        await client.close()

    @respx.mock
    @pytest.mark.parametrize("status, error", [(429, RateLimitError), (503, APIError)])
    @pytest.mark.parametrize("as_date", [False, True])
    async def test_long_retry_after_fails_without_sleeping(
        self, client: OpenAIClient, status: int, error: type[Exception], as_date: bool
    ) -> None:
        header = format_datetime(datetime.now(UTC) + timedelta(hours=1)) if as_date else "3600"
        route = respx.get("https://api.test.com/v1/models").respond(
            status, headers={"Retry-After": header}
        )
        with (
            patch("chatty.client.openai_client.asyncio.sleep", new_callable=AsyncMock) as sleep,
            pytest.raises(error, match=r"Try again in 3[56]\d{2} seconds"),
        ):
            await client.models()
        sleep.assert_not_awaited()
        assert route.call_count == 1
        await client.close()

    @respx.mock
    async def test_valid_integer_delay_beyond_float_range(self, client: OpenAIClient) -> None:
        route = respx.get("https://api.test.com/v1/models").respond(
            429, headers={"Retry-After": "9" * 309}
        )
        with (
            patch("chatty.client.openai_client.asyncio.sleep", new_callable=AsyncMock) as sleep,
            pytest.raises(RateLimitError, match="retry delay is too long"),
        ):
            await client.models()
        sleep.assert_not_awaited()
        assert route.call_count == 1
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
    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            ({"max_model_len": 32768}, 32768),
            ({"context_length": None, "context_window": "8192"}, 8192),
            ({"max_context_length": 4096}, 4096),
            ({"context_length": 32768.0}, 32768),
            ({"context_window": -1}, None),
            ({"context_length": True}, None),
            ({"context_length": 1.5}, None),
            ({"max_tokens": 2048}, None),
        ],
    )
    async def test_model_context_length(
        self,
        client: OpenAIClient,
        fields: dict[str, object],
        expected: int | None,
    ) -> None:
        respx.get("https://api.test.com/v1/models").respond(
            json={
                "data": [
                    {"id": "other", "context_length": 99999},
                    {"id": "unfamiliar", **fields},
                ]
            }
        )
        assert await client.get_model_context_length("unfamiliar") == expected
        await client.close()

    @respx.mock
    async def test_context_lookup_bounds_retry_after(self, client: OpenAIClient) -> None:
        client.config.timeout_s = 1
        respx.get("https://api.test.com/v1/models").respond(429, headers={"Retry-After": "60"})
        with pytest.raises(APIError, match="lookup timed out"):
            await client.get_model_context_length("unfamiliar")
        await client.close()

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
