"""Citation numbering and reply ownership across session loads and regeneration."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import Mock

import pytest

from chatty.client.openai_client import AssistantMessage, ChattyClientError, Message, OpenAIClient
from chatty.config import Config, ConfigWithSources
from chatty.core.session import Session, SessionMetadata, save_session
from chatty.rag.litkit_provider import LitkitProvider, RetrievedChunk
from chatty.ui.app import ChatApp
from chatty.ui.widgets import ChatInput, ChatLog, CitationsWidget, MessageWidget


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ConfigWithSources:
    monkeypatch.setattr(LitkitProvider, "_check_litkit_installed", lambda _self: None)
    monkeypatch.setattr(LitkitProvider, "_validate_workspace", lambda _self: None)
    return ConfigWithSources(
        Config(
            base_url="https://test.invalid/v1",
            rag_provider="litkit",
            rag_rewrite_enabled=False,
            session_path=str(tmp_path),
            context_window=32000,
            stream=False,
        )
    )


@pytest.fixture
def retrieve(monkeypatch: pytest.MonkeyPatch) -> Mock:
    result = Mock(return_value=([RetrievedChunk(1, "Passage", "Paper")], [42]))
    monkeypatch.setattr(LitkitProvider, "_retrieve_chunks", result)
    return result


@pytest.fixture
def prompts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    prompts: list[str] = []

    async def chat(
        _client: OpenAIClient,
        messages: list[Message],
        stream: bool = True,
    ) -> AssistantMessage | AsyncIterator[str]:
        prompts.append(messages[-1].content)
        content = f"Reply {len(prompts)}"

        async def tokens() -> AsyncIterator[str]:
            yield content

        return tokens() if stream else AssistantMessage(content)

    monkeypatch.setattr(OpenAIClient, "chat", chat)
    return prompts


async def submit(app: ChatApp, question: str) -> None:
    app.query_one(ChatInput).text = question
    app.action_submit()
    await app.workers.wait_for_complete()


@pytest.mark.parametrize("startup", [True, False], ids=["startup", "browser"])
async def test_loaded_session_retrieves_from_its_own_history(
    settings: ConfigWithSources,
    retrieve: Mock,
    prompts: list[str],
    startup: bool,
) -> None:
    messages = [
        Message("system", "Example format [99.1]"),
        Message("user", "Question"),
        Message("assistant", "Sources [7.2] and [3.1]."),
    ]
    path = settings.config.get_session_path() / "session-archive.json"
    archive = Session(SessionMetadata.create("Archive", "model", 3), "", messages).to_dict()
    archive["metadata"]["id"] = "archive"
    archive["messages"] = [{"role": m.role, "content": m.content} for m in messages]
    path.write_text(json.dumps(archive))
    app = ChatApp(config_with_sources=settings, session_file=path if startup else None)
    async with app.run_test() as pilot:
        if not startup:
            await submit(app, "From the old session")
            await pilot.press("ctrl+l", "enter")
            await pilot.click("#load-btn")
        await submit(app, "What about that paper?")
        assert retrieve.call_args.args[1] is None
        assert "[8.1]" in prompts[-1]
        # The generated answer has no inline citation. Its sources still reserve turn 8.
        app.action_save()
        app.action_new_session()
        app._handle_session_load(path)
        await submit(app, "Another follow-up")
        assert "[9.1]" in prompts[-1]


@pytest.mark.parametrize("stream", [True, False], ids=["streaming", "nonstreaming"])
@pytest.mark.parametrize("sourced", [True, False], ids=["cited", "uncited"])
async def test_regenerate_replaces_only_the_selected_turn(
    settings: ConfigWithSources,
    retrieve: Mock,
    prompts: list[str],
    stream: bool,
    sourced: bool,
) -> None:
    settings.config.stream = stream
    found = retrieve.return_value
    retrieve.side_effect = [found, found if sourced else ([], []), found]
    app = ChatApp(config_with_sources=settings)
    async with app.run_test() as pilot:
        await submit(app, "First question")
        log = app.query_one(ChatLog)
        first_sources = log.query_one(CitationsWidget)
        await submit(app, "Second question")
        app._handle_first_save("Two turns")
        await pilot.press("ctrl+r")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert "[2.1]" in prompts[-1]
        assert first_sources in log.query(CitationsWidget)
        assert len(log.query(CitationsWidget)) == 2
        assert [
            w.message_content for w in log.query(".assistant-message").results(MessageWidget)
        ] == [
            "Reply 1",
            "Reply 3",
        ]
        assert any("Session saved:" in w.message_content for w in log.query(MessageWidget))
        assert app.conversation is not None
        assert [m.content for m in app.conversation.messages if m.role == "user"] == [
            "First question",
            "Second question",
        ]


@pytest.mark.parametrize("stream", [True, False], ids=["partial-stream", "nonstreaming"])
async def test_failed_regeneration_leaves_original_reply_available(
    settings: ConfigWithSources,
    retrieve: Mock,
    prompts: list[str],
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    settings.config.stream = stream
    app = ChatApp(config_with_sources=settings)
    async with app.run_test() as pilot:
        await submit(app, "Question")
        assert app.conversation is not None
        original = list(app.conversation.messages)
        log = app.query_one(ChatLog)
        sources = log.query_one(CitationsWidget)

        async def partial_stream() -> AsyncIterator[str]:
            yield "Partial replacement"
            raise ChattyClientError("test failure")

        async def failed(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
            if stream:
                return partial_stream()
            raise ChattyClientError("test failure")

        with monkeypatch.context() as failure:
            failure.setattr(OpenAIClient, "chat", failed)
            await pilot.press("ctrl+r")
            await app.workers.wait_for_complete()
        assert app.conversation.messages == original
        assert log.query_one(".assistant-message", MessageWidget).message_content == "Reply 1"
        assert sources in log.query(CitationsWidget)
        await pilot.press("ctrl+r")
        await app.workers.wait_for_complete()
        assert retrieve.call_count == 3
        assert "[1.1]" in prompts[-1]


async def test_switch_while_answer_is_pending_preserves_loaded_session(
    settings: ConfigWithSources,
    retrieve: Mock,
    prompts: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [Message("user", "Saved question"), Message("assistant", "Saved reply")]
    path = save_session(
        Session(SessionMetadata.create("Archive", "model", 2), "", messages),
        settings.config.get_session_path(),
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed(*_args: object, **_kwargs: object) -> AssistantMessage:
        started.set()
        await release.wait()
        return AssistantMessage("Obsolete reply")

    app = ChatApp(config_with_sources=settings)
    async with app.run_test() as pilot:
        with monkeypatch.context() as ongoing:
            ongoing.setattr(OpenAIClient, "chat", delayed)
            app.query_one(ChatInput).text = "Question in the old session"
            app.action_submit()
            await asyncio.wait_for(started.wait(), timeout=5)
            app._handle_session_load(path)
            release.set()
            await pilot.pause()
        assert app.conversation is not None
        assert app.conversation.messages == messages
        await submit(app, "What about that?")
        assert retrieve.call_args.args[1] is None
        assert "[1.1]" in prompts[-1]
