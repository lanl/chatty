# Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (typer)                             │
│                    chatty chat | doctor | print-config          │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Textual App (ui/app.py)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Chat Log    │  │ Input       │  │ Status Bar              │  │
│  │ (scrollable)│  │ (multi-line)│  │ (model, stream, status) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Layer                                   │
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │ Conversation      │  │ RAGProvider       │                   │
│  │ (message history) │  │ (NullProvider v0.1│                   │
│  │                   │  │  LitkitProvider   │                   │
│  │                   │  │  later)           │                   │
│  └───────────────────┘  └───────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Client Layer                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ OpenAIClient (httpx + httpx-sse)                          │  │
│  │ - chat(messages, stream) → response | async iterator      │  │
│  │ - models() → list                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │ OpenAI-compatible   │
                    │ Endpoint            │
                    │ (institutional API) │
                    └─────────────────────┘
```

## Module Breakdown

### `chatty/config.py`

Pydantic Settings class handling configuration from multiple sources.

**Responsibilities:**
- Load from environment variables, CLI flags, and `~/.config/chatty/config.toml`
- Validate all configuration values
- Redact secrets for logging/display
- Provide typed access to all settings

**Key Design Decisions:**
- Precedence: CLI > env vars > config file > defaults
- Uses `pydantic-settings` for environment variable binding
- TOML format chosen for consistency with Python ecosystem (`pyproject.toml`)
- `api_key` never appears in `__repr__` or logs

### `chatty/client/openai_client.py`

Async HTTP client for OpenAI-compatible APIs.

**Responsibilities:**
- POST to `/chat/completions` with proper headers
- Handle streaming via SSE (Server-Sent Events)
- Implement retry logic for transient failures
- Surface errors with actionable context

**Key Design Decisions:**
- Uses `httpx.AsyncClient` for async HTTP
- Uses `httpx-sse` for SSE parsing (cleaner than mixing sync `sseclient-py`)
- No Textual imports—pure async Python, testable in isolation
- Exponential backoff on 429/5xx (max 2 retries)
- Timeout configurable per-request

**Interface:**
```python
class OpenAIClient:
    async def chat(
        self,
        messages: list[Message],
        stream: bool = True
    ) -> AssistantMessage | AsyncIterator[str]:
        ...

    async def models(self) -> list[str]:
        ...
```

### `chatty/core/conversation.py`

In-memory conversation state management.

**Responsibilities:**
- Store messages in OpenAI schema format (`role`, `content`)
- Truncate history when approaching context limits
- Provide iteration/serialization for API calls and logging

**Key Design Decisions:**
- v0.1: Simple message count limit (configurable, e.g., last 50 messages)
- v0.2+: Token-aware truncation with `tiktoken`
- System prompt is prepended but not stored in editable history
- Immutable message objects (new list on mutation)

### `chatty/core/rendering.py`

Output formatting and display utilities.

**Responsibilities:**
- Format assistant messages for display
- v0.2+: Render citations and source metadata

**Key Design Decisions:**
- Separating rendering from conversation keeps concerns clean
- Future citation rendering won't require changes to conversation logic

### `chatty/rag/provider.py`

Protocol defining the RAG augmentation interface.

**Responsibilities:**
- Define `RAGProvider` protocol
- Implement `NullProvider` (passthrough, v0.1 default)
- v0.3+: Implement `LitkitProvider`

**Interface:**
```python
class RAGProvider(Protocol):
    async def augment(
        self,
        conversation: Conversation,
        user_text: str
    ) -> tuple[list[Message], RAGMetadata]:
        """
        Given conversation history and new user input,
        return augmented messages and retrieval metadata.
        """
        ...

class NullProvider:
    """Passthrough provider—no augmentation."""
    async def augment(self, conversation, user_text):
        messages = conversation.messages + [{"role": "user", "content": user_text}]
        return messages, RAGMetadata(sources=[])
```

### `chatty/ui/app.py`

Textual application and widgets.

**Responsibilities:**
- Full-screen TUI with chat log, input, and status bar
- Handle keyboard shortcuts
- Manage streaming display and cancellation
- Surface errors inline (not modal dialogs)

**Key Design Decisions:**
- Single-screen layout optimized for terminal use
- Streaming tokens append to current message widget
- `Esc` cancels in-flight generation by cancelling the async task
- Status bar shows: model name, streaming on/off, connection status
- Error messages appear as styled message cards in chat log

**Keyboard Bindings:**
| Key | Action |
|-----|--------|
| `Ctrl+C` | Quit (clean shutdown) |
| `Ctrl+T` | Toggle streaming mode |
| `Ctrl+R` | Regenerate last response |
| `Esc` | Cancel current generation |

### `chatty/cli.py`

Typer-based CLI entrypoint.

**Commands:**

| Command | Description |
|---------|-------------|
| `chatty chat` | Launch the TUI (default) |
| `chatty doctor` | Run connectivity diagnostics |
| `chatty print-config` | Show resolved config (redacted) |

**`doctor` Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | OK |
| 1 | Missing required config (base_url, api_key) |
| 2 | Authentication failure |
| 3 | Network/TLS failure |
| 4 | API incompatibility |

## Data Flow

### Standard Chat Flow

```
1. User types message in Input widget
2. UI calls RAGProvider.augment(conversation, user_text)
   - v0.1: NullProvider returns messages unchanged
3. UI calls OpenAIClient.chat(messages, stream=True)
4. Client yields tokens via SSE
5. UI appends tokens to ChatLog
6. On completion, Conversation stores full assistant message
7. Optional: Write to transcript JSONL
```

### Streaming Cancellation

```
1. User presses Esc
2. UI cancels the asyncio Task running the stream
3. Partial message is discarded OR saved (configurable)
4. UI returns to ready state
```

## Extension Points

### Adding a New RAG Provider

1. Create `chatty/rag/litkit_provider.py`
2. Implement `RAGProvider` protocol
3. Register in config: `rag_provider = "litkit"`
4. Provider factory in `chatty/rag/__init__.py` resolves at startup

### Adding New Commands

1. Add function in `chatty/cli.py` with `@app.command()` decorator
2. Use existing config/client modules

### Custom Rendering

1. Extend `chatty/core/rendering.py`
2. UI calls render functions, which can be swapped for testing

## Testing Strategy

| Layer | Test Approach |
|-------|---------------|
| Config | Unit tests with env var fixtures |
| Client | `respx` to mock httpx, `pytest-asyncio` for async |
| Conversation | Pure unit tests, no I/O |
| UI | `pytest-textual-snapshot` for regression testing |
| Integration | Real endpoint in CI (optional, requires secrets) |

## Dependencies

### Runtime
- `textual>=0.40` — TUI framework
- `httpx>=0.25` — Async HTTP client
- `httpx-sse>=0.4` — SSE parsing for httpx
- `pydantic>=2.0` — Config validation
- `pydantic-settings>=2.0` — Environment variable binding
- `typer>=0.9` — CLI framework
- `rich>=13.0` — Terminal formatting (Textual dependency)

### Development
- `pytest>=7.0`
- `pytest-asyncio>=0.21`
- `respx>=0.20` — httpx mocking
- `ruff>=0.1` — Linting and formatting
- `mypy>=1.0` — Type checking
- `pre-commit>=3.0`
