# Architecture

## Scope

**chatty is the query interface** for interacting with a pre-built corpus. It enables operators to:
- Chat with an LLM (with or without RAG)
- Load and submit queries
- Inspect retrieved context
- Manage conversation sessions

**Explicitly out of scope:**
- Corpus building/indexing (handled by `litkit` CLI)
- PDF conversion to JATS XML (handled by `text-fetch` + GROBID)
- SLURM job orchestration for HPC builds
- Any modification to the vector store

chatty expects a pre-built index to exist. If the index is missing, chatty displays an actionable error with instructions for building it via litkit—but does not attempt to build it.

A separate build UI may be developed in a future project.

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
│  │ Conversation      │  │ QueryRewriter     │                   │
│  │ (message history) │  │ (v0.3+)           │                   │
│  └───────────────────┘  └───────────────────┘                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ RAGProvider (NullProvider v0.1 | LitkitProvider v0.3)     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│      Client Layer           │  │    litkit (optional, v0.3)  │
│  ┌───────────────────────┐  │  │  ┌───────────────────────┐  │
│  │ OpenAIClient          │  │  │  │ shortlist_papers()    │  │
│  │ (httpx + httpx-sse)   │  │  │  │ search_chunks_constr..│  │
│  │ - chat(messages)      │  │  │  │ get_chunks()          │  │
│  │ - models()            │  │  │  └───────────────────────┘  │
│  └───────────────────────┘  │  └─────────────────────────────┘
└─────────────────────────────┘
                    │
                    ▼
       ┌─────────────────────┐
       │ OpenAI-compatible   │
       │ Endpoint            │
       │ (institutional API) │
       └─────────────────────┘
```

## Package Management

**uv is the only supported package manager.** No pip, poetry, or pip-tools.

```bash
# Install dependencies
uv sync

# Add a dependency
uv add httpx

# Add a dev dependency
uv add --dev pytest

# Run a command in the venv
uv run chatty chat
```

This simplifies the toolchain and ensures reproducible builds across laptop and HPC.

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
- Provide iteration/serialization for API calls and logging
- Track token usage and context window utilization

**Key Design Decisions:**
- Token counting via `tiktoken` for accurate context window tracking
- Status bar displays context usage (e.g., "12K / 128K tokens")
- Warning when approaching 80% of context window
- System prompt is prepended but not stored in editable history
- Immutable message objects (new list on mutation)

**Context Management:**
- `Ctrl+K` triggers context compression (LLM summarizes conversation history)
- Manual "New Session" option to start fresh
- API response `usage` field provides actual token counts post-request

### `chatty/core/query_rewriter.py` (v0.3+)

LLM-based query rewriting for RAG.

**Problem:** litkit's retrieval is single-query—it has no conversation memory. Multi-turn chat like:
```
User: Tell me about HIV treatments
Assistant: [response about HIV treatments]
User: What about the side effects?
```
...fails because "What about the side effects?" doesn't mention HIV.

**Solution:** Before calling litkit, rewrite the user's message as a standalone query:
```
"What about the side effects?" → "What are the side effects of HIV treatments?"
```

**Interface:**
```python
class QueryRewriter:
    def __init__(self, client: OpenAIClient):
        self.client = client

    async def rewrite(
        self,
        conversation: Conversation,
        user_text: str
    ) -> str:
        """
        Given conversation history and new user input,
        return a standalone query suitable for RAG retrieval.
        """
        ...
```

**Implementation:**
```python
REWRITE_PROMPT = """Given the conversation history below, rewrite the user's latest message as a standalone question that captures all necessary context.

Conversation:
{history}

Latest message: {user_text}

Rewritten standalone question:"""
```

**Conditional Rewriting:**

Long queries (committee-written, 1-2 pages) are already self-contained. Short follow-ups need context.

```python
LONG_QUERY_THRESHOLD = 800  # ~200 tokens ≈ 800 chars

async def augment(self, conversation, user_text):
    if len(user_text) > LONG_QUERY_THRESHOLD:
        query = user_text  # Already standalone, skip rewriting
    else:
        query = await self.rewriter.rewrite(conversation, user_text)
    ...
```

**Key Design Decisions:**
- Uses the same LLM endpoint as chat (no separate config)
- Short prompt, short response (~50-100 tokens total)
- Non-streaming for simplicity (latency acceptable)
- Falls back to original query on error
- Long queries (> 800 chars) skip rewriting—assumed self-contained

### `chatty/core/rendering.py`

Output formatting and display utilities.

**Responsibilities:**
- Format assistant messages for display
- v0.3+: Render citations and source metadata from RAG

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

### `chatty/rag/litkit_provider.py` (v0.3+)

litkit integration for corpus-grounded responses.

**Responsibilities:**
- Rewrite multi-turn queries to standalone form
- Call litkit retrieval functions
- Inject retrieved context into messages
- Return source metadata for citation rendering
- Surface explicit errors (no silent fallbacks)

**Error Handling (Explicit, Actionable):**
```python
# Index not found
LitkitError: "FAISS index not found at /path/to/indices. Run `litkit --build-only`."

# Database not found  
LitkitError: "SQLite database not found at /path/to/db. Index may be corrupted."

# litkit not installed (when rag_provider="litkit")
ConfigError: "litkit package not installed. Run `uv add litkit` or set `rag_provider = none`."
```

All errors appear inline as styled message cards—never silent fallback to non-RAG mode.

**Inspect Mode (v0.3):**
Operators can preview retrieved context before LLM generation:
- `Ctrl+I` toggles inspect mode — queries run retrieval only, no LLM call
- Retrieved context display shows document title, relevance score, snippet
- From inspect view: `Enter` proceeds to LLM, `Esc` cancels to refine query
- `Ctrl+G` after normal response shows what context was injected

**Interface:**
```python
class LitkitProvider:
    def __init__(self, client: OpenAIClient, rewriter: QueryRewriter):
        self.client = client
        self.rewriter = rewriter

    async def augment(self, conversation, user_text):
        # 1. Rewrite query for RAG
        standalone_query = await self.rewriter.rewrite(conversation, user_text)
        
        # 2. Call litkit (sync functions, run in thread pool)
        docs = await asyncio.to_thread(self._retrieve, standalone_query)
        
        # 3. Build augmented messages with context
        augmented = self._inject_context(conversation, user_text, docs)
        
        return augmented, RAGMetadata(sources=docs)
    
    def _retrieve(self, query: str) -> list[Document]:
        """Sync wrapper around litkit retrieval."""
        from litkit.cli import shortlist_papers, search_chunks_constrained, get_chunks
        from litkit.db import connect_db
        
        papers = shortlist_papers(query, k=500)
        chunk_ids, _ = search_chunks_constrained(query, papers, k=30)
        conn = connect_db(DB_PATH)
        return get_chunks(conn, chunk_ids)
```

**Key Design Decisions:**
- litkit is an **optional dependency**—chatty works without it
- litkit functions are sync; we run them in `asyncio.to_thread()` to avoid blocking the UI
- Query rewriting happens in chatty (async, streaming-capable) not litkit
- LLM calls stay in chatty's client (streaming, better UX)
- litkit only provides retrieval: `shortlist_papers()`, `search_chunks_constrained()`, `get_chunks()`

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

**Input Methods:**
- **Type** — Direct text input for short queries and follow-ups
- **Paste** — Multi-line paste for medium-length queries
- **Load from file** — `Ctrl+O` opens file path prompt for long committee queries
- **CLI argument** — `--query-file` for scripted/reproducible sessions

**Visual Features:**
- **Markdown rendering** — Headers, lists, code blocks with syntax highlighting (via Rich)
- **Color-coded messages** — User (blue), Assistant (green), System (gray)
- **"Thinking..." spinner** — Animated with elapsed time while awaiting response
- **Response time** — "Generated in 3.2s" shown after each response
- **Timestamps** — Optional, configurable per message

**Keyboard Bindings (v0.1):**
| Key | Action |
|-----|--------|
| `Ctrl+C` | Quit (clean shutdown) |
| `Ctrl+T` | Toggle streaming mode |
| `Ctrl+R` | Regenerate last response |
| `Ctrl+O` | Load query from file |
| `Ctrl+K` | Compress context (summarize history) |
| `Ctrl+N` | New session (clear history) |
| `Ctrl+Y` | Copy last message to clipboard |
| `Esc` | Cancel current generation |

**Keyboard Bindings (v0.2):**
| Key | Action |
|-----|--------|
| `Ctrl+S` | Save session |
| `Ctrl+F` | Search conversation |
| `Ctrl+B` | Toggle bookmark on message |
| `Ctrl+Z` | Undo last exchange |

**Keyboard Bindings (v0.3):**
| Key | Action |
|-----|--------|
| `Ctrl+I` | Toggle inspect mode (retrieval only) |
| `Ctrl+G` | View retrieved context for last response |
| `Shift+Enter` | Submit in inspect mode (one-time) |

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

### Standard Chat Flow (v0.1)

```
1. User types message in Input widget
2. UI calls RAGProvider.augment(conversation, user_text)
   - NullProvider returns messages unchanged
3. UI calls OpenAIClient.chat(messages, stream=True)
4. Client yields tokens via SSE
5. UI appends tokens to ChatLog
6. On completion, Conversation stores full assistant message
7. Optional: Write to transcript JSONL
```

### RAG-Augmented Flow (v0.3)

```
1. User types message in Input widget
2. UI calls LitkitProvider.augment(conversation, user_text)
   a. QueryRewriter rewrites to standalone query
   b. litkit retrieves relevant documents
   c. Context injected into messages
3. UI calls OpenAIClient.chat(augmented_messages, stream=True)
4. Client yields tokens via SSE
5. UI appends tokens to ChatLog (with citation markers)
6. On completion:
   - Conversation stores full assistant message
   - Rendering displays citations from RAGMetadata
7. Optional: Write to transcript JSONL (includes sources)
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

1. Create `chatty/rag/my_provider.py`
2. Implement `RAGProvider` protocol
3. Register in config: `rag_provider = "my_provider"`
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
| QueryRewriter | Mock LLM responses, test prompt construction |
| LitkitProvider | Mock litkit functions, test integration flow |
| UI | `pytest-textual-snapshot` for regression testing |
| Integration | Real endpoint in CI (optional, requires secrets) |

## Deployment

### Charliecloud Container (HPC)

chatty + litkit are bundled in a single Charliecloud container for HPC deployment.

**Why containers?**
- **Reproducibility** — Same binary, same dependencies, same behavior
- **Security** — Containers run with limited privileges; admins can audit contents
- **Isolation** — No conflicts with system Python or other users
- **Distribution** — Single .sqfs file transfers via sneakernet to air-gapped systems

**Container structure:**
```
chatty-litkit.sqfs
├── /opt/venv/           # uv-managed virtualenv
│   ├── bin/chatty       # CLI entrypoint
│   └── lib/python3.11/  # Dependencies
├── /opt/litkit/         # litkit package + indices
│   ├── workspace/       # FAISS indices, SQLite DB
│   └── hf_home/         # Cached embedding models
└── /etc/chatty/         # Default config (overridable)
```

**Building the container:**
```bash
# Build from Dockerfile
docker build -t chatty-litkit:latest .

# Convert to Charliecloud squashfs
ch-convert -i docker chatty-litkit:latest chatty-litkit.sqfs
```

**Running on HPC:**
```bash
# Interactive
ch-run chatty-litkit.sqfs -- chatty chat

# With custom config via environment
ch-run -b /path/to/workspace:/opt/litkit/workspace \
       --set-env=OPENAI_BASE_URL=https://internal-llm/v1 \
       chatty-litkit.sqfs -- chatty chat
```

**Environment variables for HPC:**
| Variable | Purpose |
|----------|---------|
| `OPENAI_BASE_URL` | LLM endpoint URL |
| `OPENAI_API_KEY` | Authentication token |
| `LITKIT_WORKSPACE` | Path to indices/DB inside container |
| `HF_HOME` | Hugging Face cache (for embedding models) |

**Same image, different environments:**
- **Networked HPC:** `OPENAI_BASE_URL=https://institutional-llm.internal/v1`
- **Air-gapped HPC:** `OPENAI_BASE_URL=https://enclave-llm.local/v1`
- **Laptop:** `OPENAI_BASE_URL=http://localhost:1234/v1` (LM Studio)

## Dependencies

### Runtime
- `textual>=0.40` — TUI framework
- `httpx>=0.25` — Async HTTP client
- `httpx-sse>=0.4` — SSE parsing for httpx
- `pydantic>=2.0` — Config validation
- `pydantic-settings>=2.0` — Environment variable binding
- `typer>=0.9` — CLI framework
- `rich>=13.0` — Terminal formatting (Textual dependency)
- `tiktoken>=0.5` — Token counting for context window tracking

### Optional (v0.3+)
- `litkit` — RAG retrieval (imports conditionally)

### Development
- `pytest>=7.0`
- `pytest-asyncio>=0.21`
- `respx>=0.20` — httpx mocking
- `ruff>=0.1` — Linting and formatting
- `mypy>=1.0` — Type checking
- `pre-commit>=3.0`
