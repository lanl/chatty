# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.4] - 2026-01-13

### Added

**Search Conversation (`Ctrl+F`)**
- `Ctrl+F` opens inline search bar
- `Ctrl+F` again advances to next match (while search is open)
- Case-insensitive search through chat history
- Word-level highlighting for matching terms
- Match counter ("3/12 matches")
- `Esc` closes search and clears highlights
- Auto-scroll to matched message
- During search: assistant messages temporarily render as plain text (enables word highlighting)
- `ui/search.py` — new SearchBar widget with input and match counter

**MessageWidget Highlights**
- `contains_query()` method for searching
- `set_highlight()` / `clear_highlight()` for search highlighting
- `.search-match` and `.current-match` CSS classes
- `_apply_highlight()` adds `[reverse]` markup for word-level highlighting

**ChatLog Search**
- `search()` method finds matching messages
- `next_match()` / `prev_match()` for navigation
- `clear_search()` to remove highlights

### Changed

- Test count: 261 → 281 (+20)
- `^F Find` now visible in footer

---

## [0.2.3] - 2026-01-12

### Added

**Test Coverage & Validation**
- `pytest-textual-snapshot` for visual regression tests
- Widget unit tests (StatusBar, ChatInput, MessageWidget, ChatLog)
- Footer tests (click detection, width truncation, binding positions)
- Modal tests (FileInputModal, SessionBrowserModal, ModelPickerModal, ModelInputModal)
- Integration tests (app launch, actions, conversation flow)
- Session round-trip tests (save/load/export)
- Config validation tests (temperature, timeout bounds)
- app.py coverage tests (action handlers, callbacks, startup flows)

**Config Validation**
- Pydantic validators for `temperature` (0.0–2.0 range)
- Pydantic validators for `timeout_s` (positive integer)

### Changed

- Test count: 160 → 261 (+101)
- Coverage: 41% → 80% (+39%)
- app.py: 0% → 61%

---

## [0.2.2] - 2026-01-12

### Added

**Custom Footer Widget**
- New `ui/footer.py` with `ChattyFooter` widget
- Clickable keybindings with position tracking
- Width-aware truncation (fewer bindings on narrow terminals)
- Priority bindings preserved (Submit, Stop, Quit always visible)

### Changed

**Keyboard Bindings**
- `Ctrl+E` → `Ctrl+P` for Submit ("P for Prompt")
- `Ctrl+Y` → `Ctrl+C` for Copy
- `Ctrl+B` → `Ctrl+E` for Export
- Command palette (`Ctrl+P`) disabled to avoid conflict

### Removed

- Dependency on Textual's built-in Footer widget

---

## [0.2.1] - 2026-01-12

### Changed

**Code Refactoring**
- Split `ui/app.py` (~1600 lines) into smaller modules:
  - `ui/widgets.py` — ChatLog, MessageWidget, StatusBar, ChatInput (357 lines)
  - `ui/modals.py` — FileInputModal, SessionBrowserModal, ModelPickerModal, ModelInputModal (477 lines)
  - `ui/app.tcss` — Extracted stylesheet (65 lines)
  - `ui/app.py` — ChatApp class, bindings, main entry (918 lines)

**Code Cleanup**
- Removed duplicate defaults in `config.py` (now uses `_CONFIG_FIELDS` list)
- Removed unused `ConfigSource` dataclass
- Fixed tiktoken hardcode in `conversation.py` (now uses model attribute with cl100k_base fallback)

### Fixed

- Resolved `# type: ignore` comments (fixed with `thread=True` worker)

---

## [0.2.0] - 2026-01-09

### Added

**RAG Provider Integration**
- `rag_provider` config option (`"none"` default, `"litkit"` in v0.3)
- `CHATTY_RAG_PROVIDER` environment variable
- `get_provider()` factory function in `chatty.rag`
- `UnknownProviderError` for invalid provider names
- `RAGProvider.augment()` now called in chat flow (NullProvider passthrough)
- `last_rag_metadata` stored for future citation display

**Session Save/Load**
- `Ctrl+S` saves current conversation to JSON file
- `--session` CLI option loads a saved session on startup
- `session_path` config option (default: `~/.config/chatty/sessions`)
- `CHATTY_SESSION_PATH` environment variable
- Auto-generated session names from first user message
- Session format: JSON with version field for future compatibility

**Session History Browser**
- `Ctrl+L` opens modal to browse saved sessions
- OptionList displays session name, date, message count
- Enter or double-click to load, Escape to cancel
- Empty state shows helpful hint to save with Ctrl+S

**Copy to Clipboard**
- `Ctrl+Y` copies last assistant response to system clipboard
- Toast notification confirms copy action
- `copy_fallback_path` config for headless HPC (default: `./copies`)
- `CHATTY_COPY_FALLBACK_PATH` environment variable
- Falls back to timestamped file when clipboard unavailable

**Model Picker**
- `Ctrl+G` opens model picker modal
- Fetches available models from `/models` endpoint
- Falls back to manual entry when endpoint unavailable
- Current model marked with bullet (●) in list
- Updates status bar when model changes
- Runtime-only change (config file not modified)

**Export to Markdown**
- `Ctrl+B` exports conversation to readable Markdown file
- `export_path` config option (default: `./exports`)
- `CHATTY_EXPORT_PATH` environment variable
- Session name and model included in header
- System messages excluded from export

### Changed

- Chat flow now uses `RAGProvider.augment()` instead of building messages directly
- User messages added to conversation after successful LLM response (enables query rewriting in v0.3)
- `action_regenerate()` updated to work with deferred message flow
- `Ctrl+N` (new session) now resets session tracking

### Known Limitations

- **Reasoning/thinking models** — Models that expose chain-of-thought (DeepSeek R1, QwQ, Apriel Thinker) output their reasoning inline. Chatty does not filter this; use non-reasoning variants for cleaner output.

### Developer Notes

- v0.2 acceptance criterion: `RAGProvider.augment()` called on every user message
- v0.2 acceptance criterion: `NullProvider` produces identical behavior to v0.1
- Session module: `SessionMetadata`, `Session`, `save_session()`, `load_session()`, `list_sessions()`
- Model picker modals: `ModelPickerModal`, `ModelInputModal`

## [0.1.0] - 2026-01-09

Initial release of chatty — a terminal UI chatbot for OpenAI-compatible endpoints.

### Added

**Core Features**
- Full-screen TUI built with Textual
- OpenAI-compatible async client with SSE streaming
- Markdown rendering with syntax-highlighted code blocks
- Configuration via TOML file, environment variables, and CLI flags
- `chatty chat` — Launch interactive chat UI
- `chatty doctor` — Diagnose configuration and connectivity
- `chatty print-config` — Display resolved configuration (secrets redacted)

**HPC & Institutional Network Support**
- TLS/CA bundle configuration (`ca_bundle`, `verify_tls`)
- Secure API key storage via `api_key_file` (chmod 600)
- HTTP proxy support (`http_proxy`, `no_proxy`)
- Offline installation via wheelhouse for air-gapped systems
- Scripts: `build-wheelhouse.sh`, `install-offline.sh`

**Input & Interaction**
- Keyboard shortcuts (Ctrl+E submit, Ctrl+Q quit, Ctrl+N new session, etc.)
- Load queries from files (`Ctrl+O` or `--query-file`)
- Multi-line paste support
- Interrupt generation with `Esc`

**Context & Session Management**
- In-memory conversation history
- Context window tracking (server-reported or tiktoken estimate)
- New session (`Ctrl+N`) clears history

**Feedback & Polish**
- Color-coded messages (user/assistant/system)
- "Thinking..." spinner with elapsed time
- Response time shown after generation
- Inline error display with actionable hints
- Auto-retry with countdown on transient failures

**Transcript Logging**
- JSONL transcript files with timestamps
- Optional metadata: model, response_time_s, tokens
- Configurable via `transcript_enabled` and `transcript_path`

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+E` | Submit query |
| `Ctrl+Q` | Quit |
| `Ctrl+N` | New session |
| `Ctrl+O` | Load file |
| `Esc` | Interrupt generation |
| `Enter` | Insert newline |
| `Ctrl+R` | Regenerate last response |
| `Ctrl+T` | Toggle streaming mode |

### Configuration

Config file search order:
1. `CHATTY_CONFIG` environment variable
2. `./chatty.toml` (repo-local)
3. `~/.config/chatty/config.toml` (user default)

### Notes

- **UI test coverage deferred** — Requires `pytest-textual-snapshot` setup; will be addressed when UI stabilizes.
- **RAGProvider protocol exists** — `NullProvider` implemented as passthrough; wiring into chat flow deferred to v0.2.
- **tiktoken is optional** — Token counting degrades gracefully if unavailable.

[0.2.0]: https://github.com/your-org/chatty/releases/tag/v0.2.0
[0.1.0]: https://github.com/your-org/chatty/releases/tag/v0.1.0
