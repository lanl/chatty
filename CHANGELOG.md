# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.14] - 2026-01-19

### Added

**Test Coverage Improvements**
- Overall coverage improved from 80% to 88%
- `session_handlers.py` coverage: 46% → 94% (+48%)
- `workers.py` coverage: 52% → 100% (+48%)
- `cli.py` coverage: 64% → 82% (+18%)
- `modals.py` coverage: 66% → 85% (+19%)

### Tests Added

**New Test Files**
- `tests/test_session_handlers.py` — 27 tests for session management functions
- `tests/test_workers.py` — 22 tests for async worker functions

**Extended Test Files**
- `tests/test_cli.py` — CLI argument tests (--query-file, --session, --model, etc.)
- `tests/test_modals.py` — Session browser rename/delete action tests

### Developer Notes

- 410 total tests (was 351)
- 88% coverage target exceeded (goal was 83%)
- All acceptance criteria met
- Pre-commit hooks pass

---

## [0.2.13] - 2026-01-19

### Changed

**Workers Extraction**
- Extracted async worker logic to `ui/workers.py` (236 lines)
- `app.py` reduced from 932 to 760 lines (-172 lines)

### Extracted Functions

| Function | Description |
|----------|-------------|
| `fetch_context_window()` | Fetch context window from /models endpoint |
| `fetch_and_show_models()` | Fetch models and show picker modal |
| `send_message()` | Main LLM call with streaming support |

### Developer Notes

- Completes the original v0.2.11 extraction goals
- Uses `TYPE_CHECKING` guards to avoid circular imports
- Functions receive `ChatApp` instance to access state
- Coverage maintained at 80%
- All 351 tests pass

---

## [0.2.12] - 2026-01-19

### Changed

**Session Handlers Extraction**
- Extracted session management logic to `ui/session_handlers.py` (251 lines)
- `app.py` reduced from 1075 to 932 lines (-143 lines)

### Extracted Functions

| Function | Description |
|----------|-------------|
| `load_session_file()` | Load session from file on startup |
| `load_and_submit_query_file()` | Load query file on startup |
| `handle_session_load()` | Handle session selected from browser |
| `handle_first_save()` | Handle first-save modal result |
| `save_current_session()` | Save current session to disk |
| `handle_file_path()` | Handle file path modal result |

### Developer Notes

- Uses `TYPE_CHECKING` guards to avoid circular imports
- Functions receive `ChatApp` instance to access state
- Coverage maintained at 80%
- All 351 tests pass

---

## [0.2.11] - 2026-01-19

### Changed

**Partial Refactoring**
- Extracted clipboard logic to `ui/clipboard.py` (59 lines)
- `app.py` now delegates `_copy_to_clipboard()` to clipboard module
- `app.py` reduced from 1102 to 1075 lines (-27 lines)

### Developer Notes

- Clipboard module has `copy_to_clipboard()` and `write_copy_file()` functions
- Workers and session handlers remain in app.py (future extraction deferred)
- Coverage maintained at 80%

---

## [0.2.10] - 2026-01-19

### Added

**Test Coverage Improvements**
- Overall coverage improved from 78% to 80%
- `app.py` coverage: 62% → 64% (+2%)
- `modals.py` coverage: 63% → 66% (+3%)
- `widgets.py` coverage: 90% → 93% (+3%)
- `search.py` coverage: 74% → 77% (+3%)

### Tests Added

- Help modal integration tests (open/close)
- Search bar integration tests
- Submit whitespace handling tests
- Streaming toggle status bar tests
- First save flow tests (_handle_first_save)
- ModelPickerModal selection tests
- ModelInputModal enter key submission tests
- SearchBar.set_matches tests
- MessageWidget.count_occurrences tests
- ChatLog.total_occurrences tests

### Changed

- `pyproject.toml` and `__init__.py` version updated to 0.2.10

### Developer Notes

- 351 total tests (was 331)
- 80% coverage target achieved

---

## [0.2.9] - 2026-01-19

### Added

**Color Accessibility**
- System messages now have `ℹ` icon prefix (U+2139)
- System messages have left border for shape distinction
- Colorblind users can distinguish message types without relying on color
- Uses Unicode symbols (not emoji) for HPC terminal compatibility

### Changed

- System messages: `System` → `ℹ System`
- `pyproject.toml` and `__init__.py` version updated to 0.2.9

### Developer Notes

- Added 4 tests for ROLE_PREFIXES in `test_widgets.py`
- CSS: `.system-message` now has `border-left: thick $accent`

---

## [0.2.8] - 2026-01-19

### Added

**Session Deletion**
- `d` key in session browser deletes selected session
- `SessionDeleteConfirmModal` for delete confirmation
- Delete button in red `variant="error"` for visual warning
- "This cannot be undone" warning text
- Cancel button focused by default (safer UX)
- `delete_session()` function in `session.py`
- If deleting currently-loaded session, clears app state
- Hint text updated: `r rename d delete Enter load`
- 7 new tests for deletion functionality

### Changed

- `pyproject.toml` version updated to 0.2.8

---

## [0.2.7] - 2026-01-19

### Added

**Session Naming Improvements**
- First `Ctrl+S` prompts for session name with auto-generated default
- `r` key in session browser renames selected session
- `SessionRenameModal` for name input with cancel/rename buttons
- `rename_session()` function in `session.py`
- Hint text in session browser showing `r rename Enter load`
- 17 new tests for rename functionality

### Changed

- Session browser refreshes list after rename, preserving selection
- `action_save()` now uses `_handle_first_save()` and `_save_current_session()`

---

## [0.2.6] - 2026-01-19

### Added

**Help Widget**
- `F1` opens in-app help modal
- Displays all keyboard shortcuts with descriptions
- Shows current configuration (model, streaming, context window, transcript)
- Quick tips section for common workflows
- Version information with diagnostic hint
- `ui/help.py` — new HelpModal component (210 lines)
- `F1 Help` added to footer (visible before Quit)
- 21 new tests in `tests/test_help.py`

### Changed

- Footer now includes `F1 Help` binding before `^Q Quit`
- Updated `__version__` in `src/chatty/__init__.py` to "0.2.6"

---

## [0.2.5] - 2026-01-13

### Added

**Context Window Configuration**
- `context_window` config option: `"auto"` (default) or integer
- `CHATTY_CONTEXT_WINDOW` environment variable
- Auto-detect from `/models` endpoint when set to `"auto"`
- Clear error message if auto-detection fails
- `chatty.toml.example` — sanitized example config for version control

**New Methods**
- `OpenAIClient.get_model_context_length()` — fetch context_length from endpoint
- `Conversation.set_context_window()` — update context window at runtime

### Changed

- Context window now resolved on startup (async worker)
- Status bar displays actual context limit from endpoint or config
- Default context_window changed from hardcoded 128K to `"auto"`

---

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
