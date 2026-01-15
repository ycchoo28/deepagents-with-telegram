# DeepAgents CLI Architecture Analysis

> **Purpose**: Technical analysis for creating alternative interfaces (e.g., Telegram) based on the existing CLI architecture.

---

## 1. Overview Architecture

The `deepagents-cli` is a Textual-based terminal UI application that provides an interactive chat interface to the core `deepagents` library. It follows a clean layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    main.py / cli_main()                         │
│                 Entry Point & CLI Arguments                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    app.py / DeepAgentsApp                       │
│              Textual TUI Application Controller                 │
│         ┌─────────────────┬─────────────────────┐              │
│         │  Widget System  │  Event Handling     │              │
│         │  (widgets/*.py) │  (Bindings, Actions)│              │
│         └─────────────────┴─────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              textual_adapter.py / TextualUIAdapter              │
│        Bridges Agent Execution ↔ Textual UI Rendering           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    agent.py / create_cli_agent()                │
│              Agent Factory with Middleware Stack                │
│    ┌──────────────┬──────────────┬──────────────┐              │
│    │MemoryMiddle  │SkillsMiddle  │ LocalContext │              │
│    │    ware      │    ware      │  Middleware  │              │
│    └──────────────┴──────────────┴──────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Core deepagents Library                        │
│     create_deep_agent() / FilesystemBackend / Middleware        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Components and File Paths

### 2.1 Entry Points

**File**: `libs/deepagents-cli/deepagents_cli/main.py`

- **Function**: `cli_main()` - Main entry point registered in `pyproject.toml`
- **Responsibilities**:
  - Parse CLI arguments (argparse)
  - Handle subcommands: `list`, `reset`, `skills`, `threads`, `help`
  - Thread resume logic (`-r` flag)
  - Create model and checkpointer
  - Launch Textual UI via `run_textual_cli_async()`

**Key Pattern**: Command pattern with subparsers for extensibility

```python
# Line 78-166: Argument parsing with subparsers
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(...)
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    # Each command has its own parser
    subparsers.add_parser("list", help="List all available agents")
    # Skills command delegated to skills module
    setup_skills_parser(subparsers)
```

### 2.2 Textual Application

**File**: `libs/deepagents-cli/deepagents_cli/app.py`

- **Class**: `DeepAgentsApp(App)` - Main Textual application
- **Key Features**:
  - Widget composition via `compose()` method
  - Key bindings for navigation (lines 94-115)
  - Session state management (`TextualSessionState`)
  - Agent task execution in background workers
  - Approval flow integration

**Reusable Patterns**:

```python
# Lines 160-174: Widget composition pattern
def compose(self) -> ComposeResult:
    with VerticalScroll(id="chat"):
        yield WelcomeBanner(id="welcome-banner")
        yield Container(id="messages")
    with Container(id="bottom-app-container"):
        yield ChatInput(cwd=self._cwd, id="input-area")
    yield StatusBar(cwd=self._cwd, id="status-bar")
```

### 2.3 UI Adapter Layer (Critical for Alternative Interfaces)

**File**: `libs/deepagents-cli/deepagents_cli/textual_adapter.py`

This is the **most reusable abstraction** for creating alternative interfaces:

- **Class**: `TextualUIAdapter` - Decouples agent execution from UI rendering
- **Callbacks Injected**:
  - `mount_message` - Display a message widget
  - `update_status` - Update status bar
  - `request_approval` - Handle HITL approval flow
  - `on_auto_approve_enabled` - Callback when auto-approve activated
  - `scroll_to_bottom` - UI scroll callback
- **Function**: `execute_task_textual()` (lines 80-595) - Core execution loop

**Key Pattern for Alternative Adapters**:

```python
class TextualUIAdapter:
    def __init__(
        self,
        mount_message: Callable,        # For Telegram: send_message()
        update_status: Callable[[str], None],  # For Telegram: edit_message()
        request_approval: Callable,     # For Telegram: inline keyboard
        on_auto_approve_enabled: Callable[[], None] | None = None,
        scroll_to_bottom: Callable[[], None] | None = None,  # N/A for Telegram
    ) -> None:
```

### 2.4 Agent Factory

**File**: `libs/deepagents-cli/deepagents_cli/agent.py`

- **Function**: `create_cli_agent()` (lines 323-469) - Creates fully configured agent
- **Returns**: `tuple[Pregel, CompositeBackend]`

**Middleware Stack Built Here**:

```python
# Lines 383-432: Middleware stack construction
agent_middleware = []

# Memory middleware - loads AGENTS.md files
if enable_memory:
    agent_middleware.append(
        MemoryMiddleware(
            backend=FilesystemBackend(),
            sources=memory_sources,
        )
    )

# Skills middleware - loads SKILL.md files
if enable_skills:
    agent_middleware.append(
        SkillsMiddleware(
            backend=FilesystemBackend(),
            sources=sources,
        )
    )

# Local context middleware - git info, directory tree
agent_middleware.append(LocalContextMiddleware())

# Shell middleware - local shell access
if enable_shell:
    agent_middleware.append(
        ShellMiddleware(
            workspace_root=str(Path.cwd()),
            env=shell_env,
        )
    )
```

**HITL Configuration (Human-in-the-Loop)**:

```python
# Lines 276-320: Interrupt configuration per tool
def _add_interrupt_on() -> dict[str, InterruptOnConfig]:
    shell_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_shell_description,
    }
    # ... per-tool configurations
    return {
        "shell": shell_interrupt_config,
        "execute": execute_interrupt_config,
        "write_file": write_file_interrupt_config,
        # ...
    }
```

---

## 3. Skills System

**Files**:
- `libs/deepagents-cli/deepagents_cli/skills/__init__.py`
- `libs/deepagents-cli/deepagents_cli/skills/load.py`
- `libs/deepagents-cli/deepagents_cli/skills/commands.py`

### Key Interfaces

```python
# skills/load.py - Skill metadata structure
class ExtendedSkillMetadata(SkillMetadata):
    source: str  # "user" or "project"

def list_skills(
    *, user_skills_dir: Path | None = None, 
    project_skills_dir: Path | None = None
) -> list[ExtendedSkillMetadata]:
```

### Skill Loading Pattern

1. Skills stored in `~/.deepagents/{agent_name}/skills/` (user) or `.deepagents/skills/` (project)
2. Each skill is a directory with `SKILL.md` containing YAML frontmatter
3. `SkillsMiddleware` from core library scans and injects into prompt
4. Progressive disclosure - full content loaded on-demand via `read_file`

---

## 4. Memory System

**Files**:
- `libs/deepagents-cli/deepagents_cli/config.py` (Settings class)
- Core library: `deepagents.middleware.MemoryMiddleware`

### Memory Sources

```python
# agent.py lines 386-396
memory_sources = [str(settings.get_user_agent_md_path(assistant_id))]
project_agent_md = settings.get_project_agent_md_path()
if project_agent_md:
    memory_sources.append(str(project_agent_md))
```

### Paths

- **User**: `~/.deepagents/{agent_name}/AGENTS.md`
- **Project**: `{project_root}/.deepagents/AGENTS.md`

---

## 5. Integration/Sandbox System (Plugin Pattern)

**File**: `libs/deepagents-cli/deepagents_cli/integrations/sandbox_factory.py`

### Factory Pattern

```python
# Lines 276-281: Provider registry
_SANDBOX_PROVIDERS = {
    "modal": create_modal_sandbox,
    "runloop": create_runloop_sandbox,
    "daytona": create_daytona_sandbox,
}

# Lines 284-314: Unified interface
@contextmanager
def create_sandbox(
    provider: str,
    *,
    sandbox_id: str | None = None,
    setup_script_path: str | None = None,
) -> Generator[SandboxBackendProtocol, None, None]:
    sandbox_provider = _SANDBOX_PROVIDERS[provider]
    with sandbox_provider(...) as backend:
        yield backend
```

### Backend Protocol (from core library)

```python
# deepagents/backends/protocol.py defines:
class SandboxBackendProtocol(Protocol):
    def execute(self, command: str) -> ExecuteResponse: ...
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]: ...
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]: ...
```

### Example Implementation

**File**: `libs/deepagents-cli/deepagents_cli/integrations/modal.py`

```python
class ModalBackend(BaseSandbox):
    def execute(self, command: str) -> ExecuteResponse:
        process = self._sandbox.exec("bash", "-c", command, timeout=self._timeout)
        process.wait()
        # ...
        return ExecuteResponse(output=output, exit_code=process.returncode, truncated=False)
```

---

## 6. Widget System

**File**: `libs/deepagents-cli/deepagents_cli/widgets/__init__.py`

### Widget Classes

| Widget | File | Purpose |
|--------|------|---------|
| `UserMessage` | `messages.py` | Display user input |
| `AssistantMessage` | `messages.py` | Streaming markdown output |
| `ToolCallMessage` | `messages.py` | Tool invocation display with collapsible output |
| `DiffMessage` | `messages.py` | File diff display |
| `ErrorMessage` | `messages.py` | Error display |
| `SystemMessage` | `messages.py` | System notifications |
| `ApprovalMenu` | `approval.py` | HITL approval UI |
| `ChatInput` | `chat_input.py` | Input with autocomplete |
| `StatusBar` | `status.py` | Bottom status bar |
| `WelcomeBanner` | `welcome.py` | Welcome screen |

### Streaming Pattern

```python
# messages.py lines 116-141
class AssistantMessage(Vertical):
    async def append_content(self, text: str) -> None:
        """Uses MarkdownStream for smoother rendering."""
        stream = self._ensure_stream()
        await stream.write(text)
```

---

## 7. Session/Thread Persistence

**File**: `libs/deepagents-cli/deepagents_cli/sessions.py`

### Key Functions

```python
def get_db_path() -> Path:  # ~/.deepagents/sessions.db
def generate_thread_id() -> str:  # 8-char hex
async def list_threads(agent_name, limit) -> list[dict]
async def get_most_recent(agent_name) -> str | None
async def thread_exists(thread_id) -> bool
async def delete_thread(thread_id) -> bool

@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """LangGraph checkpoint persistence via SQLite."""
```

---

## 8. Key Patterns for Alternative Interfaces (e.g., Telegram)

### 8.1 Adapter Pattern

Create a `TelegramUIAdapter` similar to `TextualUIAdapter`:

```python
class TelegramUIAdapter:
    def __init__(self, bot, chat_id):
        self.bot = bot
        self.chat_id = chat_id

    async def mount_message(self, content: str):
        await self.bot.send_message(self.chat_id, content)

    async def request_approval(self, action_request, assistant_id) -> Future:
        # Use inline keyboards for approval
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Approve", callback_data="approve"),
             InlineKeyboardButton("Reject", callback_data="reject")]
        ])
        await self.bot.send_message(
            self.chat_id, 
            f"Approve {action_request['name']}?", 
            reply_markup=keyboard
        )
        # Return future that resolves when callback received
```

### 8.2 Reusable Components

| Component | Path | Reusable For Telegram |
|-----------|------|----------------------|
| `create_cli_agent()` | `agent.py` | **YES** - Core agent factory |
| `execute_task_textual()` | `textual_adapter.py` | Adapt callbacks |
| `Settings` | `config.py` | **YES** - Configuration |
| `MemoryMiddleware` | Core library | **YES** |
| `SkillsMiddleware` | Core library | **YES** |
| `FileOpTracker` | `file_ops.py` | **YES** - Track operations |
| Sandbox backends | `integrations/` | **YES** - If needed |
| Session management | `sessions.py` | **YES** - Thread persistence |

### 8.3 Core Interface

```python
# The key function signature to adapt:
async def execute_task_textual(
    user_input: str,
    agent: Any,  # LangGraph Pregel
    assistant_id: str | None,
    session_state: Any,  # Has auto_approve, thread_id
    adapter: TextualUIAdapter,  # Replace with TelegramAdapter
    backend: Any = None,
    image_tracker: ImageTracker | None = None,
) -> None:
```

---

## 9. Dependency Graph

```
deepagents-cli
├── deepagents (core library)
│   ├── create_deep_agent()
│   ├── FilesystemMiddleware
│   ├── MemoryMiddleware
│   ├── SkillsMiddleware
│   ├── SubAgentMiddleware
│   └── Backends (Filesystem, Composite, State, Store)
├── langchain/langchain-core
├── langgraph (Pregel, checkpointing)
├── textual (TUI framework)
├── rich (terminal formatting)
├── prompt-toolkit (input handling)
├── aiosqlite (session persistence)
└── Provider SDKs (modal, runloop, daytona)
```

---

## 10. Creating a Telegram Interface

### Step 1: Create TelegramAdapter

Implement the same callback interface as `TextualUIAdapter`:

```python
class TelegramAdapter:
    def __init__(self, bot: Bot, chat_id: int):
        self.bot = bot
        self.chat_id = chat_id
        self._pending_approval: asyncio.Future | None = None
    
    async def mount_message(self, content: str, msg_type: str):
        """Send message to Telegram (split if > 4096 chars)"""
        for chunk in split_message(content, 4096):
            await self.bot.send_message(self.chat_id, chunk, parse_mode="Markdown")
    
    async def request_approval(self, action_request, assistant_id) -> dict:
        """Show inline keyboard for HITL approval"""
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("Approve", callback_data="approve"),
            InlineKeyboardButton("Reject", callback_data="reject"),
            InlineKeyboardButton("Edit", callback_data="edit"),
        ]])
        await self.bot.send_message(
            self.chat_id,
            f"**Approve action?**\n`{action_request['name']}`",
            reply_markup=keyboard
        )
        # Wait for callback
        self._pending_approval = asyncio.get_event_loop().create_future()
        return await self._pending_approval
    
    def resolve_approval(self, decision: str):
        """Called from callback handler"""
        if self._pending_approval:
            self._pending_approval.set_result({"decision": decision})
```

### Step 2: Reuse Core Components

- `create_cli_agent()` from `agent.py`
- `Settings` from `config.py`
- `sessions.py` for thread persistence
- Core `deepagents` middleware stack

### Step 3: Adapt Execution Function

Create `execute_task_telegram()`:
- Replace widget mounting with `bot.send_message()`
- Use inline keyboards for HITL approval
- Handle streaming differently (edit messages or send chunks)

### Step 4: Handle Telegram-Specific Concerns

- **Message length limits**: Split long responses (4096 char limit)
- **Inline keyboards**: For approvals and navigation
- **Webhook or polling**: For receiving updates
- **File uploads/downloads**: Via Telegram API
- **Multi-user isolation**: User/chat ID based sessions
- **Markdown conversion**: Telegram uses different markdown syntax

---

## 11. Proposed Telegram Library Structure

```
libs/deepagents-telegram/
├── deepagents_telegram/
│   ├── __init__.py
│   ├── main.py                 # Entry point (bot setup)
│   ├── bot.py                  # Telegram bot controller
│   ├── telegram_adapter.py     # Adapts agent events → Telegram messages
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── message.py          # Handle incoming messages
│   │   ├── callback.py         # Handle inline keyboard callbacks (HITL)
│   │   └── commands.py         # /start, /skills, /threads, etc.
│   ├── formatters/
│   │   ├── __init__.py
│   │   └── markdown.py         # Convert rich output → Telegram markdown
│   └── utils/
│       ├── __init__.py
│       └── message_splitter.py # Handle 4096 char limit
└── pyproject.toml
```

---

## 12. Summary

The DeepAgents CLI architecture is well-designed for creating alternative interfaces:

1. **Clear separation of concerns** between UI layer and agent logic
2. **Adapter pattern** (`TextualUIAdapter`) decouples execution from rendering
3. **Middleware stack** is fully reusable across interfaces
4. **Session persistence** is UI-agnostic
5. **Skills and memory systems** work independently of the interface

To create a Telegram interface, you primarily need to:
1. Implement `TelegramAdapter` with the same callback interface
2. Handle Telegram-specific concerns (message limits, keyboards, etc.)
3. Reuse all core components (agent factory, middleware, sessions)
