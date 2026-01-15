# DeepAgents Telegram Bot

A Telegram bot interface for DeepAgents, providing the same capabilities as the CLI including skills, memory, and human-in-the-loop (HITL) approval.

## Features

- Chat with DeepAgents AI assistant via Telegram
- Skills system (same as CLI)
- Memory/context awareness
- Human-in-the-loop approval for tool executions
- Thread/conversation management
- Shell command execution

## Installation

```bash
# From the libs directory
cd libs/deepagents-telegram
pip install -e .

# Or install with dependencies
pip install -e ".[dev]"
```

## Configuration

### Required Environment Variables

```bash
# Telegram bot token (get from @BotFather)
export TELEGRAM_BOT_TOKEN="your_bot_token_here"

# LLM API key (at least one required)
export ANTHROPIC_API_KEY="your_key"
# or
export OPENAI_API_KEY="your_key"
# or
export GOOGLE_API_KEY="your_key"
```

### Getting a Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Follow the prompts to create your bot
4. Copy the token provided

## Usage

### Start the Bot

```bash
# Using the entry point
deepagents-telegram

# Or run directly
python -m deepagents_telegram.main
```

### Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Show welcome message |
| `/help` | Show help information |
| `/new` | Start a new conversation thread |
| `/threads` | List conversation threads |
| `/skills` | List available skills |
| `/status` | Show current session status |
| `/autoapprove` | Toggle auto-approve mode |

### Chatting

Simply send any text message to the bot to start a conversation. The AI will respond and may request approval for certain actions.

### Approval Flow

When the agent wants to execute a tool that requires approval, you'll see an inline keyboard with options:
- **Approve** - Allow this specific action
- **Reject** - Deny this action
- **Edit command** - Modify the command before executing (for shell/execute tools)
- **Auto-approve all** - Allow all future actions in this session

#### Tool-Specific Approval Messages

The approval prompt shows tool-specific information:

| Tool | Information Shown |
|------|-------------------|
| `shell` / `execute` | The command to be executed |
| `write_file` | File path and content length |
| `edit_file` | File path being edited |
| `web_search` | Search query |
| `fetch_url` | URL to fetch |
| `task` | Task description |

#### Edit Flow

For shell commands, you can click "Edit command" to modify it before execution:
1. Click "Edit command"
2. Bot prompts you for the new command
3. Reply with the edited command (or `/cancel` to reject)
4. The modified command is executed

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Telegram Bot (python-telegram-bot)         │
│   /start, /skills, /threads, message handlers               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  TelegramAdapter                             │
│   mount_message → send_message()                            │
│   request_approval → inline keyboard                        │
│   update_status → edit_message()                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Reused from deepagents-cli                      │
│   create_cli_agent(), sessions.py, config.py                │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
deepagents_telegram/
├── __init__.py
├── main.py              # Entry point
├── adapter.py           # TelegramAdapter class
├── session.py           # Session state management
├── formatters.py        # Message formatting utilities
└── handlers/
    ├── __init__.py
    ├── commands.py      # Command handlers
    ├── messages.py      # Text message handler
    └── callbacks.py     # Inline keyboard callbacks
```

## Limitations

This is a demo implementation with some limitations:

- **In-memory sessions**: Sessions are lost when the bot restarts
- **No file uploads**: File upload/download not yet supported
- **No streaming**: Responses are sent as complete messages (no real-time streaming)
- **No sandbox**: Sandbox integrations (Modal, Runloop, Daytona) not available

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format .
ruff check --fix .
```

## License

Same license as the DeepAgents project.
