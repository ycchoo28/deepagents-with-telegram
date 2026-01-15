# DeepAgents Telegram Bot

A Telegram bot interface for DeepAgents, providing the same capabilities as the CLI including skills, memory, and human-in-the-loop (HITL) approval.

## Quick Start

```bash
# 1. Navigate to the telegram bot directory
cd agent/deepagents-with-telegram/libs/deepagents-telegram

# 2. Copy and configure environment file
cp .env.example .env
# Edit .env with your TELEGRAM_BOT_TOKEN and LLM API key

# 3. Install dependencies (from deepagents-with-telegram directory)
cd ../..
pip install -e libs/deepagents -e libs/deepagents-cli -e libs/deepagents-telegram

# 4. Run the bot
cd libs/deepagents-telegram
export $(cat .env | grep -v '^#' | xargs) && deepagents-telegram
```

**Minimum required `.env` configuration:**
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
ANTHROPIC_API_KEY=your_anthropic_key  # or OPENAI_API_KEY
```

## Features

- Chat with DeepAgents AI assistant via Telegram
- Skills system (same as CLI)
- Memory/context awareness
- Human-in-the-loop approval for tool executions
- Thread/conversation management
- Shell command execution
- **Custom OpenAI-compatible API support** (DeepSeek, Ollama, vLLM, LM Studio, etc.)

## Installation

```bash
# From the deepagents-with-telegram directory
cd agent/deepagents-with-telegram

# Create virtual environment (optional but recommended)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies in order
pip install -e libs/deepagents
pip install -e libs/deepagents-cli
pip install -e libs/deepagents-telegram

# Or install all at once
pip install -e libs/deepagents -e libs/deepagents-cli -e libs/deepagents-telegram
```

## Configuration

### Quick Setup

1. Copy the example environment file:
```bash
cd libs/deepagents-telegram
cp .env.example .env
```

2. Edit `.env` with your settings (see below)

3. Run the bot:
```bash
deepagents-telegram
```

### Environment Variables

#### Required

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token from @BotFather |
| `OPENAI_API_KEY` | API key for OpenAI or compatible API (at least one LLM key required) |

#### Optional - Custom OpenAI-Compatible API

| Variable | Description |
|----------|-------------|
| `OPENAI_API_BASE` | Custom API endpoint URL (e.g., `http://localhost:8000/v1`) |
| `OPENAI_PROVIDER` | Force provider detection: `openai`, `anthropic`, or `google` |
| `OPENAI_MODEL` | Custom model name (e.g., `deepseek-chat`, `llama3:70b`) |

#### Alternative LLM Providers

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude models) |
| `GOOGLE_API_KEY` | Google API key (for Gemini models) |

### Getting a Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Follow the prompts to create your bot
4. Copy the token provided

### Example Configurations

#### Standard OpenAI
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o
```

#### DeepSeek
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_PROVIDER=openai
OPENAI_MODEL=deepseek-chat
```

#### Ollama (Local)
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
OPENAI_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_PROVIDER=openai
OPENAI_MODEL=llama3:70b
```

#### vLLM / LM Studio / LocalAI
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
OPENAI_API_KEY=your_key_or_dummy
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_PROVIDER=openai
OPENAI_MODEL=your-model-name
```

#### Anthropic Claude
```bash
TELEGRAM_BOT_TOKEN=your_bot_token
ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
```

## Usage

### Start the Bot

```bash
# Navigate to the telegram package directory
cd agent/deepagents-with-telegram/libs/deepagents-telegram

# Load environment and run
export $(cat .env | grep -v '^#' | xargs) && deepagents-telegram

# Or run directly with python
export $(cat .env | grep -v '^#' | xargs) && python -m deepagents_telegram.main
```

### One-liner (from project root)

```bash
cd agent/deepagents-with-telegram/libs/deepagents-telegram && export $(cat .env | grep -v '^#' | xargs) && deepagents-telegram
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

## Troubleshooting

### 404 Error - Model Not Found

If you see `Error code: 404 - {'detail': 'Not Found'}`:

1. **Check your base URL** - Make sure it ends with `/v1`:
   ```bash
   # Wrong
   OPENAI_API_BASE=http://localhost:8000
   
   # Correct
   OPENAI_API_BASE=http://localhost:8000/v1
   ```

2. **Verify the model name** - Test with curl:
   ```bash
   curl http://localhost:8000/v1/models
   ```

3. **Set OPENAI_PROVIDER** - Required for non-standard model names:
   ```bash
   OPENAI_PROVIDER=openai
   ```

### Provider Detection Failed

If you see "Could not detect provider from model name":

Set `OPENAI_PROVIDER=openai` to force OpenAI-compatible mode for custom model names.

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
