# DeepAgents Quickstart Guide

This guide covers how to start the CLI agent and Telegram bot, and how they share the same agent configuration.

## Prerequisites

```bash
# Navigate to the project
cd agent/deepagents-with-telegram

# Install all packages
pip install -e libs/deepagents -e libs/deepagents-cli -e libs/deepagents-telegram

# Set your LLM API key (at least one required)
export ANTHROPIC_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"
```

## Starting the CLI Agent

```bash
# Basic usage
deepagents

# With specific model
deepagents --model claude-sonnet-4-5-20250929
deepagents --model gpt-4o

# Auto-approve all tool executions (no prompts)
deepagents --auto-approve

# Resume a previous conversation thread
deepagents -r              # Resume most recent
deepagents -r abc123       # Resume specific thread

# Use a named agent configuration
deepagents --agent mybot
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--model` | Specify LLM model (auto-detects provider) |
| `--agent` | Agent name for separate config/memory (default: `agent`) |
| `--auto-approve` | Skip human-in-the-loop approval prompts |
| `-r, --resume` | Resume previous thread (optionally specify thread ID) |
| `--sandbox` | Run in remote sandbox (`modal`, `runloop`, `daytona`) |

### CLI Commands

Inside the CLI, you can use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/new` | Start new conversation thread |
| `/threads` | List conversation threads |
| `/skills` | List available skills |
| `/clear` | Clear screen |
| `/exit` | Exit the CLI |

## Starting the Telegram Bot

### 1. Get a Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the prompts
3. Copy the token provided

### 2. Configure Environment

```bash
cd libs/deepagents-telegram

# Create .env file
cat > .env << 'EOF'
TELEGRAM_BOT_TOKEN=your_bot_token_here
ANTHROPIC_API_KEY=your_anthropic_key
EOF
```

### 3. Start the Bot

```bash
# Load environment and run
export $(cat .env | grep -v '^#' | xargs) && deepagents-telegram
```

Or as a one-liner from project root:

```bash
cd libs/deepagents-telegram && export $(cat .env | grep -v '^#' | xargs) && deepagents-telegram
```

### Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/help` | Show help |
| `/new` | Start new conversation thread |
| `/threads` | List conversation threads |
| `/skills` | List available skills |
| `/status` | Show current session status |
| `/autoapprove` | Toggle auto-approve mode |

## Shared Agent Configuration

**CLI and Telegram share the same agent by default.** This means they share:

| Resource | Location |
|----------|----------|
| Agent memory | `~/.deepagents/agent/AGENTS.md` |
| User skills | `~/.deepagents/agent/skills/` |
| Project skills | `.deepagents/skills/` (in project root) |
| Conversation history | `~/.deepagents/sessions.db` |

### Sharing Conversation Threads

Both interfaces use the same SQLite database for conversation history. To continue a conversation across interfaces:

1. **Get thread ID from Telegram**: Use `/status` to see current thread ID
2. **Resume in CLI**: `deepagents -r <thread_id>`

Or vice versa:

1. **Get thread ID from CLI**: Run `deepagents threads list`
2. **Switch in Telegram**: (Currently requires code change to set `session.thread_id`)

### Using Different Agent Names

If you want separate configurations for CLI and Telegram:

```bash
# CLI with custom agent
deepagents --agent my-cli-agent

# Telegram uses "agent" by default
# To change, edit libs/deepagents-telegram/deepagents_telegram/handlers/messages.py
# Change: assistant_id = "agent" to assistant_id = "my-telegram-agent"
```

## Directory Structure

```
~/.deepagents/
├── agent/                    # Default agent configuration
│   ├── AGENTS.md             # Agent memory/personality
│   └── skills/               # User-level skills
│       └── my-skill/
│           └── SKILL.md
├── sessions.db               # Conversation history (SQLite)
└── my-other-agent/           # Another agent configuration
    ├── AGENTS.md
    └── skills/

your-project/
├── .git/
└── .deepagents/
    ├── AGENTS.md             # Project-specific instructions
    └── skills/               # Project-specific skills
```

## Model Configuration

### Environment Variables

```bash
# API Keys (at least one required)
export ANTHROPIC_API_KEY="sk-ant-xxx"
export OPENAI_API_KEY="sk-xxx"
export GOOGLE_API_KEY="xxx"

# Default models per provider
export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"
export OPENAI_MODEL="gpt-4o"
export GOOGLE_MODEL="gemini-2.5-pro"
```

### Custom OpenAI-Compatible APIs

For DeepSeek, Ollama, vLLM, LM Studio, etc:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_PROVIDER="openai"
export OPENAI_MODEL="your-model-name"
```

## Quick Reference

| Task | CLI | Telegram |
|------|-----|----------|
| Start | `deepagents` | `deepagents-telegram` |
| New thread | `/new` | `/new` |
| List threads | `deepagents threads list` | `/threads` |
| List skills | `/skills` or `deepagents skills list` | `/skills` |
| Auto-approve | `--auto-approve` flag | `/autoapprove` |
| Exit | `/exit` or Ctrl+C | Ctrl+C (server) |

## Troubleshooting

### "No LLM API key found"

Set at least one of:
```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### "TELEGRAM_BOT_TOKEN not set"

Get a token from @BotFather on Telegram and set:
```bash
export TELEGRAM_BOT_TOKEN="your-token"
```

### Model not found (404)

For custom APIs, ensure:
1. Base URL ends with `/v1`
2. Model name is correct (check with `curl $OPENAI_API_BASE/models`)
3. `OPENAI_PROVIDER=openai` is set
