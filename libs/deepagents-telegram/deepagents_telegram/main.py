"""Main entry point for DeepAgents Telegram bot."""

from __future__ import annotations

import logging
import os
import sys

import dotenv

# Load .env file from current directory or parent directories
dotenv.load_dotenv()

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from .handlers.commands import (
    start_command,
    new_command,
    threads_command,
    skills_command,
    status_command,
    autoapprove_command,
    help_command,
)
from .handlers.messages import handle_message
from .handlers.callbacks import handle_approval_callback

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Reduce noise from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main entry point for the Telegram bot.
    
    Reads TELEGRAM_BOT_TOKEN from environment and starts the bot.
    """
    # Get bot token from environment
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        print()
        print("To get a token:")
        print("1. Open Telegram and search for @BotFather")
        print("2. Send /newbot and follow the prompts")
        print("3. Copy the token and set it:")
        print("   export TELEGRAM_BOT_TOKEN='your_token_here'")
        print()
        sys.exit(1)
    
    # Check for LLM API keys
    has_llm_key = any([
        os.environ.get("ANTHROPIC_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
    ])
    
    if not has_llm_key:
        print("Warning: No LLM API key found.")
        print("Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY")
        print()
    
    # Create application with concurrent updates enabled
    # This is critical for HITL approval flow - the message handler awaits user approval,
    # and the callback handler (for approval buttons) must run concurrently to resolve it.
    # Without concurrent_updates=True, handlers block sequentially, causing a deadlock.
    application = Application.builder().token(token).concurrent_updates(True).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("new", new_command))
    application.add_handler(CommandHandler("threads", threads_command))
    application.add_handler(CommandHandler("skills", skills_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("autoapprove", autoapprove_command))
    
    # Add message handler for text messages (excluding commands)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    
    # Add callback query handler for inline keyboard buttons
    application.add_handler(CallbackQueryHandler(handle_approval_callback))
    
    # Log startup
    logger.info("Starting DeepAgents Telegram bot...")
    print()
    print("=" * 50)
    print("  DeepAgents Telegram Bot")
    print("=" * 50)
    print()
    print("Bot is running! Send a message to your bot to start.")
    print("Press Ctrl+C to stop.")
    print()
    
    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
