"""Command handlers for Telegram bot."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import ContextTypes

from ..session import get_or_create_session, get_session, list_sessions
from ..formatters import format_skills_list, format_threads_list

logger = logging.getLogger(__name__)

# Welcome message
WELCOME_MESSAGE = """
👋 *Welcome to DeepAgents Bot!*

I'm an AI assistant that can help you with various tasks.

*Commands:*
• /start - Show this welcome message
• /new - Start a new conversation thread
• /threads - List your conversation threads
• /skills - List available skills
• /status - Show current session status
• /autoapprove - Toggle auto-approve mode

Just send me a message to get started!
"""


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /start command - show welcome message.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    chat_id = update.effective_chat.id
    
    # Create or get session
    session = get_or_create_session(chat_id)
    
    await update.message.reply_text(
        WELCOME_MESSAGE,
        parse_mode="Markdown",
    )
    
    logger.info(f"User {chat_id} started bot, thread_id={session.thread_id}")


async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /new command - start a new conversation thread.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    chat_id = update.effective_chat.id
    session = get_or_create_session(chat_id)
    
    old_thread = session.thread_id
    session.new_thread()
    
    await update.message.reply_text(
        f"🔄 Started new conversation thread.\n"
        f"Previous: `{old_thread}`\n"
        f"New: `{session.thread_id}`",
        parse_mode="Markdown",
    )
    
    logger.info(f"User {chat_id} started new thread: {old_thread} -> {session.thread_id}")


async def threads_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /threads command - list conversation threads.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    chat_id = update.effective_chat.id
    session = get_session(chat_id)
    
    # For demo, we only have in-memory sessions
    # In a full implementation, this would query the database
    if session:
        await update.message.reply_text(
            f"💬 *Current Thread:* `{session.thread_id}`\n\n"
            f"Note: Thread history is stored in memory only for this demo.\n"
            f"Use /new to start a fresh conversation.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "No active session. Send a message to start a conversation.",
        )


async def skills_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /skills command - list available skills.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    try:
        # Try to import skills from deepagents-cli
        from deepagents_cli.skills.load import list_skills
        
        skills = list_skills()
        skills_data = [
            {
                "name": s.name,
                "description": s.description or "No description",
                "source": getattr(s, "source", "unknown"),
            }
            for s in skills
        ]
        
        response = format_skills_list(skills_data)
        
    except ImportError:
        response = (
            "📚 *Skills System*\n\n"
            "Skills are loaded from:\n"
            "• User: `~/.deepagents/agent/skills/`\n"
            "• Project: `.deepagents/skills/`\n\n"
            "Each skill is a directory with a `SKILL.md` file."
        )
    except Exception as e:
        logger.error(f"Error loading skills: {e}")
        response = f"❌ Error loading skills: {e}"
    
    await update.message.reply_text(response, parse_mode="Markdown")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /status command - show current session status.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    chat_id = update.effective_chat.id
    session = get_session(chat_id)
    
    if session:
        auto_approve_status = "✅ Enabled" if session.auto_approve else "❌ Disabled"
        processing_status = "⏳ Processing" if session.is_processing else "💤 Idle"
        
        await update.message.reply_text(
            f"📊 *Session Status*\n\n"
            f"Chat ID: `{chat_id}`\n"
            f"Thread ID: `{session.thread_id}`\n"
            f"Auto-approve: {auto_approve_status}\n"
            f"Status: {processing_status}",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "No active session. Send a message to start.",
        )


async def autoapprove_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /autoapprove command - toggle auto-approve mode.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    chat_id = update.effective_chat.id
    session = get_or_create_session(chat_id)
    
    # Toggle auto-approve
    if session.auto_approve:
        session.disable_auto_approve()
        await update.message.reply_text(
            "🔒 Auto-approve *disabled*.\n"
            "You will be asked to approve tool executions.",
            parse_mode="Markdown",
        )
    else:
        session.enable_auto_approve()
        await update.message.reply_text(
            "🔓 Auto-approve *enabled*.\n"
            "⚠️ Tools will execute without confirmation!",
            parse_mode="Markdown",
        )
    
    logger.info(f"User {chat_id} set auto_approve={session.auto_approve}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /help command - show help message.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    await update.message.reply_text(
        WELCOME_MESSAGE,
        parse_mode="Markdown",
    )
