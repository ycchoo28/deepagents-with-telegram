"""Callback query handlers for Telegram bot."""

from __future__ import annotations

import logging

from telegram import Update, ForceReply
from telegram.ext import ContextTypes

from ..session import get_session

logger = logging.getLogger(__name__)

# Track users waiting for edit input
_waiting_for_edit: dict[int, dict] = {}


def is_waiting_for_edit(chat_id: int) -> bool:
    """Check if a user is waiting to provide an edited command."""
    return chat_id in _waiting_for_edit


def get_edit_context(chat_id: int) -> dict | None:
    """Get the edit context for a user."""
    return _waiting_for_edit.get(chat_id)


def clear_edit_context(chat_id: int) -> None:
    """Clear the edit context for a user."""
    _waiting_for_edit.pop(chat_id, None)


async def handle_approval_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handle inline keyboard button presses for HITL approval.
    
    This handler processes approval decisions (approve/reject/auto_approve_all/edit)
    when users click buttons on approval request messages.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    query = update.callback_query
    
    if not query:
        return
    
    # Answer the callback query to remove loading state
    await query.answer()
    
    chat_id = query.message.chat_id
    decision = query.data  # "approve", "reject", "auto_approve_all", or "edit"
    
    logger.info(f"Approval callback from {chat_id}: {decision}")
    
    # Get the session and resolve the pending approval
    session = get_session(chat_id)
    
    if not session:
        await query.edit_message_text(
            "Session expired. Please start a new conversation.",
        )
        return
    
    if not session.adapter:
        await query.edit_message_text(
            "No pending approval request.",
        )
        return
    
    if not session.adapter.has_pending_approval():
        await query.edit_message_text(
            "This approval request has expired.",
        )
        return
    
    # Handle edit request
    if decision == "edit":
        # Get the current action request
        action_request = session.adapter.get_pending_action_request()
        if action_request:
            tool_name = action_request.get("name", "")
            tool_args = action_request.get("args", {})
            
            # Get the current command
            current_command = ""
            if tool_name in ("shell", "execute"):
                current_command = tool_args.get("command", "")
            
            # Store edit context
            _waiting_for_edit[chat_id] = {
                "action_request": action_request,
                "original_message_id": query.message.message_id,
            }
            session.adapter.set_waiting_for_edit(True)
            
            # Ask user for edited command
            await query.edit_message_text(
                f"📝 *Edit Command*\n\n"
                f"Current command:\n`{current_command}`\n\n"
                f"Reply with the edited command, or send /cancel to reject.",
                parse_mode="Markdown",
            )
            
            # Send a force reply to make it easier for user
            await context.bot.send_message(
                chat_id=chat_id,
                text="Enter the edited command:",
                reply_markup=ForceReply(selective=True, input_field_placeholder="Enter command..."),
            )
            return
        else:
            await query.answer("Cannot edit this action", show_alert=True)
            return
    
    # Resolve the approval for non-edit decisions
    session.adapter.resolve_approval(decision)
    
    # Update the message to show the decision
    if decision == "approve":
        status_text = "✅ Approved"
        status_emoji = "✅"
    elif decision == "reject":
        status_text = "❌ Rejected"
        status_emoji = "❌"
    elif decision == "auto_approve_all":
        status_text = "🔓 Auto-approve enabled"
        status_emoji = "🔓"
    else:
        status_text = f"⚠️ Unknown: {decision}"
        status_emoji = "⚠️"
    
    try:
        # Get the original message text and append the decision
        original_text = query.message.text or ""
        
        # Remove the inline keyboard and update text
        await query.edit_message_text(
            text=f"{original_text}\n\n{status_text}",
        )
    except Exception as e:
        logger.warning(f"Failed to edit approval message: {e}")
        # Try simpler update
        try:
            await query.edit_message_text(
                text=f"Action {status_emoji}",
            )
        except Exception:
            pass
    
    logger.info(f"Resolved approval for {chat_id}: {decision}")


async def handle_edit_response(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    """
    Handle a message that might be an edit response.
    
    Returns True if the message was handled as an edit response.
    
    Args:
        update: Telegram update object
        context: Bot context
        
    Returns:
        True if handled, False otherwise
    """
    if not update.message or not update.message.text:
        return False
    
    chat_id = update.effective_chat.id
    
    if not is_waiting_for_edit(chat_id):
        return False
    
    text = update.message.text
    
    # Handle cancel
    if text.lower() == "/cancel":
        session = get_session(chat_id)
        if session and session.adapter:
            session.adapter.set_waiting_for_edit(False)
            session.adapter.resolve_approval("reject")
        clear_edit_context(chat_id)
        
        await update.message.reply_text("❌ Command rejected.")
        return True
    
    # Get the session and resolve with edited value
    session = get_session(chat_id)
    if session and session.adapter:
        session.adapter.set_waiting_for_edit(False)
        session.adapter.resolve_approval("edit", edit_value=text)
        
        await update.message.reply_text(f"✏️ Using edited command:\n`{text}`", parse_mode="Markdown")
    
    clear_edit_context(chat_id)
    return True


async def handle_generic_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handle other callback queries that aren't approval-related.
    
    This is a catch-all handler for other inline button callbacks.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    query = update.callback_query
    
    if not query:
        return
    
    await query.answer()
    
    callback_data = query.data
    logger.info(f"Generic callback: {callback_data}")
    
    # Handle different callback types based on data prefix
    if callback_data.startswith("thread_"):
        # Switch to a different thread
        thread_id = callback_data.replace("thread_", "")
        chat_id = query.message.chat_id
        session = get_session(chat_id)
        
        if session:
            session.thread_id = thread_id
            await query.edit_message_text(
                f"✅ Switched to thread: `{thread_id}`",
                parse_mode="Markdown",
            )
        else:
            await query.edit_message_text(
                "❌ Session not found.",
            )
    
    elif callback_data.startswith("skill_"):
        # Show skill details (future feature)
        skill_name = callback_data.replace("skill_", "")
        await query.edit_message_text(
            f"📚 Skill: {skill_name}\n\nDetails coming soon...",
        )
    
    else:
        # Unknown callback
        await query.answer(text="Unknown action", show_alert=True)
