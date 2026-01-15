"""Telegram UI Adapter - bridges agent execution with Telegram messaging."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import TelegramError

from .formatters import split_message, escape_markdown_v2, format_tool_call

logger = logging.getLogger(__name__)


# Tool-specific formatting for approval messages
TOOL_FORMATTERS = {
    "shell": lambda args: f"Command: `{args.get('command', 'N/A')}`",
    "execute": lambda args: f"Command: `{args.get('command', 'N/A')}`",
    "write_file": lambda args: f"File: `{args.get('path', 'N/A')}`\nContent length: {len(str(args.get('content', '')))} chars",
    "edit_file": lambda args: f"File: `{args.get('path', 'N/A')}`",
    "web_search": lambda args: f"Query: `{args.get('query', 'N/A')}`",
    "fetch_url": lambda args: f"URL: `{args.get('url', 'N/A')}`",
    "task": lambda args: f"Task: `{args.get('description', args.get('prompt', 'N/A')[:50])}`",
}


class TelegramAdapter:
    """
    Adapter that bridges DeepAgents execution with Telegram messaging.
    
    Implements the same callback interface as TextualUIAdapter from deepagents-cli,
    allowing reuse of the core agent execution logic.
    """
    
    def __init__(self, bot: Bot, chat_id: int):
        """
        Initialize the Telegram adapter.
        
        Args:
            bot: The Telegram Bot instance
            chat_id: The chat ID to send messages to
        """
        self.bot = bot
        self.chat_id = chat_id
        self._pending_approval: asyncio.Future | None = None
        self._pending_approval_id: str | None = None  # Track which approval is pending
        self._status_message_id: int | None = None
        self._last_tool_message_id: int | None = None
        self._approval_message_id: int | None = None
        self._token_tracker: Any = None
        self._edit_callback: Callable[[str], None] | None = None  # For edit option
    
    def set_token_tracker(self, tracker: Any) -> None:
        """Set token tracker (for compatibility with TextualUIAdapter interface)."""
        self._token_tracker = tracker
    
    async def mount_message(
        self,
        content: str,
        msg_type: str = "assistant",
        **kwargs: Any,
    ) -> int | None:
        """
        Send a message to Telegram.
        
        Args:
            content: The message content
            msg_type: Type of message (assistant, user, tool, error, system)
            **kwargs: Additional arguments (tool_name, tool_args for tool messages)
            
        Returns:
            Message ID if successful, None otherwise
        """
        if not content:
            return None
            
        try:
            if msg_type == "tool":
                return await self._send_tool_message(
                    tool_name=kwargs.get("tool_name", "unknown"),
                    tool_args=kwargs.get("tool_args", {}),
                )
            elif msg_type == "tool_result":
                return await self._update_tool_result(
                    result=content,
                    success=kwargs.get("success", True),
                )
            elif msg_type == "error":
                return await self._send_error_message(content)
            elif msg_type == "system":
                return await self._send_system_message(content)
            else:
                return await self._send_text_message(content)
        except TelegramError as e:
            logger.error(f"Failed to send message: {e}")
            return None
    
    async def _send_text_message(self, content: str) -> int | None:
        """Send a plain text/markdown message, splitting if necessary."""
        chunks = split_message(content, max_length=4000)
        message_id = None
        
        for chunk in chunks:
            try:
                # Try markdown first, fall back to plain text
                msg = await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=chunk,
                    parse_mode=None,  # Plain text to avoid escaping issues
                )
                message_id = msg.message_id
            except TelegramError as e:
                logger.warning(f"Failed to send message chunk: {e}")
                # Try without any formatting
                msg = await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=chunk,
                )
                message_id = msg.message_id
                
        return message_id
    
    async def _send_tool_message(self, tool_name: str, tool_args: dict) -> int:
        """Send a tool call notification."""
        text = format_tool_call(tool_name, tool_args)
        msg = await self.bot.send_message(
            chat_id=self.chat_id,
            text=f"🔧 {text}",
        )
        self._last_tool_message_id = msg.message_id
        return msg.message_id
    
    async def _update_tool_result(self, result: str, success: bool = True) -> int | None:
        """Update the last tool message with its result."""
        if not self._last_tool_message_id:
            return None
            
        status = "✅" if success else "❌"
        # Truncate result for display
        display_result = result[:500] + "..." if len(result) > 500 else result
        
        try:
            await self.bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=self._last_tool_message_id,
                text=f"{status} {display_result}",
            )
        except TelegramError as e:
            logger.warning(f"Failed to update tool message: {e}")
            # Send as new message if edit fails
            msg = await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"{status} {display_result}",
            )
            return msg.message_id
            
        return self._last_tool_message_id
    
    async def _send_error_message(self, error: str) -> int:
        """Send an error message."""
        msg = await self.bot.send_message(
            chat_id=self.chat_id,
            text=f"❌ Error: {error}",
        )
        return msg.message_id
    
    async def _send_system_message(self, message: str) -> int:
        """Send a system notification message."""
        msg = await self.bot.send_message(
            chat_id=self.chat_id,
            text=f"ℹ️ {message}",
        )
        return msg.message_id
    
    async def update_status(self, status: str) -> None:
        """
        Update or create a status message.
        
        Args:
            status: The status text to display
        """
        try:
            if self._status_message_id:
                # Edit existing status message
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self._status_message_id,
                    text=f"⏳ {status}",
                )
            else:
                # Create new status message
                msg = await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=f"⏳ {status}",
                )
                self._status_message_id = msg.message_id
        except TelegramError as e:
            logger.warning(f"Failed to update status: {e}")
    
    async def clear_status(self) -> None:
        """Clear/delete the status message."""
        if self._status_message_id:
            try:
                await self.bot.delete_message(
                    chat_id=self.chat_id,
                    message_id=self._status_message_id,
                )
            except TelegramError:
                pass
            self._status_message_id = None
    
    async def request_approval(
        self,
        action_request: dict,
        assistant_id: str | None = None,
    ) -> dict:
        """
        Request user approval for an action via inline keyboard.
        
        Displays tool-specific information and provides approve/reject/edit options.
        
        Args:
            action_request: The action details (name, args, description)
            assistant_id: The assistant ID (unused, for interface compatibility)
            
        Returns:
            Dict with decision: {"type": "approve"}, {"type": "reject"}, 
            {"type": "auto_approve_all"}, or {"type": "edit", "value": "..."}
        """
        tool_name = action_request.get("name", "unknown")
        tool_args = action_request.get("args", {})
        description = action_request.get("description", "")
        
        # Format tool-specific details
        if tool_name in TOOL_FORMATTERS:
            try:
                tool_details = TOOL_FORMATTERS[tool_name](tool_args)
            except Exception:
                tool_details = self._format_generic_args(tool_args)
        else:
            tool_details = self._format_generic_args(tool_args)
        
        # Build the approval message
        text_parts = [
            "🔐 *Approval Required*",
            "",
            f"*Tool:* `{tool_name}`",
        ]
        
        if description:
            # Clean description for display
            clean_desc = description.replace("*", "").replace("`", "")[:300]
            text_parts.append(f"*Description:* {clean_desc}")
        
        text_parts.append("")
        text_parts.append(tool_details)
        
        # Create inline keyboard with options
        # For shell/execute commands, add an "Edit" option
        if tool_name in ("shell", "execute"):
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Approve", callback_data="approve"),
                    InlineKeyboardButton("❌ Reject", callback_data="reject"),
                ],
                [
                    InlineKeyboardButton("✏️ Edit command", callback_data="edit"),
                    InlineKeyboardButton("🔓 Auto-approve all", callback_data="auto_approve_all"),
                ],
            ])
        else:
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Approve", callback_data="approve"),
                    InlineKeyboardButton("❌ Reject", callback_data="reject"),
                ],
                [
                    InlineKeyboardButton("🔓 Auto-approve all", callback_data="auto_approve_all"),
                ],
            ])
        
        try:
            msg = await self.bot.send_message(
                chat_id=self.chat_id,
                text="\n".join(text_parts),
                reply_markup=keyboard,
                parse_mode="Markdown",
            )
            self._approval_message_id = msg.message_id
        except TelegramError:
            # Fallback without markdown
            msg = await self.bot.send_message(
                chat_id=self.chat_id,
                text="\n".join(text_parts).replace("*", "").replace("`", ""),
                reply_markup=keyboard,
            )
            self._approval_message_id = msg.message_id
        
        # Store action request for potential editing
        self._pending_action_request = action_request
        
        # Create a future that will be resolved by the callback handler
        loop = asyncio.get_running_loop()
        self._pending_approval = loop.create_future()
        
        # Wait for user decision
        result = await self._pending_approval
        self._pending_approval = None
        self._pending_action_request = None
        
        return result
    
    def _format_generic_args(self, args: dict) -> str:
        """Format generic tool arguments for display."""
        if not args:
            return "No arguments"
        
        try:
            args_str = json.dumps(args, indent=2, ensure_ascii=False)
            if len(args_str) > 500:
                args_str = args_str[:500] + "\n..."
            return f"```\n{args_str}\n```"
        except Exception:
            args_str = str(args)
            if len(args_str) > 300:
                args_str = args_str[:300] + "..."
            return f"`{args_str}`"
    
    def get_pending_action_request(self) -> dict | None:
        """Get the current pending action request for editing."""
        return getattr(self, "_pending_action_request", None)
    
    def resolve_approval(self, decision: str, edit_value: str | None = None) -> None:
        """
        Resolve a pending approval request.
        
        Called by the callback handler when user clicks a button.
        
        Args:
            decision: One of "approve", "reject", "auto_approve_all", "edit"
            edit_value: If decision is "edit", the new value to use
        """
        if self._pending_approval and not self._pending_approval.done():
            if decision == "auto_approve_all":
                self._pending_approval.set_result({"type": "auto_approve_all"})
            elif decision == "reject":
                self._pending_approval.set_result({"type": "reject"})
            elif decision == "edit" and edit_value is not None:
                self._pending_approval.set_result({"type": "edit", "value": edit_value})
            elif decision == "edit":
                # Edit requested but no value yet - keep pending
                # The callback handler will prompt for input
                pass
            else:
                self._pending_approval.set_result({"type": "approve"})
    
    def set_waiting_for_edit(self, waiting: bool) -> None:
        """Set whether we're waiting for an edited command."""
        self._waiting_for_edit = waiting
    
    def is_waiting_for_edit(self) -> bool:
        """Check if we're waiting for an edited command."""
        return getattr(self, "_waiting_for_edit", False)
    
    def has_pending_approval(self) -> bool:
        """Check if there's a pending approval request."""
        return self._pending_approval is not None and not self._pending_approval.done()
    
    # Compatibility methods for TextualUIAdapter interface
    def on_auto_approve_enabled(self) -> None:
        """Called when auto-approve is enabled."""
        pass
    
    def scroll_to_bottom(self) -> None:
        """No-op for Telegram (no scrolling concept)."""
        pass
