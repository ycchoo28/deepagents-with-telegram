"""Session state management for Telegram bot."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapter import TelegramAdapter


def generate_thread_id() -> str:
    """Generate a unique thread ID."""
    return uuid.uuid4().hex[:8]


@dataclass
class TelegramSessionState:
    """
    Session state for a Telegram chat.
    
    Tracks conversation thread, auto-approve setting, and the active adapter.
    """
    
    chat_id: int
    thread_id: str = field(default_factory=generate_thread_id)
    auto_approve: bool = False
    adapter: "TelegramAdapter | None" = None
    is_processing: bool = False
    
    def new_thread(self) -> None:
        """Start a new conversation thread."""
        self.thread_id = generate_thread_id()
        self.auto_approve = False
    
    def enable_auto_approve(self) -> None:
        """Enable auto-approve mode for this session."""
        self.auto_approve = True
    
    def disable_auto_approve(self) -> None:
        """Disable auto-approve mode for this session."""
        self.auto_approve = False


# Global in-memory session store
# Maps chat_id -> TelegramSessionState
_sessions: dict[int, TelegramSessionState] = {}


def get_session(chat_id: int) -> TelegramSessionState | None:
    """
    Get an existing session for a chat.
    
    Args:
        chat_id: The Telegram chat ID
        
    Returns:
        The session state if it exists, None otherwise
    """
    return _sessions.get(chat_id)


def get_or_create_session(chat_id: int) -> TelegramSessionState:
    """
    Get or create a session for a chat.
    
    Args:
        chat_id: The Telegram chat ID
        
    Returns:
        The session state (existing or newly created)
    """
    if chat_id not in _sessions:
        _sessions[chat_id] = TelegramSessionState(chat_id=chat_id)
    return _sessions[chat_id]


def delete_session(chat_id: int) -> bool:
    """
    Delete a session for a chat.
    
    Args:
        chat_id: The Telegram chat ID
        
    Returns:
        True if session was deleted, False if it didn't exist
    """
    if chat_id in _sessions:
        del _sessions[chat_id]
        return True
    return False


def list_sessions() -> list[TelegramSessionState]:
    """
    List all active sessions.
    
    Returns:
        List of all session states
    """
    return list(_sessions.values())


def clear_all_sessions() -> int:
    """
    Clear all sessions.
    
    Returns:
        Number of sessions cleared
    """
    count = len(_sessions)
    _sessions.clear()
    return count
