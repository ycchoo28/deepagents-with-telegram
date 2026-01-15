"""Message formatting utilities for Telegram."""

from __future__ import annotations

import re
from typing import Any


# Telegram message length limit
MAX_MESSAGE_LENGTH = 4096

# Characters that need escaping in MarkdownV2
MARKDOWN_V2_SPECIAL_CHARS = r"_*[]()~`>#+-=|{}.!"


def escape_markdown_v2(text: str) -> str:
    """
    Escape special characters for Telegram MarkdownV2 format.
    
    Args:
        text: The text to escape
        
    Returns:
        Escaped text safe for MarkdownV2
    """
    # Escape all special characters
    for char in MARKDOWN_V2_SPECIAL_CHARS:
        text = text.replace(char, f"\\{char}")
    return text


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH - 100) -> list[str]:
    """
    Split a long message into chunks that fit Telegram's limit.
    
    Attempts to split at natural boundaries (newlines, sentences, words).
    
    Args:
        text: The text to split
        max_length: Maximum length per chunk (default: 4000 to leave room for formatting)
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    remaining = text
    
    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break
        
        # Try to find a good split point
        split_point = max_length
        
        # Try to split at a double newline (paragraph break)
        para_break = remaining.rfind("\n\n", 0, max_length)
        if para_break > max_length // 2:
            split_point = para_break + 2
        else:
            # Try to split at a single newline
            newline = remaining.rfind("\n", 0, max_length)
            if newline > max_length // 2:
                split_point = newline + 1
            else:
                # Try to split at a sentence end
                for end_char in [". ", "! ", "? "]:
                    pos = remaining.rfind(end_char, 0, max_length)
                    if pos > max_length // 2:
                        split_point = pos + 2
                        break
                else:
                    # Try to split at a space
                    space = remaining.rfind(" ", 0, max_length)
                    if space > max_length // 2:
                        split_point = space + 1
        
        chunks.append(remaining[:split_point])
        remaining = remaining[split_point:]
    
    return chunks


def format_tool_call(tool_name: str, tool_args: dict[str, Any]) -> str:
    """
    Format a tool call for display.
    
    Args:
        tool_name: Name of the tool
        tool_args: Tool arguments
        
    Returns:
        Formatted string for display
    """
    # Truncate long arguments
    args_display = {}
    for key, value in tool_args.items():
        str_value = str(value)
        if len(str_value) > 100:
            str_value = str_value[:100] + "..."
        args_display[key] = str_value
    
    if args_display:
        args_str = ", ".join(f"{k}={v}" for k, v in args_display.items())
        return f"{tool_name}({args_str})"
    else:
        return f"{tool_name}()"


def format_error(error: str | Exception) -> str:
    """
    Format an error message for display.
    
    Args:
        error: The error string or exception
        
    Returns:
        Formatted error string
    """
    error_str = str(error)
    if len(error_str) > 500:
        error_str = error_str[:500] + "..."
    return f"Error: {error_str}"


def format_code_block(code: str, language: str = "") -> str:
    """
    Format code as a code block.
    
    Args:
        code: The code to format
        language: Optional language for syntax highlighting
        
    Returns:
        Formatted code block
    """
    # Use plain text formatting for Telegram
    return f"```{language}\n{code}\n```"


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: The text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_file_diff(diff: str, file_path: str) -> str:
    """
    Format a file diff for display.
    
    Args:
        diff: The diff content
        file_path: Path to the file
        
    Returns:
        Formatted diff display
    """
    truncated_diff = truncate_text(diff, max_length=1500)
    return f"📄 {file_path}\n```diff\n{truncated_diff}\n```"


def format_skills_list(skills: list[dict]) -> str:
    """
    Format a list of skills for display.
    
    Args:
        skills: List of skill metadata dicts
        
    Returns:
        Formatted skills list
    """
    if not skills:
        return "No skills available."
    
    lines = ["📚 *Available Skills:*", ""]
    for skill in skills:
        name = skill.get("name", "unknown")
        description = skill.get("description", "No description")
        source = skill.get("source", "unknown")
        lines.append(f"• *{name}* ({source})")
        lines.append(f"  {description[:100]}")
        lines.append("")
    
    return "\n".join(lines)


def format_threads_list(threads: list[dict]) -> str:
    """
    Format a list of conversation threads for display.
    
    Args:
        threads: List of thread metadata dicts
        
    Returns:
        Formatted threads list
    """
    if not threads:
        return "No conversation threads found."
    
    lines = ["💬 *Conversation Threads:*", ""]
    for thread in threads[:10]:  # Limit to 10 most recent
        thread_id = thread.get("thread_id", "unknown")
        created = thread.get("created_at", "unknown")
        preview = thread.get("preview", "")[:50]
        lines.append(f"• `{thread_id}` - {created}")
        if preview:
            lines.append(f"  {preview}...")
        lines.append("")
    
    return "\n".join(lines)
