"""Message handler for Telegram bot."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command, Interrupt
from pydantic import TypeAdapter, ValidationError

try:
    from langchain.agents.middleware.human_in_the_loop import (
        HITLRequest,
        HITLResponse,
    )
    _HITL_REQUEST_ADAPTER = TypeAdapter(HITLRequest)
except ImportError:
    # Fallback for older versions
    HITLRequest = None
    HITLResponse = None
    _HITL_REQUEST_ADAPTER = None

from ..adapter import TelegramAdapter
from ..session import get_or_create_session
from ..formatters import format_tool_call, truncate_text

logger = logging.getLogger(__name__)

# Global agent cache (for demo - in production use proper DI)
_agent_cache: dict[str, tuple[Any, Any]] = {}


def get_model():
    """Get the configured LLM model."""
    try:
        from deepagents_cli.config import create_model
        return create_model()
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise


async def get_agent(assistant_id: str = "agent"):
    """Get or create an agent instance."""
    if assistant_id in _agent_cache:
        return _agent_cache[assistant_id]
    
    try:
        from deepagents_cli.agent import create_cli_agent
        from deepagents_cli.sessions import get_checkpointer
        
        async with get_checkpointer() as checkpointer:
            agent, backend = create_cli_agent(
                model=get_model(),
                assistant_id=assistant_id,
                checkpointer=checkpointer,
                enable_shell=True,
                enable_memory=True,
                enable_skills=True,
            )
            _agent_cache[assistant_id] = (agent, backend)
            return agent, backend
            
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle incoming text messages.
    
    This is the main message handler that sends user input to the agent
    and streams the response back to Telegram.
    
    Args:
        update: Telegram update object
        context: Bot context
    """
    chat_id = update.effective_chat.id
    user_input = update.message.text
    
    if not user_input:
        return
    
    # Check if this is an edit response first
    from .callbacks import handle_edit_response
    if await handle_edit_response(update, context):
        return  # Message was handled as an edit response
    
    # Get or create session
    session = get_or_create_session(chat_id)
    
    # Check if already processing
    if session.is_processing:
        await update.message.reply_text(
            "⏳ Still processing previous request. Please wait..."
        )
        return
    
    session.is_processing = True
    
    # Create adapter for this conversation
    adapter = TelegramAdapter(context.bot, chat_id)
    session.adapter = adapter
    
    try:
        await execute_task_telegram(
            user_input=user_input,
            session=session,
            adapter=adapter,
        )
    except Exception as e:
        logger.exception(f"Error processing message: {e}")
        await adapter.mount_message(f"Error: {e}", msg_type="error")
    finally:
        session.is_processing = False
        session.adapter = None
        await adapter.clear_status()


async def execute_task_telegram(
    user_input: str,
    session: Any,
    adapter: TelegramAdapter,
) -> None:
    """
    Execute a task with output directed to Telegram.
    
    This is a simplified version of execute_task_textual adapted for Telegram.
    
    Args:
        user_input: The user's input message
        session: The session state
        adapter: The TelegramAdapter for messaging
    """
    assistant_id = "agent"
    
    try:
        # Import here to allow graceful fallback
        from deepagents_cli.agent import create_cli_agent
        from deepagents_cli.sessions import get_checkpointer
        
        await adapter.update_status("Connecting to agent...")
        
        async with get_checkpointer() as checkpointer:
            agent, backend = create_cli_agent(
                model=get_model(),
                assistant_id=assistant_id,
                checkpointer=checkpointer,
                enable_shell=True,
                enable_memory=True,
                enable_skills=True,
            )
            
            await _run_agent_loop(
                user_input=user_input,
                agent=agent,
                backend=backend,
                session=session,
                adapter=adapter,
                assistant_id=assistant_id,
            )
            
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        await adapter.mount_message(
            f"Missing dependency: {e}\nMake sure deepagents-cli is installed.",
            msg_type="error",
        )
    except Exception as e:
        logger.exception(f"Agent execution error: {e}")
        await adapter.mount_message(str(e), msg_type="error")


async def _run_agent_loop(
    user_input: str,
    agent: Any,
    backend: Any,
    session: Any,
    adapter: TelegramAdapter,
    assistant_id: str,
) -> None:
    """
    Run the agent execution loop.
    
    Handles streaming, tool calls, and HITL approvals.
    """
    thread_id = session.thread_id
    config = {
        "configurable": {"thread_id": thread_id},
    }
    
    await adapter.update_status("Agent is thinking...")
    
    stream_input: dict | Command = {
        "messages": [{"role": "user", "content": user_input}]
    }
    
    accumulated_text = ""
    last_sent_length = 0
    
    try:
        while True:
            interrupt_occurred = False
            pending_interrupts: dict = {}
            
            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates"],
                subgraphs=True,
                config=config,
            ):
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue
                
                namespace, stream_mode, data = chunk
                ns_key = tuple(namespace) if namespace else ()
                is_main_agent = ns_key == ()
                
                # Handle interrupts (HITL approval requests)
                if stream_mode == "updates":
                    if not isinstance(data, dict):
                        continue
                    
                    if "__interrupt__" in data:
                        interrupts = data["__interrupt__"]
                        if interrupts and _HITL_REQUEST_ADAPTER:
                            for interrupt_obj in interrupts:
                                try:
                                    validated = _HITL_REQUEST_ADAPTER.validate_python(
                                        interrupt_obj.value
                                    )
                                    pending_interrupts[interrupt_obj.id] = validated
                                    interrupt_occurred = True
                                except (ValidationError, Exception) as e:
                                    logger.warning(f"Invalid interrupt: {e}")
                
                # Handle messages (content and tool calls)
                elif stream_mode == "messages":
                    if not is_main_agent:
                        continue
                    
                    if not isinstance(data, tuple) or len(data) != 2:
                        continue
                    
                    message, _metadata = data
                    
                    # Handle tool results
                    if isinstance(message, ToolMessage):
                        tool_name = getattr(message, "name", "")
                        tool_status = getattr(message, "status", "success")
                        tool_content = str(message.content)[:500] if message.content else ""
                        
                        success = tool_status == "success"
                        await adapter.mount_message(
                            tool_content or ("Success" if success else "Failed"),
                            msg_type="tool_result",
                            success=success,
                        )
                        
                        await adapter.update_status("Agent is thinking...")
                        continue
                    
                    # Handle AI message chunks
                    if not hasattr(message, "content_blocks"):
                        continue
                    
                    for block in message.content_blocks:
                        block_type = block.get("type")
                        
                        if block_type == "text":
                            text = block.get("text", "")
                            if text:
                                accumulated_text += text
                        
                        elif block_type in ("tool_call_chunk", "tool_call"):
                            tool_name = block.get("name")
                            tool_args = block.get("args", {})
                            
                            if tool_name:
                                # Send any accumulated text first
                                if accumulated_text and len(accumulated_text) > last_sent_length + 100:
                                    await adapter.mount_message(accumulated_text)
                                    last_sent_length = len(accumulated_text)
                                
                                # Parse args if string
                                if isinstance(tool_args, str):
                                    try:
                                        import json
                                        tool_args = json.loads(tool_args)
                                    except:
                                        tool_args = {"args": tool_args}
                                
                                await adapter.mount_message(
                                    "",
                                    msg_type="tool",
                                    tool_name=tool_name,
                                    tool_args=tool_args if isinstance(tool_args, dict) else {},
                                )
                                
                                await adapter.update_status(f"Running {tool_name}...")
            
            # Send any remaining accumulated text
            if accumulated_text and len(accumulated_text) > last_sent_length:
                await adapter.mount_message(accumulated_text)
            
            # Handle HITL interrupts
            # The HITL middleware expects resume={interrupt_id: {"decisions": [...]}, ...}
            # where each decision is {"type": "approve"} or {"type": "reject", "message": "..."}
            # IMPORTANT: The number of decisions must match the number of action_requests per interrupt
            if interrupt_occurred and pending_interrupts:
                hitl_response: dict[str, dict] = {}
                any_rejected = False
                
                for interrupt_id, hitl_request in pending_interrupts.items():
                    # Extract action_requests from HITLRequest
                    # HITLRequest has action_requests: list[ActionRequest]
                    # We need to generate one decision per action_request
                    action_requests = None
                    if isinstance(hitl_request, dict):
                        action_requests = hitl_request.get("action_requests", [])
                    elif hasattr(hitl_request, "action_requests"):
                        action_requests = hitl_request.action_requests
                    
                    if not action_requests:
                        # Fallback to old structure for compatibility
                        action = getattr(hitl_request, "action", None)
                        if action is None:
                            action = hitl_request.get("action", {}) if isinstance(hitl_request, dict) else {}
                        action_requests = [action] if action else []
                    
                    # Edge case: no action requests found
                    if not action_requests:
                        logger.warning(f"HITLRequest has no action_requests: {hitl_request}")
                        # Provide a default approval to avoid middleware errors
                        hitl_response[interrupt_id] = {"decisions": [{"type": "approve"}]}
                        continue
                    
                    decisions: list[dict] = []
                    
                    # Check auto-approve - approve all actions
                    if session.auto_approve:
                        decisions = [{"type": "approve"} for _ in action_requests]
                        hitl_response[interrupt_id] = {"decisions": decisions}
                        continue
                    
                    # Process each action request
                    for idx, action in enumerate(action_requests):
                        if isinstance(action, dict):
                            action_name = action.get("name", "unknown")
                            action_args = action.get("args", {})
                        elif hasattr(action, "name"):
                            action_name = action.name
                            action_args = getattr(action, "args", {})
                        else:
                            action_name = "unknown"
                            action_args = {}
                        
                        # Request approval from user for this action
                        action_request_info = {
                            "name": action_name,
                            "args": action_args,
                            "description": getattr(hitl_request, "description", ""),
                        }
                        
                        try:
                            result = await asyncio.wait_for(
                                adapter.request_approval(action_request_info, assistant_id),
                                timeout=300.0,  # 5 minute timeout
                            )
                            
                            if result.get("type") == "auto_approve_all":
                                session.enable_auto_approve()
                                # Approve this action
                                decisions.append({"type": "approve"})
                                # Approve all remaining actions in this interrupt
                                remaining_count = len(action_requests) - len(decisions)
                                for _ in range(remaining_count):
                                    decisions.append({"type": "approve"})
                                await adapter.mount_message(
                                    "Auto-approve enabled for this session.",
                                    msg_type="system",
                                )
                                break  # Exit inner loop, continue to next interrupt
                                
                            elif result.get("type") == "edit":
                                # User edited the command
                                edited_value = result.get("value", "")
                                if action_name in ("shell", "execute") and edited_value:
                                    # Modify the action args with the edited command
                                    # Note: We update the original action so the agent uses it
                                    if isinstance(action, dict):
                                        action["args"] = {"command": edited_value}
                                    elif hasattr(action, "args"):
                                        action.args = {"command": edited_value}
                                    
                                    decisions.append({"type": "approve"})
                                    await adapter.mount_message(
                                        f"Executing edited command: `{edited_value}`",
                                        msg_type="system",
                                    )
                                else:
                                    decisions.append({"type": "approve"})
                                    
                            elif result.get("type") == "reject":
                                decisions.append({
                                    "type": "reject",
                                    "message": "Action rejected by user",
                                })
                                any_rejected = True
                            else:
                                decisions.append({"type": "approve"})
                                
                        except asyncio.TimeoutError:
                            decisions.append({
                                "type": "reject",
                                "message": "Approval timed out",
                            })
                            any_rejected = True
                            await adapter.mount_message(
                                "Approval timed out - action rejected.",
                                msg_type="system",
                            )
                    
                    hitl_response[interrupt_id] = {"decisions": decisions}
                
                # If any action was rejected, inform user and stop
                if any_rejected:
                    await adapter.mount_message(
                        "Command rejected. Tell me what you'd like instead.",
                        msg_type="system",
                    )
                    break
                
                # Resume with responses in the format expected by HITL middleware
                stream_input = Command(resume=hitl_response)
                accumulated_text = ""
                last_sent_length = 0
                continue
            
            # No more interrupts, we're done
            break
            
    except Exception as e:
        logger.exception(f"Agent loop error: {e}")
        await adapter.mount_message(str(e), msg_type="error")
