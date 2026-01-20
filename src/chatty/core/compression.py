"""Context compression utilities.

This module provides LLM-based conversation summarization for reducing
context window usage while preserving key information.
"""

from __future__ import annotations

from chatty.client.openai_client import Message

COMPRESSION_PROMPT = """Summarize the following conversation concisely.
Preserve:
- Key topics discussed
- Important decisions or conclusions
- Technical details that may be referenced later

Conversation:
{conversation}

Summary:"""


def build_compression_prompt(messages: list[Message]) -> str:
    """Build prompt for compression LLM call.

    Formats conversation messages into a prompt for summarization.
    System messages are excluded from the summary.

    Args:
        messages: List of conversation messages.

    Returns:
        Formatted prompt string for the LLM.
    """
    conversation_lines: list[str] = []

    for msg in messages:
        if msg.role == "system":
            continue  # Skip system messages
        role_label = msg.role.capitalize()
        conversation_lines.append(f"{role_label}: {msg.content}")

    conversation_text = "\n\n".join(conversation_lines)
    return COMPRESSION_PROMPT.format(conversation=conversation_text)


def detect_code_blocks(messages: list[Message]) -> bool:
    """Check if any messages contain code blocks (``` markers).

    Args:
        messages: List of conversation messages.

    Returns:
        True if any message contains ``` markers.
    """
    return any("```" in msg.content for msg in messages)


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count using tiktoken.

    Args:
        text: Text to count tokens for.
        model: Model name for encoding selection.

    Returns:
        Estimated token count.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            # Model not recognized, use cl100k_base (GPT-4 encoding)
            enc = tiktoken.get_encoding("cl100k_base")

        return len(enc.encode(text))
    except ImportError:
        # Fallback: rough estimate of 4 chars per token
        return len(text) // 4


def estimate_messages_tokens(messages: list[Message], model: str = "gpt-4") -> int:
    """Estimate total token count for a list of messages.

    Args:
        messages: List of messages to count tokens for.
        model: Model name for encoding selection.

    Returns:
        Estimated total token count.
    """
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content, model)
        # Add overhead for role tokens (approximately 4 tokens per message)
        total += 4
    return total
