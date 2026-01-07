"""In-memory conversation state management."""

from dataclasses import dataclass, field

from chatty.client.openai_client import Message


@dataclass
class Conversation:
    """Manages conversation history and token tracking."""

    messages: list[Message] = field(default_factory=list)
    server_reported_tokens: int | None = None
    context_window: int = 128000  # Default context window size

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str, usage: dict[str, int] | None = None) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append(Message(role="assistant", content=content))
        if usage and "total_tokens" in usage:
            self.server_reported_tokens = usage["total_tokens"]

    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation."""
        self.messages.append(Message(role="system", content=content))

    def clear(self) -> None:
        """Clear all messages and reset token count."""
        self.messages = []
        self.server_reported_tokens = None

    def get_messages_for_api(self) -> list[dict[str, str]]:
        """Return messages in OpenAI API format."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def get_token_display(self) -> str:
        """Get token count for status bar display."""
        if self.server_reported_tokens is not None:
            used = self.server_reported_tokens // 1000
            total = self.context_window // 1000
            return f"{used}K / {total}K tokens"

        # Try tiktoken estimate
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model("gpt-4")
            total_tokens = sum(len(enc.encode(m.content)) for m in self.messages)
            used = total_tokens // 1000
            total = self.context_window // 1000
            return f"~{used}K / {total}K tokens (est)"
        except ImportError:
            return "usage unknown"

    def is_near_context_limit(self, threshold: float = 0.8) -> bool:
        """Check if conversation is approaching context limit."""
        if self.server_reported_tokens is not None:
            return self.server_reported_tokens > (self.context_window * threshold)
        return False
