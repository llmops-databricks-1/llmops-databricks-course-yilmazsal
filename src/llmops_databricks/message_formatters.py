from typing import Literal, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def _msg(role: Literal["system", "user", "assistant"], content: str) -> ChatMessage:
    """Helper function to format messages."""
    return {"role": role, "content": content}


def system(content: str) -> ChatMessage:
    """Format a system message."""
    return _msg("system", content)


def user(content: str) -> ChatMessage:
    """Format a user message."""
    return _msg("user", content)


def assistant(content: str) -> ChatMessage:
    """Format an assistant message."""
    return _msg("assistant", content)
