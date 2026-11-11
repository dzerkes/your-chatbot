"""Errors consumers can handle without depending on a provider or transport."""


class ChatbotError(Exception):
    """Base for expected application failures; messages must be safe to show users."""

    code = "chatbot_error"

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ProviderError(ChatbotError):
    """A provider request failed. Adapters may mark transient failures retryable."""

    code = "provider_error"


class ToolError(ChatbotError):
    """A tool could not complete its request."""

    code = "tool_error"


class ContextOverflow(ChatbotError):
    """The built context exceeds the allowed input budget."""

    code = "context_overflow"


class StepLimitExceeded(ChatbotError):
    """The agent reached its configured step limit."""

    code = "step_limit_exceeded"
