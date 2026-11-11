"""Provider-neutral history records and persistence protocols."""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, Protocol
from uuid import UUID, uuid4

from pydantic import AwareDatetime, Field, NonNegativeInt

from chatbot.types import DomainModel, Message, User


class Session(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    owner_id: UUID
    title: str | None = None
    summary: str | None = None
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))


class Turn(DomainModel):
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    trace_id: str | None = None
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cache_read_input_tokens: NonNegativeInt = 0
    cache_creation_input_tokens: NonNegativeInt = 0
    cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    latency_ms: NonNegativeInt | None = None
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: AwareDatetime | None = None


class SessionStore(Protocol):
    async def upsert_user(self, user: User) -> User:
        """Return the canonical user ID for an external identity."""

    async def create_session(self, owner_id: UUID, title: str | None = None) -> Session:
        """Create a session owned by an existing user."""

    async def get_session(self, session_id: UUID, owner_id: UUID) -> Session | None:
        """Return a session only when it belongs to owner_id."""

    async def list_sessions(self, owner_id: UUID) -> list[Session]:
        """List one user's sessions, newest activity first."""

    async def update_session(self, session: Session) -> Session:
        """Update title and summary for an existing session."""


class MessageStore(Protocol):
    async def save_turn(self, turn: Turn, messages: Sequence[Message], owner_id: UUID) -> None:
        """Insert one turn and all its messages atomically."""

    async def list_turns(self, session_id: UUID, owner_id: UUID) -> list[Turn]:
        """List turns in order, scoped to the session owner."""

    async def list_messages(self, session_id: UUID, owner_id: UUID) -> list[Message]:
        """List complete history in session sequence order."""


class HistoryStore(SessionStore, MessageStore, Protocol):
    """A store that provides both session and turn/message operations."""


def validate_turn_messages(turn: Turn, messages: Sequence[Message]) -> None:
    """Reject inconsistent batches before touching storage."""
    sequences: set[int] = set()
    message_ids: set[UUID] = set()
    for message in messages:
        if message.session_id != turn.session_id or message.turn_id != turn.id:
            raise ValueError("Every message must belong to the saved turn and session")
        if message.sequence in sequences:
            raise ValueError("Message sequences must be unique within a turn")
        if message.id in message_ids:
            raise ValueError("Message IDs must be unique within a turn")
        sequences.add(message.sequence)
        message_ids.add(message.id)
