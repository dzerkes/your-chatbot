"""Terminal chat and replay: python -m chatbot.cli {chat,replay} --help."""

import argparse
import asyncio
import sys
from contextlib import AsyncExitStack, aclosing
from itertools import groupby
from uuid import UUID

from pydantic import ValidationError

from chatbot.agent import Agent
from chatbot.app import build_chatbot
from chatbot.config import Settings
from chatbot.errors import ChatbotError, NotFoundError
from chatbot.logging import configure_logging
from chatbot.store.base import Session
from chatbot.types import (
    Message,
    Text,
    TextDelta,
    ToolCallStarted,
    TurnCompleted,
    TurnFailed,
    User,
)


def _identity(args: argparse.Namespace) -> User:
    return User(
        external_id=args.user, display_name=args.display_name or args.user, groups=args.groups
    )


async def _prepare(chatbot: Agent, args: argparse.Namespace) -> tuple[User, Session]:
    user = await chatbot.store.upsert_user(_identity(args))
    if args.session is None:
        session = await chatbot.store.create_session(user.id)
    else:
        existing = await chatbot.store.get_session(args.session, user.id)
        if existing is None:
            raise NotFoundError("Session not found for this user")
        session = existing
    return user, session


async def _reply(chatbot: Agent, session: Session, question: str, user: User) -> bool:
    failed = False
    trace: str | None = None
    async with aclosing(chatbot.respond(session.id, question, user)) as events:
        async for event in events:
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStarted):
                print(f"[tool: {event.tool_use.name}]", file=sys.stderr)
            elif isinstance(event, TurnCompleted):
                trace = event.trace_id
            elif isinstance(event, TurnFailed):
                failed = True
                print(f"Error ({event.code}): {event.message}", file=sys.stderr)
    print(flush=True)
    if trace is not None:
        print(f"[trace: {trace}]", file=sys.stderr)
    return failed


def _text(message: Message) -> str:
    return "".join(block.text for block in message.content if isinstance(block, Text))


async def _replay(chatbot: Agent, user: User, source: UUID, until: UUID | None) -> bool:
    """Ask a stored session's questions again, in order, in a new session."""
    if await chatbot.store.get_session(source, user.id) is None:
        raise NotFoundError("Session not found for this user")
    traces = {turn.id: turn.trace_id for turn in await chatbot.store.list_turns(source, user.id)}
    turns: list[tuple[UUID, str, str]] = []
    for turn_id, grouped in groupby(
        await chatbot.store.list_messages(source, user.id), key=lambda m: m.turn_id
    ):
        messages = list(grouped)
        question = "".join(_text(m) for m in messages if m.role == "user")
        answer = "".join(_text(m) for m in messages if m.role == "assistant")
        turns.append((turn_id, question, answer))
        if turn_id == until:
            break
    else:
        if until is not None:
            raise NotFoundError("Turn not found in this session")
    replay = await chatbot.store.create_session(user.id, title=f"Replay of {source}")
    print(f"Session: {replay.id} (replay of {source})", file=sys.stderr)
    failed = False
    for number, (turn_id, question, answer) in enumerate(turns, 1):
        print(f"=== Turn {number}/{len(turns)}: {turn_id} (trace {traces.get(turn_id) or '-'})")
        print(f"Question: {question}")
        print(f"--- Original answer\n{answer or '(none)'}")
        print("--- Replayed answer", flush=True)
        failed = await _reply(chatbot, replay, question, user) or failed
    return failed


def chat(args: argparse.Namespace, settings: Settings) -> int:
    # Keep input() on the main thread, outside async execution, so Ctrl-C while
    # waiting for a question cannot leave an executor thread blocked on stdin.
    with asyncio.Runner() as runner:
        stack = AsyncExitStack()
        try:
            chatbot = runner.run(stack.enter_async_context(build_chatbot(settings)))
            if getattr(args, "command", "chat") == "replay":
                user = runner.run(chatbot.store.upsert_user(_identity(args)))
                return 1 if runner.run(_replay(chatbot, user, args.session, args.until)) else 0
            user, session = runner.run(_prepare(chatbot, args))
            print(f"Session: {session.id}", file=sys.stderr)
            while True:
                if args.prompt is not None:
                    question = args.prompt
                else:
                    try:
                        question = input("You: ")
                    except EOFError:
                        return 0
                    if question.strip().lower() in {"/quit", "/exit"}:
                        return 0
                    if not question.strip():
                        continue
                failed = runner.run(_reply(chatbot, session, question, user))
                if args.prompt is not None:
                    return 1 if failed else 0
        finally:
            runner.run(stack.aclose())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat over company project data")
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("chat", help="Start or resume a terminal conversation")
    replay = subcommands.add_parser(
        "replay",
        help="Ask a saved session's questions again in a new session with the current settings",
    )
    for sub in (command, replay):
        sub.add_argument("--user", default="cli-user", help="Local development user identity")
        sub.add_argument("--display-name")
        sub.add_argument(
            "--groups",
            type=lambda value: sorted({g.strip() for g in value.split(",") if g.strip()}),
            default=[],
            help="Comma-separated groups that decide which documents search_docs may return",
        )
    command.add_argument("--session", type=UUID, help="Resume a session owned by this user")
    command.add_argument("--prompt", help="Ask one question and exit")
    replay.add_argument("--session", type=UUID, required=True, help="Session to replay")
    replay.add_argument("--until", type=UUID, help="Stop after this turn")
    args = parser.parse_args(argv)
    try:
        settings = Settings()
        configure_logging(settings)
        return chat(args, settings)
    except KeyboardInterrupt:
        print("\nChat interrupted.", file=sys.stderr)
        return 130
    except ValidationError:
        print("Invalid settings or user details. Check your configuration.", file=sys.stderr)
        return 1
    except ChatbotError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print(
            "Could not start chat. Check database availability and configuration.", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
