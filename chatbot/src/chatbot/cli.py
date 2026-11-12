"""Terminal chat: python -m chatbot.cli chat [--prompt QUESTION]."""

import argparse
import asyncio
import sys
from contextlib import AsyncExitStack, aclosing
from uuid import UUID

from pydantic import ValidationError

from chatbot.agent import Agent
from chatbot.app import build_chatbot
from chatbot.config import Settings
from chatbot.errors import ChatbotError, NotFoundError
from chatbot.logging import configure_logging
from chatbot.store.base import Session
from chatbot.types import TextDelta, ToolCallStarted, TurnFailed, User


async def _prepare(chatbot: Agent, args: argparse.Namespace) -> tuple[User, Session]:
    user = await chatbot.store.upsert_user(
        User(external_id=args.user, display_name=args.display_name or args.user)
    )
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
    async with aclosing(chatbot.respond(session.id, question, user)) as events:
        async for event in events:
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStarted):
                print(f"[tool: {event.tool_use.name}]", file=sys.stderr)
            elif isinstance(event, TurnFailed):
                failed = True
                print(f"Error ({event.code}): {event.message}", file=sys.stderr)
    print()
    return failed


def chat(args: argparse.Namespace, settings: Settings) -> int:
    # Keep input() on the main thread, outside async execution, so Ctrl-C while
    # waiting for a question cannot leave an executor thread blocked on stdin.
    with asyncio.Runner() as runner:
        stack = AsyncExitStack()
        try:
            chatbot = runner.run(stack.enter_async_context(build_chatbot(settings)))
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
    command.add_argument("--user", default="cli-user", help="Local development user identity")
    command.add_argument("--display-name")
    command.add_argument("--session", type=UUID, help="Resume a session owned by this user")
    command.add_argument("--prompt", help="Ask one question and exit")
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
