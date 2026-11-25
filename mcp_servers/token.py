"""Mint a development token for an external MCP client.

    uv run python -m mcp_servers.token --user alice --groups employees,engineering

Paste the printed value into the client's ``Authorization`` header as
``Bearer <token>``. Uses MCP_TOKEN_SECRET from .env; the token expires.
"""

import argparse

from chatbot.config import Settings
from chatbot.tools.identity import mint_dev_token


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mint a dev identity token for MCP clients")
    parser.add_argument("--user", required=True, help="External user id, e.g. alice")
    parser.add_argument("--groups", default="", help="Comma-separated groups for document access")
    parser.add_argument("--ttl", type=int, default=3600, help="Lifetime in seconds")
    args = parser.parse_args(argv)
    groups = sorted({g.strip() for g in args.groups.split(",") if g.strip()})
    token, expires = mint_dev_token(
        Settings().mcp_token_secret, external_id=args.user, groups=groups, ttl_seconds=args.ttl
    )
    print(token)
    print(f"# expires {expires.isoformat(timespec='seconds')}; groups: {groups or 'none'}")


if __name__ == "__main__":
    main()
