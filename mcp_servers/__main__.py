"""Run the MCP data server: python -m mcp_servers [--host HOST]."""

import argparse

import uvicorn
from chatbot.config import Settings
from chatbot.logging import configure_logging
from chatbot.tools.identity import MIN_SECRET_LENGTH

from mcp_servers.data_server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the chatbot's data tools over MCP")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    settings = Settings()
    if len(settings.mcp_token_secret.get_secret_value()) < MIN_SECRET_LENGTH:
        raise SystemExit(f"Set MCP_TOKEN_SECRET to at least {MIN_SECRET_LENGTH} characters")
    configure_logging(settings)
    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=settings.mcp_server_port,
        log_config=None,
        timeout_graceful_shutdown=20,
    )


if __name__ == "__main__":
    main()
