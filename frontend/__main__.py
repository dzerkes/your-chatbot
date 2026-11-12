"""Run the installed Streamlit entrypoint using FRONTEND_PORT."""

import argparse
from importlib.resources import files

from streamlit.web import cli

from frontend.config import FrontendSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Streamlit chat UI")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    settings = FrontendSettings()
    cli.main(
        [
            "run",
            str(files("frontend").joinpath("app.py")),
            "--server.port",
            str(settings.frontend_port),
            "--server.address",
            args.host,
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]
    )


if __name__ == "__main__":
    main()
