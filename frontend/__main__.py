"""Run the installed Streamlit entrypoint using FRONTEND_PORT."""

from importlib.resources import files

from streamlit.web import cli

from frontend.config import FrontendSettings


def main() -> None:
    settings = FrontendSettings()
    cli.main(
        [
            "run",
            str(files("frontend").joinpath("app.py")),
            "--server.port",
            str(settings.frontend_port),
            "--server.address",
            "127.0.0.1",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]
    )


if __name__ == "__main__":
    main()
