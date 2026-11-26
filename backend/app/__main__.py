"""Start the local backend using BACKEND_PORT from shared settings."""

import uvicorn
from chatbot.config import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run("app.main:app", host="127.0.0.1", port=settings.backend_port)


if __name__ == "__main__":
    main()
