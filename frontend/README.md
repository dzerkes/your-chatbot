# Project chat UI

Streamlit provides a sidebar of saved conversations, a new-chat draft, streamed
assistant replies, and collapsed tool status panels. All conversations and
company data come from the HTTP API. This package has no dependency on the
chatbot core, provider SDKs, database drivers or backend application.

## Run locally

From the repository root, configure `.env` with database credentials and the
selected model provider's API key. Install the workspace and load demo data:

```bash
uv sync
make seed
```

Start the backend in one terminal:

```bash
uv run --package chatbot-backend python -m app
```

Start the UI in another:

```bash
uv run --package chatbot-frontend python -m frontend
```

Open `http://localhost:8501`. The frontend launcher uses `FRONTEND_PORT` and binds
to localhost. `BACKEND_URL` selects the API, defaulting to
`http://localhost:8000`. For direct Streamlit invocation, use
`uv run streamlit run frontend/app.py`.

## Use the chat

1. Enter an access token in the sidebar. With the example backend configuration,
   use `dev-token-alice` or `dev-token-bob`.
2. Choose **New chat** and ask a question such as “Which Northstar projects are
   active?” The first question creates the session and supplies its sidebar title.
3. Watch the reply stream. Tool panels show progress, then a result count or an
   error. Expand a result panel to see how many results were returned in the page.
4. Reopen any saved conversation from the sidebar and continue it. **Refresh
   conversations** reloads the list from the API.

The selected session UUID is kept in the page's `session` query parameter.
A full page reload requires entering the access token again, then restores the
selected conversation from the API. Tokens remain in Streamlit's browser-session
state; the UI places session IDs in URLs and keeps authentication out of them.
Changing or clearing the token clears the previous conversation selection.
The backend checks ownership on every request.

**New chat** starts an empty draft and preserves existing conversations.
The UI reloads canonical history after a completed or failed turn. On a broken
stream, partial text remains visible with an error and a reload action. It never
automatically resubmits a message after a connection failure.

## HTTP boundary and tests

`api_client.py` contains frontend DTOs and an HTTPX client for the session and
POST chat endpoints. Its SSE parser handles comments and multiline JSON data,
checks event/session consistency, rejects oversized frames, and requires a
terminal event. HTTP response bodies and credentials are replaced with friendly
error messages when requests fail.

`config.py` reads only frontend settings. Clients are scoped to a UI run and
closed afterward, rather than cached across accounts.

```bash
uv run pytest frontend/tests
```

HTTP client tests use mock transports. Streamlit AppTest covers new chat, ask,
reload, reopen, continue, token changes and failed streams. An additional AppTest
walkthrough was run over HTTP against the real backend and seeded Postgres
database with a scripted provider. Native browser review requires Computer Use
permissions, which were unavailable during implementation.
