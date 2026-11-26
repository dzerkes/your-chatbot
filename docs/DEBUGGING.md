# Debugging a turn

How to follow one bad answer from the UI to its trace, its stored messages
and its audit rows, then reproduce it and test a fix. Running the system is
covered in [RUNBOOK.md](RUNBOOK.md).

Every turn has one `turn_id` and one `trace_id`. They are on the turn's row
in Postgres, every message and audit row of the turn, every log line written
while it ran, every streamed event, and its trace in Langfuse. Find either one
and you can find everything else.

| Where | What it tells you | Needs |
|---|---|---|
| `turns` | Status, cost, latency, token counts, `trace_id` | Postgres |
| `messages` | Exactly what the user, the model and the tools said, in order | Postgres |
| `audit_log` | The question, each tool call's arguments and outcome, which documents and rows it returned | Postgres (owner) |
| `feedback` | The user's thumbs up/down and comment | Postgres |
| Logs | Errors with stack traces, retries, fallbacks; with `LOG_LEVEL=DEBUG`, every model request | `docker compose logs` |
| Langfuse trace | Timeline: each model call with tokens, cost, latency and time to first token; each tool call with arguments and result size | Tracing on |

Postgres is the record of every turn. Langfuse is optional and helps most
with timing and cost.

## 1. Find the turn

Open a SQL prompt as the owner, which can read every table, including the
audit log:

```bash
make psql
```

**From the UI.** The address bar holds the conversation:
`http://localhost:8501/?session=<session_id>`. List its turns, newest last:

```sql
SELECT t.id AS turn_id, t.status, t.trace_id, t.cost_usd, t.latency_ms,
       t.created_at, f.rating, f.comment
FROM turns t LEFT JOIN feedback f ON f.turn_id = t.id
WHERE t.session_id = '<session_id>'
ORDER BY t.created_at;
```

The UI shows a conversation's turns in the same order. `status` is `completed`
or `failed`.

**From a thumbs down.** Votes name the turn directly:

```sql
SELECT f.created_at, f.comment, t.session_id, t.id AS turn_id, t.trace_id
FROM feedback f JOIN turns t ON t.id = f.turn_id
WHERE f.rating = 'down'
ORDER BY f.created_at DESC LIMIT 10;
```

**From a user's report**, without a link, search by the question they asked:

```sql
SELECT a.recorded_at, u.external_id, a.session_id, a.turn_id, a.trace_id, a.question
FROM audit_log a JOIN users u ON u.id = a.user_id
WHERE a.event_type = 'turn' AND a.question ILIKE '%meal allowance%'
ORDER BY a.recorded_at DESC LIMIT 10;
```

**From the API**, `GET /sessions/<session_id>` with the user's token returns
every stored message with its `turn_id`:

```bash
curl -s http://localhost:8000/sessions/<session_id> \
  -H 'Authorization: Bearer dev-token-alice' | jq '.messages[] | {turn_id, role}'
```

## 2. Read what happened

The stored messages are what the model saw and said: the user's message, each
assistant step with its tool calls, each tool result as the tool returned it,
and the final answer. The model saw tool results wrapped as untrusted data;
storage keeps them unwrapped.

```sql
SELECT sequence, role, model, jsonb_pretty(content) AS content
FROM messages WHERE turn_id = '<turn_id>' ORDER BY sequence;
```

The audit rows show the question, every tool call's arguments and outcome
(`returned`, `failed` or `interrupted`), and the documents or rows it returned:

```sql
SELECT recorded_at, event_type, coalesce(question, tool_name) AS what,
       arguments, outcome, sources
FROM audit_log WHERE turn_id = '<turn_id>' ORDER BY id;
```

To see the document chunks a search returned:

```sql
SELECT src->>'source' AS source, c.heading, left(c.text, 120) AS text
FROM audit_log a, jsonb_array_elements(a.sources) AS src
JOIN rag.chunks c ON c.id = (src->>'chunk_id')::bigint
WHERE a.turn_id = '<turn_id>' AND a.tool_name = 'search_docs';
```

What a user is allowed to search depends on their groups:

```sql
SELECT u.external_id, u.groups FROM sessions s JOIN users u ON u.id = s.owner_id
WHERE s.id = '<session_id>';
```

With these you can usually tell where the answer went wrong:

| What you see | Where to look next |
|---|---|
| No tool call, or the wrong tool | Tool descriptions (`chatbot/src/chatbot/tools/`) and the system prompt (`prompts/system.md`) |
| Right tool, poor arguments | The tool's input model field descriptions |
| Good arguments, wrong or missing documents | Retrieval: run the same query (below), check the document's `groups:` and that it was ingested |
| Right documents, wrong answer | The prompt, or the model; compare with another model by replaying |
| A tool result with `is_error` | The error text in the result, and the `mcp` or backend log |
| An earlier answer forgotten | Context: `CONTEXT_STRATEGY`, the session's `summary` column, and the debug log (below) |
| `status = 'failed'` | The log (next section); the code is in the UI message and the trace |

## 3. Logs

Every log line carries `session_id`, `turn_id` and `trace_id`:

```bash
docker compose logs backend mcp | grep <turn_id>
```

With `LOG_FORMAT=json`, filter with
`jq 'select(.turn_id == "<turn_id>")'`. Errors such as `Agent turn failed`
include the stack trace; provider retries and fallbacks are warnings.

`LOG_LEVEL=DEBUG` adds one line per model call, `Model request, step N, about
T input tokens, D turns dropped: {...}`, with the exact request as JSON: system
prompt, messages after the context strategy, tool definitions and wrapped tool
results. It contains user text and retrieved documents, so use it for local
debugging only.

## 4. The trace

With tracing on, open the turn's trace in Langfuse: search **Tracing** by the
`trace_id`, or, when self-hosted, go to
`http://localhost:3000/project/your-chatbot/traces/<trace_id>`. The trace is
`session` → `turn` → one generation per model call and one span per tool call.

- Generations show the served model (a fallback shows up here), input, output
  and cached tokens, cost, latency and time to first token.
- Tool spans show the arguments, result size, latency and whether it failed.
- Errors mark the span where they happened, with the error message.
- A `user-feedback` score is attached when the user voted. Sessions group all
  turns of a conversation.

Slow turns: compare the generations' latency with the tool spans'. Expensive
turns: look for many steps, or input tokens growing with history (check
`cache_read` tokens; prompt caching should cover the stable prefix).

## 5. Reproduce and test a fix

### Replay the session

`replay` asks the questions of a saved session again, in order, in a new
session for the same user, with the current code and settings. It prints each
original answer next to the new one:

```bash
uv run python -m chatbot.cli replay --user alice --groups employees,engineering \
    --session <session_id> --until <turn_id>
```

- `--until` stops after the bad turn, so later turns do not cost anything.
- Pass the identity and groups the user has (see the query above, or
  `AUTH_DEV_TOKENS`); the groups decide what `search_docs` can return.
- Environment variables override `.env`, so you can try another model or
  context strategy without editing it:
  `LLM_PRIMARY=anthropic:claude-sonnet-5-5 uv run python -m chatbot.cli replay ...`
- Prompt and tool description changes take effect directly, since the CLI
  runs the code in your checkout.
- Add `LOG_LEVEL=DEBUG` to see each request the model receives.

The replay is a new session, so it is stored, audited and traced like any
other; each turn prints its trace ID. It uses `TOOLS_TRANSPORT` from `.env`
(`inprocess` by default), so the CLI needs the local models that `make ingest`
downloaded. It calls the real models and costs what the original did. A replay
asks only the user's questions: if the original answers were part of what went
wrong, the new answers can diverge from turn to turn.

### Ask one question

```bash
uv run python -m chatbot.cli chat --user alice --groups employees,engineering \
    --prompt "What is the meal allowance for business travel now?"
```

### Check retrieval alone

To see what `search_docs` returns for a query without the model, call the
tool over MCP with a token for the user's groups. Mint the token with
`uv run python -m mcp_servers.token --user alice --groups employees,engineering`,
then connect with the MCP Inspector (`npx @modelcontextprotocol/inspector`) to
`http://localhost:8001/mcp` with the header `Authorization: Bearer <token>`, and
run `search_docs` with the model's arguments from the audit row.

### Keep it fixed

Once you understand the failure, add it to the eval set as a case in
`evals/datasets/demo.yaml`: the question, the tool and sources you expect, and
the facts the answer must contain. Run only that case while you work, then the
full set before you merge:

```bash
uv run python -m evals.runner --case <case-id>
uv run python -m evals.runner --baseline
```

A new case changes the case counts, so regenerate `evals/baseline.json` in the
same change (see [evals/README.md](../evals/README.md#baseline-and-ci)).

## Debugging without the web stack

The CLI runs the same agent as the backend, with the same Postgres history,
tools, audit and tracing, so most problems reproduce there. What it skips:
authentication, rate limits and cost caps, which are in the backend, and the
MCP transport unless `TOOLS_TRANSPORT=mcp`. Backend unit tests use
`MemoryStore` and `ScriptedProvider` to reproduce a turn with no network and no
cost; see [backend/README.md](../backend/README.md#tests).
