# Future work

The planned commits (C01 to C31) are done. This page lists what would bring the
project closer to its two goals: anyone can run it and play with it, and it
shows what a real production chatbot looks like. Items are in order of value.
Each one says why it matters, what to build, and how to know it is done.

Contributions are welcome. Pick an item, open an issue to discuss the approach,
and follow the existing pattern: one focused commit, tests, docs and a
changelog row.

## 1. Run without a paid API key

**Why.** Today you need an OpenAI or Anthropic key with credit before you can
ask a single question. That is the biggest barrier for someone who just wants
to try the project.

**What.**

- An `ollama:` provider in `chatbot/src/chatbot/llm/`, registered in
  `ProviderRegistry`, so `LLM_PRIMARY=ollama:<model>` runs a local model. The
  steps are in [RUNBOOK.md](RUNBOOK.md#a-model-provider).
- An optional `ollama` service in Compose, and zero prices in `prices.toml` so
  cost accounting still works.
- A `make demo` target that starts everything with a local model and no
  secrets to fill in: generated development passwords and the dev tokens.

**Done when** a fresh clone runs `make demo`, opens the UI and gets a grounded
answer with tool calls, without any API key.

## 2. A sturdier eval set

**Why.** The gate in CI is only as good as the set behind it. Today:

- There are 21 cases, too few to catch subtle regressions.
- The judge defaults to `LLM_PRIMARY`, so the model grades its own answers.
- The baseline comes from one run. Model output varies between runs, so the
  gate can fail on noise or miss a real drop.

**What.**

- Grow the set to 50 or more cases: multi-turn conversations, long documents,
  ambiguous questions, date and number handling, more access-control and
  injection cases.
- Use a separate judge model by default, set with `--judge-model` in CI.
- Run each case several times (`--repeats N`), and store the mean and spread
  in `evals/baseline.json`. Set tolerances from the measured spread.
- Track scores over time, for example as Langfuse datasets or an uploaded
  history file, to see slow drift rather than only single drops.

**Done when** the same unchanged code passes the gate ten runs in a row, and a
deliberately worse prompt still fails it.

## 3. Real sign-in

**Why.** `AUTH_MODE=dev` maps static bearer tokens to users. A real company
chatbot signs users in through the identity provider (SSO), and takes their
groups from it. This is the first thing anyone deploying the project asks
about.

**What.**

- `AUTH_MODE=oidc` in the backend: validate tokens against the provider's
  keys, and map the `sub` and `groups` claims to `User`. `current_user` in
  `backend/app/` is already the single replacement point.
- A Keycloak service in an optional Compose file, with demo users and groups
  that match the seeded documents.
- Login in the Streamlit UI through the OIDC authorization code flow.

**Done when** alice and bob log in through Keycloak, and their Keycloak groups
decide which documents they can search.

## 4. Metrics, dashboards and alerts

**Why.** Langfuse explains one turn. Operators also need the whole system at a
glance: request rate, error rate, latency, spend and provider health, with
alerts when they go wrong.

**What.**

- A `/metrics` endpoint (Prometheus format) on the backend and the MCP server:
  turns by status and error code, turn and time-to-first-token latency, tokens
  and cost by model, fallbacks, tool calls by tool and outcome, rate-limit and
  cost-cap rejections.
- Prometheus and Grafana in an optional Compose file, with one dashboard
  checked in.
- Example alert rules: error rate, p95 latency, daily spend, a provider
  failing over.

**Done when** the dashboard shows a burst of traffic, and stopping the primary
provider's network access shows up as fallbacks and then errors.

## 5. A guided tour

**Why.** The docs explain each part, but a learner still has to find the path
through the code. A tour turns the repo into a course.

**What.** `docs/TOUR.md` follows one question from the browser to the answer
and back: the Streamlit call, authentication, admission limits, `respond()`,
context building, the model call through the gateway, a tool call over MCP,
retrieval with access control, persistence, audit, the trace and the feedback.
Each step links to the file and function, with a sentence on why it is built
that way. It ends with small exercises, such as adding a tool or changing the
context strategy, and the eval run that checks them.

**Done when** someone new to the project can explain the request path after
reading it, without opening other docs.

## 6. Deployment beyond Compose

**Why.** Compose is right for a laptop. Production needs TLS, managed secrets,
more than one replica and a repeatable deploy.

**What.**

- A Helm chart or plain Kubernetes manifests for the backend, MCP server and
  frontend, with health checks, resource limits and secrets from the cluster.
- Make the backend safe to run as several replicas. Turns of one session are
  serialized by a lock inside the agent process; across replicas that needs a
  Postgres advisory lock on the session, or sticky routing. Database sequence
  constraints already reject conflicting writes, but the user gets an error.
- TLS at the ingress, and the database credentials from a secrets manager.
- A deploy workflow that builds and pushes images, and runs migrations once
  before the new version starts.

**Done when** two backend replicas run behind a load balancer and simultaneous
messages to one session are answered in order.

## 7. Data retention and user data requests

**Why.** The audit log keeps questions and tool arguments verbatim, forever.
Messages, memory and feedback have no retention either. A real deployment
needs a policy, and must be able to export or delete one user's data.

**What.**

- Configurable retention per table, applied by a scheduled job. The audit log
  is append-only by design, so its retention needs a deliberate exception, for
  example partitions dropped by the owner role after the retention period.
- An operator command to export everything stored about one user, and one to
  delete it, including Langfuse traces through its API.
- A short document on what is stored where, and for how long.

**Done when** deleting a user removes their sessions, messages, memory and
feedback, and the documented audit and trace retention applies.

## 8. Ingestion connectors

**Why.** Documents come from a folder through a CLI. Real company data lives
in wikis, drives and ticket systems, and changes every day.

**What.** Connectors behind one interface (for example a Confluence or Google
Drive reader) that produce the same Markdown and `groups` front matter the
ingester already understands, plus a scheduled job and deletion of documents
that disappear at the source (`--prune`). Document permissions should come
from the source system, not be set by hand.

**Done when** a document edited at the source is searchable in its new form
after the next scheduled run, with the source's permissions.

## 9. A production-style web UI

**Why.** Streamlit is quick to read and change, which is why it is used here.
It is not what a product team would ship.

**What.** An optional frontend, for example React or Next.js, that talks to the
same API: streamed answers, tool activity, sources with links to documents,
feedback, memory management and sign-in. The Streamlit UI stays as the simple
reference.

**Done when** both UIs work against the same backend with no backend changes.

## 10. Verify what has not run yet

Some parts are built and tested in pieces but have never run end to end:

- The evals workflow on GitHub. It needs the `OPENAI_API_KEY` repository
  secret.
- The self-hosted Langfuse overlay (`docker-compose.langfuse.yml`). Its
  config validates, but its images have not been started.
- The runbook followed from a fresh clone by someone new to the project.

**Done when** each has run once and any fixes are committed.
