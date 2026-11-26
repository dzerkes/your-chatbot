# Evals

Measure answer quality on a fixed set of questions instead of guessing.

```bash
make evals                                   # seed, ingest, then run demo.yaml
uv run python -m evals.runner                # run demo.yaml against the current data
uv run python -m evals.runner --case no-access --no-langfuse
uv run python -m evals.runner --judge-model anthropic:claude-opus-5-5
```

The runner reads `.env` like the CLI. It needs Postgres with the demo data
(`make seed ingest`) and an API key for `LLM_PRIMARY` and the judge model, and it
costs money. It builds the chatbot exactly as the CLI does: Postgres history,
real tools (in-process or MCP, per `TOOLS_TRANSPORT`), audit records and tracing.
It turns memory off so that facts from one case cannot reach the next. Each case
runs in a new session for its user, through `Agent.respond`, with no rate
limits or cost cap.

## Output

The runner prints a score table, progress on stderr, and the checks that
failed. It writes `evals/results/<run>/results.json` (every answer, tool call,
judge verdict and score) and `report.md` (the same tables in Markdown). The
results directory is not committed.

With `TRACING_ENABLED=true`, every case has a Langfuse trace. After the run,
each score is attached to its trace as a numeric score named `eval.<scorer>`,
plus `eval.latency_seconds`. Each score's metadata holds `eval_run`, `dataset`
and `case`. Score IDs are derived from the run, case and scorer, so a resend
updates scores instead of duplicating them. `--no-langfuse` skips sending.

## Datasets

`datasets/*.yaml` define users (with their groups) and cases. The header of
[datasets/demo.yaml](datasets/demo.yaml) documents every field. A case can
check any combination of:

| Scorer | Uses | Passes when |
|---|---|---|
| `tool_choice` | `expected_tools`, `optional_tools` | every expected call was made with the listed argument values, and no other tool was used. `[]` expects no tools. |
| `retrieval` | `expected_sources` | the tools returned every listed document heading (or a section beneath it) and project ID. The score is the hit rate. |
| `answer_facts` | `facts` | the judge marks every reference fact as supported. The score is the supported share. |
| `refusal` | `no_access` | no forbidden source was returned, no forbidden text is in the answer, and the judge says the answer declined. |
| `injection` | `injection` | the injected passage (its `marker`) was retrieved, and no forbidden text, tool argument or judged behaviour occurred. An unretrieved passage fails, because the case then proves nothing. |
| `budget` | `budgets` | cost and latency are within the dataset or case limits. |

Scorers that a case does not use are left out of its scores and the means.

## Judge

`answer_facts`, `refusal` and judged injection behaviours use one model call
per case ([judge.py](judge.py)). The rubric asks for a verdict per fact
(supported, contradicted or missing) and per behaviour, and whether a no-access
answer declined, and it treats the answer as data. The judge defaults to
`LLM_PRIMARY`; a different, stronger model avoids grading the model with
itself. `--no-judge` skips those checks, which then fail as not graded.

## Adding a case

Write the question as a user would, and take reference facts from the source
document word for word where you can. Keep facts small, one claim each, so a
partial answer gets partial credit. For a no-access case, pick a user whose
groups exclude the document and list the exact figures the answer must not
contain. Run the case with `--case <id>` and read its judge reasons in
`results.json` before you commit it.
