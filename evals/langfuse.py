"""Send eval scores to Langfuse as scores on each case's trace."""

import logging
from uuid import NAMESPACE_URL, uuid5

import httpx
from chatbot.config import Settings
from pydantic import JsonValue

from evals.report import RunResult

logger = logging.getLogger(__name__)
BATCH = 50


def score_payloads(run: RunResult) -> list[dict[str, JsonValue]]:
    """One numeric score per scorer and case, named eval.<scorer>, plus latency."""
    payloads: list[dict[str, JsonValue]] = []
    timestamp = run.finished_at.isoformat()
    for result in run.results:
        trace_id = result.observation.trace_id
        if trace_id is None:
            continue
        metadata: dict[str, JsonValue] = {
            "eval_run": run.run_name,
            "dataset": run.dataset,
            "case": result.case_id,
        }
        values: list[tuple[str, float, str]] = [(s.name, s.value, s.comment) for s in result.scores]
        values.append(("latency_seconds", round(result.observation.latency_seconds, 3), ""))
        for name, value, comment in values:
            payloads.append(
                {
                    # Stable per run, case and scorer, so a resend updates instead of duplicating.
                    "id": str(uuid5(NAMESPACE_URL, f"{run.run_name}/{result.case_id}/{name}")),
                    "traceId": trace_id,
                    "name": f"eval.{name}",
                    "dataType": "NUMERIC",
                    "value": value,
                    "comment": comment[:2000],
                    "timestamp": timestamp,
                    "metadata": metadata,
                }
            )
    return payloads


async def send_scores(
    run: RunResult, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
) -> int:
    """Post the run's scores; returns how many Langfuse accepted."""
    payloads = score_payloads(run)
    accepted = 0
    async with httpx.AsyncClient(
        base_url=str(settings.langfuse_host),
        auth=(
            settings.langfuse_public_key.get_secret_value(),
            settings.langfuse_secret_key.get_secret_value(),
        ),
        timeout=10,
        transport=transport,
    ) as client:
        for start in range(0, len(payloads), BATCH):
            batch = payloads[start : start + BATCH]
            response = await client.post("api/public/scores", json=batch)
            response.raise_for_status()
            if response.status_code == 207:
                body = response.json()
                accepted += int(body.get("accepted", 0))
                logger.warning("Langfuse rejected some eval scores: %s", body.get("errors"))
            else:
                accepted += len(batch)
    return accepted
