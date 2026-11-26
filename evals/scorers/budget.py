"""Cost and latency against the dataset's budgets. Raw numbers are in the report too."""

from evals.scorers.base import Score, Scored

NAME = "budget"


def budget(scored: Scored) -> Score | None:
    limits = scored.budgets
    if limits.max_cost_usd is None and limits.max_latency_seconds is None:
        return None
    observation = scored.observation
    over: list[str] = []
    if limits.max_cost_usd is not None and observation.cost_usd > limits.max_cost_usd:
        over.append(f"cost ${observation.cost_usd:.4f} > ${limits.max_cost_usd}")
    if (
        limits.max_latency_seconds is not None
        and observation.latency_seconds > limits.max_latency_seconds
    ):
        over.append(f"latency {observation.latency_seconds:.1f}s > {limits.max_latency_seconds}s")
    return Score(
        name=NAME,
        value=0.0 if over else 1.0,
        passed=not over,
        comment="; ".join(over) or "Within cost and latency budgets",
    )
