"""Scorers in report order. Each returns None for cases it does not apply to."""

from evals.scorers.answer_facts import answer_facts
from evals.scorers.base import Score, Scored, Scorer
from evals.scorers.budget import budget
from evals.scorers.injection import injection
from evals.scorers.refusal import refusal
from evals.scorers.retrieval import retrieval
from evals.scorers.tool_choice import tool_choice

SCORERS: dict[str, Scorer] = {
    "tool_choice": tool_choice,
    "retrieval": retrieval,
    "answer_facts": answer_facts,
    "refusal": refusal,
    "injection": injection,
    "budget": budget,
}


def score_case(scored: Scored) -> list[Score]:
    return [score for scorer in SCORERS.values() if (score := scorer(scored)) is not None]


__all__ = ["SCORERS", "Score", "Scored", "Scorer", "score_case"]
