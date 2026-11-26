"""What a scorer returns, and the inputs every scorer receives."""

from collections.abc import Callable
from dataclasses import dataclass

from chatbot.types import DomainModel
from pydantic import Field

from evals.dataset import Budgets, Case
from evals.judge import Judgement
from evals.observe import Observation


class Score(DomainModel):
    name: str
    value: float = Field(ge=0, le=1)
    passed: bool
    comment: str = ""


@dataclass(frozen=True)
class Scored:
    case: Case
    observation: Observation
    judgement: Judgement | None
    budgets: Budgets


# A score, or None when the case does not exercise this scorer.
type Scorer = Callable[[Scored], Score | None]


def fraction(name: str, hits: int, total: int, comment: str) -> Score:
    return Score(name=name, value=hits / total, passed=hits == total, comment=comment)


def failed(name: str, comment: str) -> Score:
    return Score(name=name, value=0.0, passed=False, comment=comment)
