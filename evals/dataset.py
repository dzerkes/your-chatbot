"""Eval dataset schema, loaded from YAML files in evals/datasets/."""

from decimal import Decimal
from pathlib import Path
from typing import Annotated, Self

import yaml
from chatbot.types import DomainModel
from pydantic import Field, JsonValue, PositiveFloat, model_validator

DATASETS = Path(__file__).resolve().parent / "datasets"
Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=80)]


class EvalUser(DomainModel):
    display_name: str | None = None
    groups: list[str] = Field(default_factory=list)


class SourceRef(DomainModel):
    """A document heading path ("Title" or "Title > Section ...") or a project ID."""

    doc: str | None = Field(default=None, min_length=1)
    project_id: int | None = None

    @model_validator(mode="after")
    def exactly_one(self) -> Self:
        if (self.doc is None) == (self.project_id is None):
            raise ValueError("A source names either doc or project_id")
        return self

    def label(self) -> str:
        return self.doc if self.doc is not None else f"project {self.project_id}"


class ExpectedCall(DomainModel):
    name: str = Field(min_length=1)
    # Each listed argument must have this value; other arguments are free.
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class NoAccess(DomainModel):
    """The user must not learn this content: their groups do not include its documents."""

    forbidden_sources: list[SourceRef] = Field(default_factory=list)
    forbidden_text: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def something_forbidden(self) -> Self:
        if not self.forbidden_sources and not self.forbidden_text:
            raise ValueError("no_access needs forbidden_sources or forbidden_text")
        return self


class Injection(DomainModel):
    """A question whose answer comes from a passage that carries an injected instruction."""

    marker: str = Field(min_length=1)  # text of the injection, to confirm it was retrieved
    forbidden_text: list[str] = Field(default_factory=list)  # in the answer
    forbidden_tool_text: list[str] = Field(default_factory=list)  # in any tool arguments
    forbidden_behaviour: list[str] = Field(default_factory=list)  # judged by the model


class Budgets(DomainModel):
    max_cost_usd: Annotated[Decimal, Field(gt=0)] | None = None
    max_latency_seconds: PositiveFloat | None = None


class Case(DomainModel):
    id: Slug
    user: str
    question: str = Field(min_length=1)
    # None leaves tool choice unscored; [] expects an answer without tools.
    expected_tools: list[ExpectedCall] | None = None
    optional_tools: list[str] = Field(default_factory=list)  # allowed, not required
    expected_sources: list[SourceRef] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    no_access: NoAccess | None = None
    injection: Injection | None = None
    budgets: Budgets = Field(default_factory=Budgets)


class Dataset(DomainModel):
    name: Slug
    description: str = ""
    users: dict[str, EvalUser] = Field(min_length=1)
    budgets: Budgets = Field(default_factory=Budgets)
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        ids = [case.id for case in self.cases]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate case ids: {', '.join(duplicates)}")
        unknown = sorted({case.user for case in self.cases} - self.users.keys())
        if unknown:
            raise ValueError(f"Cases name undefined users: {', '.join(unknown)}")
        return self

    def budgets_for(self, case: Case) -> Budgets:
        return Budgets(
            max_cost_usd=case.budgets.max_cost_usd or self.budgets.max_cost_usd,
            max_latency_seconds=(
                case.budgets.max_latency_seconds or self.budgets.max_latency_seconds
            ),
        )


def load_dataset(path: Path) -> Dataset:
    return Dataset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
