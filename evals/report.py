"""Run results, their summary, and the printed, Markdown and JSON reports."""

import math
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from chatbot.types import DomainModel
from pydantic import AwareDatetime, Field

from evals.judge import Judgement
from evals.observe import Observation
from evals.scorers import SCORERS, Score


class CaseResult(DomainModel):
    case_id: str
    user: str
    question: str
    observation: Observation
    judgement: Judgement | None = None
    scores: list[Score] = Field(default_factory=list)


class ScorerSummary(DomainModel):
    name: str
    cases: int
    mean: float | None
    passed: int


class RunSummary(DomainModel):
    cases: int
    completed: int
    scorers: list[ScorerSummary]
    cost_usd: Decimal
    judge_cost_usd: Decimal
    mean_latency_seconds: float
    p95_latency_seconds: float
    mean_first_token_seconds: float | None


class RunResult(DomainModel):
    run_name: str
    dataset: str
    model: str
    judge_model: str | None
    started_at: AwareDatetime
    finished_at: AwareDatetime
    summary: RunSummary
    results: list[CaseResult]


def _percentile(values: list[float], share: float) -> float:
    """Nearest-rank percentile; 0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def summarize(results: list[CaseResult]) -> RunSummary:
    scorers: list[ScorerSummary] = []
    for name in SCORERS:
        scores = [s for r in results for s in r.scores if s.name == name]
        scorers.append(
            ScorerSummary(
                name=name,
                cases=len(scores),
                mean=sum(s.value for s in scores) / len(scores) if scores else None,
                passed=sum(s.passed for s in scores),
            )
        )
    latencies = [r.observation.latency_seconds for r in results]
    first = [
        r.observation.first_token_seconds
        for r in results
        if r.observation.first_token_seconds is not None
    ]
    return RunSummary(
        cases=len(results),
        completed=sum(r.observation.status == "completed" for r in results),
        scorers=scorers,
        cost_usd=sum((r.observation.cost_usd for r in results), Decimal("0")),
        judge_cost_usd=sum((r.judgement.cost_usd for r in results if r.judgement), Decimal("0")),
        mean_latency_seconds=sum(latencies) / len(latencies) if latencies else 0.0,
        p95_latency_seconds=_percentile(latencies, 0.95),
        mean_first_token_seconds=sum(first) / len(first) if first else None,
    )


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        return "  ".join(
            cell.ljust(width) if i == 0 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ).rstrip()

    return [line(header), line(["-" * w for w in widths]), *(line(row) for row in rows)]


def _scorer_rows(summary: RunSummary) -> list[list[str]]:
    return [
        [
            s.name,
            str(s.cases),
            "-" if s.mean is None else f"{s.mean:.2f}",
            f"{s.passed}/{s.cases}" if s.cases else "-",
        ]
        for s in summary.scorers
    ]


def _totals(run: RunResult) -> list[str]:
    s = run.summary
    per_case = s.cost_usd / s.cases if s.cases else Decimal("0")
    first = "-" if s.mean_first_token_seconds is None else f"{s.mean_first_token_seconds:.1f}s"
    return [
        f"Cases completed: {s.completed}/{s.cases}",
        f"Chatbot cost: ${s.cost_usd:.4f} total, ${per_case:.4f} per case",
        f"Judge cost: ${s.judge_cost_usd:.4f}",
        f"Latency: mean {s.mean_latency_seconds:.1f}s, p95 {s.p95_latency_seconds:.1f}s, "
        f"first token mean {first}",
    ]


def text_report(run: RunResult) -> str:
    lines = [
        f"Eval run {run.run_name}",
        f"Dataset {run.dataset}, model {run.model}, judge {run.judge_model or '-'}",
        "",
        *_table(["Scorer", "Cases", "Mean", "Passed"], _scorer_rows(run.summary)),
        "",
        *_totals(run),
    ]
    failures = [
        f"  {r.case_id} {s.name}: {s.comment}"
        for r in run.results
        for s in r.scores
        if not s.passed
    ]
    errors = [
        f"  {r.case_id}: {r.observation.error}"
        for r in run.results
        if r.observation.status != "completed"
    ]
    if errors:
        lines += ["", "Turns that did not complete:", *errors]
    if failures:
        lines += ["", "Failed checks:", *failures]
    return "\n".join(lines)


def _cell(text: str) -> str:
    return " ".join(text.replace("|", "\\|").split())


def markdown_report(run: RunResult) -> str:
    names = list(SCORERS)
    header = ["Case", "User", *names, "Cost", "Latency"]
    rows = []
    for r in run.results:
        by_name = {s.name: s for s in r.scores}
        rows.append(
            [
                f"`{r.case_id}`",
                r.user,
                *(
                    ""
                    if name not in by_name
                    else f"{by_name[name].value:.2f}" + ("" if by_name[name].passed else " ✗")
                    for name in names
                ),
                f"${r.observation.cost_usd:.4f}",
                f"{r.observation.latency_seconds:.1f}s",
            ]
        )
    lines = [
        f"# Eval run {run.run_name}",
        "",
        f"Dataset `{run.dataset}`, model `{run.model}`, judge `{run.judge_model or '-'}`. "
        f"Started {run.started_at:%Y-%m-%d %H:%M} UTC.",
        "",
        "| Scorer | Cases | Mean | Passed |",
        "|---|---:|---:|---:|",
        *(f"| {' | '.join(row)} |" for row in _scorer_rows(run.summary)),
        "",
        *(f"- {line}" for line in _totals(run)),
        "",
        "## Cases",
        "",
        f"| {' | '.join(header)} |",
        "|" + "---|" * len(header),
        *(f"| {' | '.join(row)} |" for row in rows),
    ]
    details = [
        f"- `{r.case_id}` {s.name}: {_cell(s.comment)}"
        for r in run.results
        for s in r.scores
        if not s.passed
    ]
    details += [
        f"- `{r.case_id}` did not complete: {_cell(r.observation.error or '')}"
        for r in run.results
        if r.observation.status != "completed"
    ]
    if details:
        lines += ["", "## Failed checks", "", *details]
    return "\n".join(lines) + "\n"


def write_reports(run: RunResult, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path, md_path = directory / "results.json", directory / "report.md"
    json_path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(run), encoding="utf-8")
    return json_path, md_path


def run_name(dataset: str, started: datetime) -> str:
    return f"{dataset}-{started:%Y%m%dT%H%M%SZ}"
