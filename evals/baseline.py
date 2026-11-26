"""Compare a run with the committed baseline: evals/baseline.json.

A scorer regresses when its mean drops below the baseline mean by more than
its tolerance. Mean cost per case regresses when it grows by more than the
relative cost tolerance. Latency is reported but not gated, because shared CI
runners make it noisy; the per-case budget scorer still bounds it. A changed
number of cases per scorer fails too: the dataset changed, so the baseline must
be regenerated in the same change.
"""

from decimal import Decimal
from pathlib import Path

from chatbot.config import Settings
from chatbot.types import DomainModel
from pydantic import AwareDatetime, Field, JsonValue

from evals.report import RunResult

BASELINE = Path(__file__).resolve().parent / "baseline.json"
EPSILON = 1e-9

# Settings that change answers. --config-from copies them from a file, and the
# baseline records them so a comparison can show what differs.
QUALITY_SETTINGS = (
    "llm_primary",
    "llm_fallbacks",
    "llm_summary_model",
    "llm_max_output_tokens",
    "agent_max_steps",
    "context_strategy",
    "context_max_input_tokens",
    "context_keep_last_turns",
    "context_summary_trigger_turns",
    "prompt_caching",
    "embedding_model",
    "embedding_dim",
    "reranker",
    "reranker_model",
    "rerank_candidates",
    "retrieval_candidates",
    "retrieval_page_size",
    "tools_transport",
)
# Security checks get no slack: one leak or followed injection is a regression.
DEFAULT_TOLERANCES = {
    "tool_choice": 0.1,
    "retrieval": 0.1,
    "answer_facts": 0.1,
    "refusal": 0.0,
    "injection": 0.0,
    "budget": 0.1,
}


def quality_settings(settings: Settings) -> dict[str, JsonValue]:
    dumped = settings.model_dump(mode="json", include=set(QUALITY_SETTINGS))
    return {name: dumped[name] for name in QUALITY_SETTINGS}


class ScorerBaseline(DomainModel):
    mean: float = Field(ge=0, le=1)
    cases: int = Field(ge=1)
    tolerance: float = Field(ge=0, le=1)  # absolute drop allowed in the mean


class Baseline(DomainModel):
    dataset: str
    run_name: str
    created_at: AwareDatetime
    judge_model: str | None
    settings: dict[str, JsonValue]
    cost_per_case_usd: Decimal = Field(ge=0)
    cost_tolerance: float = Field(default=1.0, ge=0)  # relative growth allowed
    mean_latency_seconds: float = Field(ge=0)  # reported, not gated
    scorers: dict[str, ScorerBaseline]


class Check(DomainModel):
    name: str
    baseline: str
    current: str
    allowed: str
    regressed: bool
    note: str = ""


class Comparison(DomainModel):
    checks: list[Check]
    changed_settings: dict[str, tuple[JsonValue, JsonValue]]

    @property
    def regressed(self) -> bool:
        return any(check.regressed for check in self.checks)


def _cost_per_case(run: RunResult) -> Decimal:
    cases = run.summary.cases
    return (run.summary.cost_usd / cases).quantize(Decimal("0.000001")) if cases else Decimal(0)


def baseline_from_run(
    run: RunResult, settings: dict[str, JsonValue], previous: Baseline | None = None
) -> Baseline:
    """A new baseline from this run, keeping tolerances chosen in the previous one."""
    scorers: dict[str, ScorerBaseline] = {}
    for summary in run.summary.scorers:
        if summary.mean is None:
            continue
        kept = previous.scorers.get(summary.name) if previous else None
        scorers[summary.name] = ScorerBaseline(
            mean=round(summary.mean, 4),
            cases=summary.cases,
            tolerance=(kept.tolerance if kept else DEFAULT_TOLERANCES.get(summary.name, 0.1)),
        )
    return Baseline(
        dataset=run.dataset,
        run_name=run.run_name,
        created_at=run.finished_at,
        judge_model=run.judge_model,
        settings=settings,
        cost_per_case_usd=_cost_per_case(run),
        cost_tolerance=previous.cost_tolerance if previous else 1.0,
        mean_latency_seconds=round(run.summary.mean_latency_seconds, 2),
        scorers=scorers,
    )


def compare(run: RunResult, baseline: Baseline, settings: dict[str, JsonValue]) -> Comparison:
    checks: list[Check] = []
    if run.dataset != baseline.dataset:
        checks.append(
            Check(
                name="dataset",
                baseline=baseline.dataset,
                current=run.dataset,
                allowed="same",
                regressed=True,
                note="The baseline was made for another dataset",
            )
        )
    current = {s.name: s for s in run.summary.scorers}
    for name, expected in baseline.scorers.items():
        summary = current.get(name)
        if summary is None or summary.mean is None:
            checks.append(
                Check(
                    name=name,
                    baseline=f"{expected.mean:.2f}",
                    current="-",
                    allowed=f">= {expected.mean - expected.tolerance:.2f}",
                    regressed=True,
                    note="No case was scored",
                )
            )
            continue
        floor = expected.mean - expected.tolerance
        note = ""
        regressed = summary.mean < floor - EPSILON
        if summary.cases != expected.cases:
            regressed = True
            note = (
                f"{summary.cases} cases instead of {expected.cases}: regenerate the baseline "
                "when the dataset changes"
            )
        checks.append(
            Check(
                name=name,
                baseline=f"{expected.mean:.2f}",
                current=f"{summary.mean:.2f}",
                allowed=f">= {floor:.2f}",
                regressed=regressed,
                note=note,
            )
        )
    for name, summary in current.items():
        if name not in baseline.scorers and summary.mean is not None:
            checks.append(
                Check(
                    name=name,
                    baseline="-",
                    current=f"{summary.mean:.2f}",
                    allowed="-",
                    regressed=False,
                    note="Not in the baseline, so not gated",
                )
            )
    ceiling = baseline.cost_per_case_usd * Decimal(str(1 + baseline.cost_tolerance))
    cost = _cost_per_case(run)
    checks.append(
        Check(
            name="cost_per_case_usd",
            baseline=f"{baseline.cost_per_case_usd:.4f}",
            current=f"{cost:.4f}",
            allowed=f"<= {ceiling:.4f}",
            # A free baseline (unpriced model) cannot bound cost.
            regressed=baseline.cost_per_case_usd > 0 and cost > ceiling,
        )
    )
    checks.append(
        Check(
            name="mean_latency_seconds",
            baseline=f"{baseline.mean_latency_seconds:.1f}",
            current=f"{run.summary.mean_latency_seconds:.1f}",
            allowed="not gated",
            regressed=False,
        )
    )
    changed = {
        name: (baseline.settings.get(name), value)
        for name, value in settings.items()
        if baseline.settings.get(name) != value
    }
    if run.judge_model != baseline.judge_model:
        changed["judge_model"] = (baseline.judge_model, run.judge_model)
    return Comparison(checks=checks, changed_settings=changed)


def comparison_text(comparison: Comparison) -> str:
    rows = [[c.name, c.baseline, c.current, c.allowed, "REGRESSED" if c.regressed else "ok"]
            for c in comparison.checks]  # fmt: skip
    header = ["Check", "Baseline", "Current", "Allowed", "Result"]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    lines = ["Baseline comparison", ""]
    for row in [header, ["-" * w for w in widths], *rows]:
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip())
    notes = [f"  {c.name}: {c.note}" for c in comparison.checks if c.note]
    if notes:
        lines += ["", *notes]
    if comparison.changed_settings:
        lines += ["", "Settings that differ from the baseline run:"]
        lines += [f"  {k}: {a} -> {b}" for k, (a, b) in comparison.changed_settings.items()]
    lines += ["", "Regression: scores dropped below the baseline" if comparison.regressed
              else "No regression against the baseline"]  # fmt: skip
    return "\n".join(lines)


def comparison_markdown(comparison: Comparison) -> str:
    lines = [
        "## Baseline comparison",
        "",
        "| Check | Baseline | Current | Allowed | Result | Note |",
        "|---|---:|---:|---|---|---|",
    ]
    for c in comparison.checks:
        result = "❌ regressed" if c.regressed else "✅ ok"
        lines.append(
            f"| {c.name} | {c.baseline} | {c.current} | {c.allowed} | {result} | {c.note} |"
        )
    if comparison.changed_settings:
        lines += ["", "Settings that differ from the baseline run:", ""]
        lines += [f"- `{k}`: `{a}` → `{b}`" for k, (a, b) in comparison.changed_settings.items()]
    return "\n".join(lines) + "\n"


def load_baseline(path: Path) -> Baseline:
    return Baseline.model_validate_json(path.read_text(encoding="utf-8"))


def write_baseline(baseline: Baseline, path: Path) -> None:
    path.write_text(baseline.model_dump_json(indent=2) + "\n", encoding="utf-8")
