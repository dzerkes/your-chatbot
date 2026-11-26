"""Baseline creation and the regression gate."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import JsonValue

from evals.baseline import (
    BASELINE,
    DEFAULT_TOLERANCES,
    QUALITY_SETTINGS,
    Baseline,
    baseline_from_run,
    compare,
    comparison_markdown,
    comparison_text,
    load_baseline,
    write_baseline,
)
from evals.dataset import DATASETS, load_dataset
from evals.observe import Observation
from evals.report import CaseResult, RunResult, summarize
from evals.scorers import Score

pytestmark = pytest.mark.unit
SETTINGS: dict[str, JsonValue] = {"llm_primary": "anthropic:a", "reranker": "local"}


def run_with(
    values: dict[str, list[float]], cost: str = "0.01", dataset: str = "demo"
) -> RunResult:
    """One case per value; every scorer gets the same number of cases."""
    count = max(len(v) for v in values.values())
    results = []
    for i in range(count):
        scores = [
            Score(name=name, value=v[i], passed=v[i] == 1)
            for name, v in values.items()
            if i < len(v)
        ]
        results.append(
            CaseResult(
                case_id=f"c{i}",
                user="u",
                question="Q",
                observation=Observation(
                    status="completed", cost_usd=Decimal(cost), latency_seconds=2
                ),
                scores=scores,
            )
        )
    now = datetime.now(UTC)
    return RunResult(
        run_name=f"{dataset}-run",
        dataset=dataset,
        model="anthropic:a",
        judge_model="anthropic:a",
        started_at=now,
        finished_at=now,
        summary=summarize(results),
        results=results,
    )


GOOD = {"tool_choice": [1.0] * 10, "answer_facts": [1.0] * 10, "refusal": [1.0] * 4}


def regressed(comparison: object) -> set[str]:
    return {c.name for c in comparison.checks if c.regressed}  # type: ignore[attr-defined]


def test_baseline_records_means_counts_cost_and_default_tolerances() -> None:
    baseline = baseline_from_run(run_with(GOOD), SETTINGS)
    assert set(baseline.scorers) == set(GOOD)  # scorers without cases are left out
    assert baseline.scorers["tool_choice"].cases == 10
    assert baseline.scorers["refusal"].tolerance == DEFAULT_TOLERANCES["refusal"] == 0
    assert baseline.cost_per_case_usd == Decimal("0.01")
    assert baseline.settings == SETTINGS


def test_regenerating_keeps_hand_tuned_tolerances() -> None:
    first = baseline_from_run(run_with(GOOD), SETTINGS)
    first.scorers["answer_facts"].tolerance = 0.2
    first.cost_tolerance = 1.0
    second = baseline_from_run(run_with(GOOD), SETTINGS, previous=first)
    assert second.scorers["answer_facts"].tolerance == 0.2 and second.cost_tolerance == 1.0


def test_small_drops_pass_and_larger_drops_regress() -> None:
    baseline = baseline_from_run(run_with(GOOD), SETTINGS)
    one_miss = {**GOOD, "tool_choice": [0.0] + [1.0] * 9}  # 0.90, exactly at the floor
    assert not compare(run_with(one_miss), baseline, SETTINGS).regressed
    two_misses = {**GOOD, "tool_choice": [0.0, 0.0] + [1.0] * 8}
    assert regressed(compare(run_with(two_misses), baseline, SETTINGS)) == {"tool_choice"}


def test_security_scorers_allow_no_drop() -> None:
    baseline = baseline_from_run(run_with(GOOD), SETTINGS)
    leak = {**GOOD, "refusal": [0.0, 1.0, 1.0, 1.0]}
    comparison = compare(run_with(leak), baseline, SETTINGS)
    assert regressed(comparison) == {"refusal"}
    assert "REGRESSED" in comparison_text(comparison)
    assert "❌ regressed" in comparison_markdown(comparison)


def test_changed_case_counts_missing_scorers_and_dataset_regress() -> None:
    baseline = baseline_from_run(run_with(GOOD), SETTINGS)
    fewer = {**GOOD, "refusal": [1.0] * 3}
    comparison = compare(run_with(fewer), baseline, SETTINGS)
    assert regressed(comparison) == {"refusal"}
    assert "regenerate the baseline" in comparison_text(comparison)
    without = {k: v for k, v in GOOD.items() if k != "refusal"}
    assert regressed(compare(run_with(without), baseline, SETTINGS)) == {"refusal"}
    assert "dataset" in regressed(compare(run_with(GOOD, dataset="other"), baseline, SETTINGS))


def test_new_scorers_are_reported_but_not_gated() -> None:
    baseline = baseline_from_run(run_with(GOOD), SETTINGS)
    comparison = compare(run_with({**GOOD, "budget": [0.0] * 10}), baseline, SETTINGS)
    assert not comparison.regressed
    assert any(c.name == "budget" and "not gated" in c.note for c in comparison.checks)


def test_cost_growth_beyond_tolerance_regresses_and_settings_diff_is_shown() -> None:
    baseline = baseline_from_run(run_with(GOOD, cost="0.010"), SETTINGS)
    assert not compare(run_with(GOOD, cost="0.019"), baseline, SETTINGS).regressed
    changed = {**SETTINGS, "llm_primary": "openai:b"}
    comparison = compare(run_with(GOOD, cost="0.021"), baseline, changed)
    assert regressed(comparison) == {"cost_per_case_usd"}
    assert comparison.changed_settings == {"llm_primary": ("anthropic:a", "openai:b")}
    assert "anthropic:a -> openai:b" in comparison_text(comparison)


def test_baseline_round_trips_through_json(tmp_path: Path) -> None:
    baseline = baseline_from_run(run_with(GOOD), SETTINGS)
    write_baseline(baseline, tmp_path / "baseline.json")
    assert load_baseline(tmp_path / "baseline.json") == baseline


def test_committed_baseline_matches_the_demo_dataset() -> None:
    baseline = Baseline.model_validate_json(BASELINE.read_text())
    dataset = load_dataset(DATASETS / "demo.yaml")
    assert baseline.dataset == dataset.name
    assert baseline.scorers["tool_choice"].cases == sum(
        c.expected_tools is not None for c in dataset.cases
    )
    assert baseline.scorers["refusal"].cases == sum(c.no_access is not None for c in dataset.cases)
    assert set(baseline.settings) == set(QUALITY_SETTINGS)
