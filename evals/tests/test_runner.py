"""The runner end to end with a scripted model, a fake tool and a scripted judge."""

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from chatbot.agent import Agent
from chatbot.config import Settings
from chatbot.errors import ProviderError
from chatbot.llm.base import LLMEvent, LLMStop, LLMTextDelta, LLMToolCall, LLMUsage
from chatbot.llm.fake import ScriptedProvider
from chatbot.store.memory import MemoryStore
from chatbot.tools.base import Tool, ToolContext, ToolInput
from chatbot.tools.pagination import ToolPage
from chatbot.tools.registry import ToolRegistry
from chatbot.types import ToolUse, Usage
from pydantic import JsonValue, SecretStr

from evals.dataset import Dataset
from evals.judge import GRADE_TOOL, Judge
from evals.langfuse import score_payloads, send_scores
from evals.report import markdown_report, text_report, write_reports
from evals.runner import evaluate, select_cases

pytestmark = pytest.mark.unit

DATASET = Dataset.model_validate(
    {
        "name": "unit",
        "users": {"casey": {"groups": ["employees"]}},
        "budgets": {"max_cost_usd": "0.05"},
        "cases": [
            {
                "id": "leave",
                "user": "casey",
                "question": "How much leave?",
                "expected_tools": [{"name": "search_docs"}],
                "expected_sources": [{"doc": "Employee handbook > Leave"}],
                "facts": ["26 days per year", "Five days carry over"],
            },
            {"id": "hello", "user": "casey", "question": "Hi", "expected_tools": []},
        ],
    }
)


class SearchInput(ToolInput):
    query: str


def registry(groups_seen: list[list[str]]) -> ToolRegistry:
    async def search(inputs: SearchInput, context: ToolContext) -> ToolPage:
        groups_seen.append(context.user.groups)
        item: JsonValue = {
            "title": "Employee handbook",
            "section": "Leave > Annual leave",
            "snippet": "26",
        }
        return ToolPage(items=[item], total=1)

    tools = ToolRegistry()
    tools.register(Tool("search_docs", "Search", SearchInput, search))
    return tools


def usage(cost: str) -> LLMUsage:
    return LLMUsage(usage=Usage(input_tokens=10, output_tokens=5), cost_usd=Decimal(cost))


def chatbot_script() -> list[list[LLMEvent | Exception]]:
    call = ToolUse(id="call-1", name="search_docs", arguments={"query": "annual leave"})
    return [
        [LLMToolCall(tool_use=call), usage("0.01"), LLMStop(reason="tool_use")],
        [LLMTextDelta(text="You get 26 days."), usage("0.01"), LLMStop(reason="end_turn")],
        [LLMTextDelta(text="Hello!"), usage("0.001"), LLMStop(reason="end_turn")],
    ]


def judge_script() -> list[list[LLMEvent | Exception]]:
    grade: dict[str, JsonValue] = {
        "facts": [
            {"fact": 1, "verdict": "supported", "reason": "states 26"},
            {"fact": 2, "verdict": "missing", "reason": "no carry-over"},
        ],
        "behaviours": [],
    }
    return [
        [
            LLMToolCall(tool_use=ToolUse(id="g", name=GRADE_TOOL, arguments=grade)),
            usage("0.002"),
            LLMStop(reason="tool_use"),
        ]
    ]


async def test_evaluate_runs_cases_as_their_users_and_scores_them(tmp_path: Path) -> None:
    groups: list[list[str]] = []
    judge_provider = ScriptedProvider(judge_script())
    agent = Agent(
        MemoryStore(), ScriptedProvider(chatbot_script()), registry(groups), Settings(),
        system="Test",
    )  # fmt: skip
    run = await evaluate(
        DATASET, agent, Judge(judge_provider), model="fake:model", judge_model="fake:judge"
    )
    assert [r.case_id for r in run.results] == ["leave", "hello"]
    assert groups == [["employees"]]
    leave, hello = run.results
    assert leave.observation.status == "completed"
    assert leave.observation.answer == "You get 26 days."
    assert leave.observation.cost_usd == Decimal("0.02")
    scores = {s.name: (s.value, s.passed) for s in leave.scores}
    assert scores == {
        "tool_choice": (1.0, True),
        "retrieval": (1.0, True),
        "answer_facts": (0.5, False),
        "budget": (1.0, True),
    }
    # The greeting needs no judge, so the judge was asked exactly once.
    assert hello.judgement is None and len(judge_provider.requests) == 1
    assert "You get 26 days." in judge_provider.requests[0].messages[0].content[0].text  # type: ignore[union-attr]
    summary = {s.name: s for s in run.summary.scorers}
    assert summary["tool_choice"].passed == 2 and summary["answer_facts"].mean == 0.5
    assert summary["refusal"].cases == 0 and summary["refusal"].mean is None
    assert run.summary.cost_usd == Decimal("0.021")
    assert run.summary.judge_cost_usd == Decimal("0.002")

    table = text_report(run)
    assert "answer_facts" in table and "leave answer_facts: missing" in table
    json_path, md_path = write_reports(run, tmp_path / "out")
    assert json.loads(json_path.read_text())["summary"]["completed"] == 2
    assert "| `leave` | casey |" in md_path.read_text()
    assert md_path.read_text() == markdown_report(run)


async def test_a_failing_turn_and_judge_are_recorded_not_raised() -> None:
    provider = ScriptedProvider([[ProviderError("down")]])
    judge = Judge(ScriptedProvider([[ProviderError("judge down")]]))
    agent = Agent(MemoryStore(), provider, registry([]), Settings(), system="Test")
    run = await evaluate(select_cases(DATASET, ["leave"]), agent, judge, model="fake:model")
    (result,) = run.results
    assert result.observation.status == "failed" and result.observation.error
    assert result.judgement is not None and "Judge call failed" in (result.judgement.error or "")
    assert all(not s.passed for s in result.scores if s.name != "budget")
    assert run.summary.completed == 0
    assert "Turns that did not complete" in text_report(run)


async def test_scores_are_sent_to_each_trace_with_stable_ids() -> None:
    agent = Agent(
        MemoryStore(), ScriptedProvider(chatbot_script()), registry([]), Settings(), system="T"
    )
    run = await evaluate(DATASET, agent, None, model="fake:model")
    for result in run.results:
        result.observation.trace_id = "a" * 32 if result.case_id == "leave" else None
    payloads = score_payloads(run)
    assert {p["traceId"] for p in payloads} == {"a" * 32}
    assert [p["name"] for p in payloads] == [
        "eval.tool_choice", "eval.retrieval", "eval.answer_facts", "eval.budget",
        "eval.latency_seconds",
    ]  # fmt: skip
    assert payloads == score_payloads(run)  # ids are deterministic
    assert len({p["id"] for p in payloads}) == len(payloads)
    sent: list[object] = []

    def accept(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/scores"
        sent.extend(json.loads(request.content))
        return httpx.Response(202, json={"message": "Accepted"})

    settings = Settings(langfuse_public_key=SecretStr("pk"), langfuse_secret_key=SecretStr("sk"))
    assert await send_scores(run, settings, transport=httpx.MockTransport(accept)) == 5
    assert sent == payloads


def test_select_cases_filters_by_id_substring() -> None:
    assert [c.id for c in select_cases(DATASET, ["hel"]).cases] == ["hello"]
    assert select_cases(DATASET, []) is DATASET
    with pytest.raises(SystemExit):
        select_cases(DATASET, ["nope"])
