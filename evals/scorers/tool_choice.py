"""Did the agent call the expected tools, with the expected arguments, and nothing else?"""

from evals.dataset import ExpectedCall
from evals.observe import ToolCallRecord
from evals.scorers.base import Score, Scored, failed, fraction

NAME = "tool_choice"


def _matches(expected: ExpectedCall, call: ToolCallRecord) -> bool:
    return call.name == expected.name and all(
        call.arguments.get(key) == value for key, value in expected.arguments.items()
    )


def tool_choice(scored: Scored) -> Score | None:
    expected = scored.case.expected_tools
    if expected is None:
        return None
    calls = scored.observation.tool_calls
    names = [call.name for call in calls]
    allowed = {call.name for call in expected} | set(scored.case.optional_tools)
    if not expected:
        if set(names) - allowed:
            return failed(NAME, f"Expected no tools, called {', '.join(names)}")
        return Score(
            name=NAME, value=1.0, passed=True, comment="No required tools, none unexpected"
        )
    unexpected = sorted(set(names) - allowed)
    if unexpected:
        return failed(NAME, f"Called unexpected tools: {', '.join(unexpected)}")
    missing = [e for e in expected if not any(_matches(e, call) for call in calls)]
    hits = len(expected) - len(missing)
    comment = (
        "All expected calls made"
        if not missing
        else "Missing: "
        + "; ".join(f"{e.name}({e.arguments})" if e.arguments else e.name for e in missing)
    )
    return fraction(NAME, hits, len(expected), comment)
