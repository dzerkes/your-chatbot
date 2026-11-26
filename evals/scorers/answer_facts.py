"""Answer facts, graded by the judge model: the share of reference facts the answer states."""

from evals.scorers.base import Score, Scored, failed

NAME = "answer_facts"


def answer_facts(scored: Scored) -> Score | None:
    facts = scored.case.facts
    if not facts:
        return None
    judgement = scored.judgement
    if judgement is None or judgement.grade is None:
        return failed(NAME, judgement.error if judgement and judgement.error else "Not graded")
    verdicts = {v.fact: v for v in judgement.grade.facts}
    supported = 0
    notes: list[str] = []
    for number, fact in enumerate(facts, 1):
        verdict = verdicts.get(number)
        if verdict is not None and verdict.verdict == "supported":
            supported += 1
        else:
            label = verdict.verdict if verdict else "not graded"
            reason = f": {verdict.reason}" if verdict and verdict.reason else ""
            notes.append(f"{label} '{fact}'{reason}")
    return Score(
        name=NAME,
        value=supported / len(facts),
        passed=supported == len(facts),
        comment="All facts supported" if not notes else "; ".join(notes),
    )
