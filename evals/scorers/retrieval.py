"""Retrieval hit rate: the share of expected sources that the tools returned."""

from evals.scorers.base import Score, Scored, fraction

NAME = "retrieval"


def retrieval(scored: Scored) -> Score | None:
    expected = scored.case.expected_sources
    if not expected:
        return None
    missing = [s.label() for s in expected if not scored.observation.returned(s)]
    hits = len(expected) - len(missing)
    comment = (
        "All expected sources returned" if not missing else "Not returned: " + "; ".join(missing)
    )
    return fraction(NAME, hits, len(expected), comment)
