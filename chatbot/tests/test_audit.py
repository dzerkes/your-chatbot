import pytest
from chatbot.audit import returned_sources
from chatbot.types import ToolResult

pytestmark = pytest.mark.unit


def test_document_audit_records_only_returned_source_identifiers() -> None:
    result = ToolResult(
        tool_use_id="s",
        content={
            "items": [
                {"source": "docs/a", "chunk_id": 2, "snippet": "Document contents"},
                {"source": "docs/a", "chunk_id": 5},
            ],
            "total": 20,
        },
    )
    assert returned_sources("search_docs", result) == [
        {"source": "docs/a", "chunk_id": 2},
        {"source": "docs/a", "chunk_id": 5},
    ]
    result.is_error = True
    assert returned_sources("search_docs", result) == []
