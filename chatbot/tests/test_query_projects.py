"""Project filter validation does not require a database."""

import pytest
from chatbot.tools.query_projects import QueryProjectsInput
from pydantic import ValidationError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "arguments",
    [
        {"status": "unknown"},
        {"client_id": 0},
        {"text": "   "},
        {"client": ""},
        {"start_date_from": "2026-02-30"},
        {"start_date_from": "2026-12-01", "start_date_to": "2026-01-01"},
        {"sql": "SELECT * FROM raw.projects"},
        {"user": "admin"},
    ],
)
def test_invalid_project_filters(arguments: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        QueryProjectsInput.model_validate(arguments)


def test_project_filters_have_model_schema_and_normalized_text() -> None:
    inputs = QueryProjectsInput(client="  Northstar  ", text="  portal  ")
    assert inputs.client == "Northstar"
    assert inputs.text == "portal"
    schema = QueryProjectsInput.model_json_schema()
    assert schema["additionalProperties"] is False
    assert "date" in str(schema["properties"]["start_date_from"])
