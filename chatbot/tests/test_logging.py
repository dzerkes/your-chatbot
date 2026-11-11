import asyncio
import json
import logging
from collections.abc import Iterator
from io import StringIO

import pytest
from chatbot.config import Settings
from chatbot.logging import configure_logging, log_context

pytestmark = pytest.mark.unit


@pytest.fixture
def output() -> Iterator[StringIO]:
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    stream = StringIO()
    try:
        yield stream
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            if handler not in previous_handlers:
                handler.close()
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)


def test_json_logging_restores_nested_context_after_error(output: StringIO) -> None:
    configure_logging(Settings(log_format="json"), stream=output)
    logger = logging.getLogger("chatbot.test")
    logger.info("outside")
    with log_context(session_id="session-1", turn_id="turn-1", trace_id="trace-1"):
        logger.info("outer")
        with pytest.raises(RuntimeError), log_context(turn_id="turn-2"):
            logger.info("inner")
            raise RuntimeError("test")
        logger.info("restored")
    logger.info("finished")
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[0]["session_id"] is None
    assert records[1]["trace_id"] == "trace-1"
    assert records[2]["turn_id"] == "turn-2"
    assert records[2]["session_id"] == "session-1"
    assert records[3]["turn_id"] == "turn-1"
    assert records[4]["turn_id"] is None


async def test_concurrent_turns_keep_separate_context(output: StringIO) -> None:
    configure_logging(Settings(log_format="json"), stream=output)
    logger = logging.getLogger("chatbot.test")

    async def turn(name: str) -> None:
        with log_context(session_id=name, turn_id=name, trace_id=name):
            await asyncio.sleep(0)
            logger.info("%s", name)

    await asyncio.gather(turn("first"), turn("second"))
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert {record["message"] for record in records} == {"first", "second"}
    for record in records:
        assert record["session_id"] == record["turn_id"] == record["trace_id"] == record["message"]


def test_reconfigure_and_log_level(output: StringIO) -> None:
    settings = Settings(log_format="console", log_level="WARNING")
    configure_logging(settings, stream=output)
    configure_logging(settings, stream=output)
    logger = logging.getLogger("chatbot.test")
    logger.info("hidden")
    with log_context(session_id="session-1", turn_id="turn-1", trace_id="trace-1"):
        logger.warning("visible")
    assert output.getvalue().count("visible") == 1
    assert "hidden" not in output.getvalue()
    assert "session_id=session-1 turn_id=turn-1 trace_id=trace-1" in output.getvalue()


def test_json_exception_is_one_line(output: StringIO) -> None:
    configure_logging(Settings(log_format="json"), stream=output)
    try:
        raise RuntimeError("test failure")
    except RuntimeError:
        logging.getLogger("chatbot.test").exception("request failed\nwith details")
    records = output.getvalue().splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["message"] == "request failed\nwith details"
    assert "RuntimeError: test failure" in record["exception"]
