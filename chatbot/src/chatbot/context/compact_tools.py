"""Replace tool results from earlier turns with a one-line note.

Tool results are the bulk of a long conversation: a search returns kilobytes,
and the assistant's answer already carries what mattered from it. Results from
the turn in progress stay whole, because the model is still reading them.

Every earlier turn is compacted, not only the old ones. A turn's compacted
form never changes once it enters history, so the prompt prefix stays the same
from one request to the next and provider prompt caching keeps working. A
rule like "keep the last N turns whole" would rewrite a turn each time it
crossed the line and invalidate the cache from that point on.
"""

import json

from chatbot.types import Message, ToolResult, ToolUse


def compact_note(tool: str, content: object) -> str:
    size = len(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    shown = ""
    if isinstance(content, dict) and isinstance(content.get("items"), list):
        total = content.get("total")
        shown = f" {len(content['items'])} of {total} items," if isinstance(total, int) else ""
    return (
        f"[Earlier {tool} result omitted to save context:{shown} {size / 1024:.1f} KB. "
        f"Call {tool} again if you need its details.]"
    )


def compact_turn(turn: list[Message]) -> list[Message]:
    """A copy of one stored turn with each successful tool result replaced by a note."""
    names: dict[str, str] = {}
    compacted: list[Message] = []
    for message in turn:
        for block in message.content:
            if isinstance(block, ToolUse):
                names[block.id] = block.name
        if message.role != "tool":
            compacted.append(message)
            continue
        content = [
            block.model_copy(
                update={
                    "content": compact_note(names.get(block.tool_use_id, "tool"), block.content)
                }
            )
            if isinstance(block, ToolResult) and not block.is_error
            else block
            for block in message.content
        ]
        compacted.append(message.model_copy(update={"content": content}))
    return compacted
