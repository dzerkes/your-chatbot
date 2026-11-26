"""Split Markdown into searchable chunks that follow the document's structure.

Sections come from headings, so a chunk never mixes two sections. Inside a
section, paragraphs are packed up to a size limit; a paragraph that is too long
is split at sentence and then word boundaries. Consecutive chunks of the same
section share an overlap so a fact cut at the boundary is still found. Fenced
code blocks are kept whole when they fit, and headings inside them are ignored.

Sizes are in characters, not tokens: about four characters per token for
English, which keeps the default limit well under small embedding models'
512-token input.
"""

import re
from dataclasses import dataclass

# Bump when the output for the same input changes, so ingestion re-chunks.
CHUNKER_VERSION = "markdown-v1"
DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP_CHARS = 200

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_JOIN = "\n\n"


@dataclass(frozen=True)
class Chunk:
    position: int
    heading: str  # heading path, e.g. "Handbook > Leave > Sick leave"; "" before any heading
    text: str


@dataclass
class _Section:
    path: tuple[str, ...]
    lines: list[str]


def chunk_markdown(
    text: str,
    *,
    title: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Return chunks in document order. ``title`` prefixes every heading path."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if not 0 <= overlap_chars < max_chars // 2:
        raise ValueError("overlap_chars must be between 0 and half of max_chars")
    chunks: list[Chunk] = []
    for section in _sections(text):
        path = section.path
        if title and (not path or path[0] != title):
            path = (title, *path)
        heading = " > ".join(path)
        for body in _pack(_blocks(section.lines), max_chars, overlap_chars):
            chunks.append(Chunk(position=len(chunks), heading=heading, text=body))
    return chunks


def _sections(text: str) -> list[_Section]:
    sections = [_Section(path=(), lines=[])]
    stack: list[tuple[int, str]] = []
    in_fence = False
    for line in text.replace("\r\n", "\n").split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            level, name = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name))
            sections.append(_Section(path=tuple(n for _, n in stack), lines=[]))
        else:
            sections[-1].lines.append(line)
    return sections


def _blocks(lines: list[str]) -> list[str]:
    """Paragraphs separated by blank lines; a fenced code block is one block."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
        if not in_fence and not line.strip() and not _FENCE.match(line):
            if current:
                blocks.append("\n".join(current).strip("\n"))
                current = []
            continue
        current.append(line.rstrip())
    if current:
        blocks.append("\n".join(current).strip("\n"))
    return [block for block in blocks if block.strip()]


def _pack(blocks: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    # Long blocks are split small enough that the overlap still fits in front of them.
    limit = max_chars - (overlap_chars + len(_JOIN) if overlap_chars else 0)
    pieces = [piece for block in blocks for piece in _split_long(block, limit)]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{_JOIN}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        # A large block leaves less room; shrink the overlap rather than drop it.
        room = max_chars - len(piece) - len(_JOIN)
        overlap = _tail(current, min(overlap_chars, max(room, 0)))
        current = f"{overlap}{_JOIN}{piece}" if overlap else piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    parts: list[str] = []
    current = ""
    for unit in _units(block, max_chars):
        candidate = f"{current} {unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            parts.append(current)
            current = unit
    if current:
        parts.append(current)
    return parts


def _units(block: str, max_chars: int) -> list[str]:
    """Sentences, falling back to words, then to hard cuts for very long tokens."""
    units: list[str] = []
    for sentence in _SENTENCE_END.split(block.strip()):
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue
        for word in sentence.split():
            units.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
    return units


def _tail(text: str, size: int) -> str:
    """The last ``size`` characters, starting at a sentence or word boundary."""
    if size == 0 or len(text) <= size:
        return "" if size == 0 else text
    tail = text[-size:]
    sentence = _SENTENCE_END.search(tail)
    if sentence:
        return tail[sentence.end() :].strip()
    space = tail.find(" ")
    return tail[space + 1 :].strip() if space != -1 else tail.strip()
