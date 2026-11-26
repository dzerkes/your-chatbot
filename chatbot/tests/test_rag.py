"""Chunking, document parsing, hashing and embedders; no database or network."""

import itertools
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from chatbot.config import Settings
from chatbot.errors import IngestError, ProviderError
from chatbot.rag.chunking import chunk_markdown
from chatbot.rag.embeddings import (
    FakeEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    build_embedder,
)
from chatbot.rag.ingest import content_hash, load_documents, parse_document
from openai import AsyncOpenAI
from pydantic import SecretStr

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "db/seed/documents"


def doc(front: str, body: str = "# Title\n\nSome text.") -> str:
    return f"---\n{front}\n---\n{body}"


# --- chunking ---------------------------------------------------------------


def test_sections_follow_headings_and_carry_the_title() -> None:
    text = (
        "Intro before any heading.\n\n"
        "# Guide\n\nTop text.\n\n"
        "## Leave\n\n"
        "### Sick leave\n\nCall your manager.\n\n"
        "## Expenses\n\nKeep receipts."
    )
    chunks = chunk_markdown(text, title="Guide")
    assert [(c.position, c.heading, c.text) for c in chunks] == [
        (0, "Guide", "Intro before any heading."),
        (1, "Guide", "Top text."),
        # "Leave" has no text of its own, so only its subsection becomes a chunk.
        (2, "Guide > Leave > Sick leave", "Call your manager."),
        (3, "Guide > Expenses", "Keep receipts."),
    ]
    assert chunk_markdown("## A\n\ntext")[0].heading == "A"


def test_code_fences_stay_whole_and_hide_headings() -> None:
    fence = "```bash\n# not a heading\n\necho hi\n```"
    chunks = chunk_markdown(f"## Deploy\n\nRun this:\n\n{fence}\n\nDone.")
    assert len(chunks) == 1
    assert chunks[0].heading == "Deploy"
    assert fence in chunks[0].text


def test_long_sections_split_with_overlap_inside_the_section_only() -> None:
    sentences = [f"Sentence number {i} explains a separate detail." for i in range(60)]
    text = "## Long\n\n" + " ".join(sentences) + "\n\n## Next\n\nShort section."
    chunks = chunk_markdown(text, max_chars=400, overlap_chars=100)
    long_chunks = [c for c in chunks if c.heading == "Long"]
    assert len(long_chunks) > 3
    assert all(len(c.text) <= 400 for c in chunks)
    for previous, current in itertools.pairwise(long_chunks):
        # The next chunk starts with whole sentences repeated from the previous one.
        first_sentence = current.text.split(". ")[0]
        assert first_sentence in previous.text
    # Every sentence survives, and the next section starts fresh.
    joined = " ".join(c.text for c in long_chunks)
    assert all(sentence in joined for sentence in sentences)
    assert chunks[-1].text == "Short section."
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_unbreakable_text_is_cut_to_the_limit() -> None:
    chunks = chunk_markdown("x" * 1000, max_chars=300, overlap_chars=0)
    assert [len(c.text) for c in chunks] == [300, 300, 300, 100]


def test_chunking_rejects_bad_sizes_and_skips_empty_text() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_markdown("text", max_chars=50)
    with pytest.raises(ValueError, match="overlap_chars"):
        chunk_markdown("text", max_chars=200, overlap_chars=150)
    assert chunk_markdown("\r\n# Only a heading\r\n\r\n") == []


# --- documents --------------------------------------------------------------


def test_front_matter_becomes_title_groups_and_metadata() -> None:
    parsed = parse_document(
        "a/b.md",
        doc("title: Pricing\ngroups: Finance, leadership , finance\ntags: x, y\nowner: Ops"),
    )
    assert parsed.title == "Pricing"
    assert parsed.groups == ("finance", "leadership")
    assert parsed.metadata == {"owner": "Ops", "tags": ["x", "y"]}
    assert parsed.body.startswith("# Title")


def test_title_falls_back_to_first_heading_then_file_name() -> None:
    assert parse_document("x.md", doc("groups: a")).title == "Title"
    assert parse_document("dir/plain-name.md", doc("groups: a", "No heading.")).title == (
        "plain-name"
    )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("# No front matter", "missing front matter"),
        ("---\ngroups: a\n# never closed", "not closed"),
        (doc("title: No groups"), "at least one group"),
        (doc("groups: , ,"), "at least one group"),
        (doc("groups: Bad Group!"), "invalid group"),
        (doc("groups: a\ngroups: b"), "duplicate key"),
        (doc("groups: a\nnot a pair"), "expected 'key: value'"),
        (doc("groups: a", "   "), "no text"),
    ],
)
def test_invalid_documents_are_rejected(raw: str, message: str) -> None:
    with pytest.raises(IngestError, match=message):
        parse_document("doc.md", raw)


def test_load_documents_uses_relative_sources_and_skips_other_files(tmp_path: Path) -> None:
    (tmp_path / "team").mkdir()
    (tmp_path / "team/plan.md").write_text(doc("groups: team"))
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden/secret.md").write_text(doc("groups: team"))
    assert [d.source for d in load_documents(tmp_path)] == ["team/plan.md"]
    (tmp_path / "empty").mkdir()
    with pytest.raises(IngestError, match="No Markdown"):
        load_documents(tmp_path / "empty")
    with pytest.raises(IngestError, match="not a directory"):
        load_documents(tmp_path / "missing")


def test_seed_documents_parse_and_include_restricted_ones() -> None:
    documents = {d.source: d for d in load_documents(SEED)}
    assert len(documents) == 10  # 7 regular, 3 adversarial (C21)
    assert documents["handbook/employee-handbook.md"].groups == ("employees",)
    assert documents["finance/pricing-guidelines-2026.md"].groups == ("finance", "leadership")
    assert documents["hr/salary-bands.md"].groups == ("hr", "leadership")
    for document in documents.values():
        chunks = chunk_markdown(document.body, title=document.title)
        assert chunks and all(c.heading.startswith(document.title) for c in chunks)


def test_content_hash_covers_everything_that_changes_the_index() -> None:
    base = parse_document("d.md", doc("groups: a\ntags: t"))
    embedder = FakeEmbedder(8)
    digest = content_hash(base, embedder)
    assert digest == content_hash(parse_document("d.md", doc("groups: a\ntags: t")), embedder)
    changed = [
        parse_document("d.md", doc("groups: a, b\ntags: t")),
        parse_document("d.md", doc("groups: a\ntags: u")),
        parse_document("d.md", doc("groups: a\ntags: t", "# Title\n\nOther text.")),
    ]
    assert all(content_hash(d, embedder) != digest for d in changed)
    assert content_hash(base, FakeEmbedder(8, "other")) != digest
    assert content_hash(base, FakeEmbedder(16)) != digest


# --- embedders --------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_fake_embedder_is_deterministic_normalized_and_lexical() -> None:
    embedder = FakeEmbedder(64)
    first, second, other = await embedder.embed_documents(
        ["annual leave policy", "annual leave rules", "deployment rollback"]
    )
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0)
    assert first == (await embedder.embed_documents(["annual leave policy"]))[0]
    query = await embedder.embed_query("annual leave")
    assert cosine(query, first) > cosine(query, other)
    assert cosine(first, second) > cosine(first, other)
    assert len(await embedder.embed_query("")) == 64  # no words still gives a unit vector


def test_build_embedder_selects_by_provider_prefix() -> None:
    local = build_embedder(Settings(embedding_model="local:BAAI/bge-small-en-v1.5"))
    assert isinstance(local, LocalEmbedder)
    assert (local.name, local.dim) == ("local:BAAI/bge-small-en-v1.5", 384)
    fake = build_embedder(Settings(embedding_model="fake:hash", embedding_dim=12))
    assert (fake.name, fake.dim) == ("fake:hash", 12)
    openai = build_embedder(
        Settings(embedding_model="openai:text-embedding-3-small", openai_api_key=SecretStr("k"))
    )
    assert isinstance(openai, OpenAIEmbedder)
    with pytest.raises(ProviderError, match="API key"):
        build_embedder(Settings(embedding_model="openai:text-embedding-3-small"))
    with pytest.raises(ProviderError, match="Unknown embedding provider"):
        build_embedder(Settings(embedding_model="nope:model"))


class RecordingEmbeddings:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        texts = kwargs["input"]
        # Return out of order to check that results are matched by index.
        data = [
            SimpleNamespace(index=i, embedding=[float(len(texts[i]))] * self.dim)
            for i in reversed(range(len(texts)))
        ]
        return SimpleNamespace(data=data)


async def test_openai_embedder_batches_orders_and_requests_the_dimension() -> None:
    embeddings = RecordingEmbeddings(dim=3)
    client = SimpleNamespace(embeddings=embeddings)
    embedder = OpenAIEmbedder("text-embedding-3-small", 3, client=cast(AsyncOpenAI, client))
    embedder.batch_size = 2
    vectors = await embedder.embed_documents(["a", "bb", "ccc"])
    assert vectors == [[1.0] * 3, [2.0] * 3, [3.0] * 3]
    assert [call["input"] for call in embeddings.calls] == [["a", "bb"], ["ccc"]]
    assert {call["dimensions"] for call in embeddings.calls} == {3}
    assert await embedder.embed_documents([]) == []

    wrong = OpenAIEmbedder("m", 4, client=cast(AsyncOpenAI, client))
    with pytest.raises(ProviderError, match="expected 4 dimensions"):
        await wrong.embed_query("text")
