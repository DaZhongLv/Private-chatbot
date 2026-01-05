import re
from private_chatbot.utils import _safe_slug, _extract_text, format_chat_history_as_markdown, write_markdown_to_tempfile

def test_safe_slug_basic():
    assert _safe_slug("Hello World") == "hello_world"
    assert _safe_slug("  ") == "default"
    assert _safe_slug("A---B") == "a---b"
    assert _safe_slug("a中文b") == "a_b"

def test_extract_text_variants():
    assert _extract_text("hi") == "hi"
    assert _extract_text({"text": "yo"}) == "yo"
    assert _extract_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert _extract_text(None) == ""

def test_format_chat_history_markdown_smoke():
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    md = format_chat_history_as_markdown(history)
    assert md.startswith("# Chat History")
    assert "**You:**" in md
    assert "**Assistant:**" in md
    assert "Hello" in md
    assert "Hi there" in md
    assert re.search(r"Exported on: \d{4}-\d{2}-\d{2}", md)

def test_write_markdown_to_tempfile(tmp_path):
    md = "# Title\n\ncontent"
    out = write_markdown_to_tempfile(md, filename="x.md", out_dir=str(tmp_path))
    p = tmp_path / "x.md"
    assert out == str(p)
    assert p.exists()
    assert p.read_text(encoding="utf-8") == md

