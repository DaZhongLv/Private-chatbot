# private_chatbot/utils.py
import re
import datetime
import tempfile
from pathlib import Path
from llama_index.core.llms import ChatMessage

def _safe_slug(s: str) -> str:
    s = (s or "default").strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "default"

def _extract_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if "text" in x and isinstance(x["text"], str):
            return x["text"]
        return str(x)
    if isinstance(x, list):
        parts = []
        for blk in x:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict) and "text" in blk and isinstance(blk["text"], str):
                parts.append(blk["text"])
        return "".join(parts)
    return str(x)

def _history_to_chatmessages(history):
    msgs = []
    history = history or []
    if not history:
        return msgs

    if isinstance(history[0], dict):
        for h in history:
            role = h.get("role", "user")
            text = _extract_text(h.get("content")).strip()
            if text:
                msgs.append(ChatMessage(role=role, content=text))
        return msgs

    for item in history:
        if not item:
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            u, a = item[0], item[1]
            u = _extract_text(u).strip()
            a = _extract_text(a).strip()
            if u:
                msgs.append(ChatMessage(role="user", content=u))
            if a:
                msgs.append(ChatMessage(role="assistant", content=a))
    return msgs

def format_chat_history_as_markdown(history_messages):
    if not history_messages:
        return "No conversation history."

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = f"# Chat History\n*Exported on: {now}*\n\n"

    for m in history_messages:
        role = (m.get("role") or "assistant").strip()
        content = _extract_text(m.get("content", "")).strip()
        if not content:
            continue

        if role == "user":
            md += f"**You:**\n\n{content}\n\n"
        else:
            md += f"**Assistant:**\n\n{content}\n\n"
        md += "---\n\n"

    return md

def write_markdown_to_tempfile(md_text: str, filename: str = "chat_history.md", out_dir: str | None = None) -> str:
    tmp_dir = Path(out_dir) if out_dir else Path(tempfile.gettempdir())
    out_path = tmp_dir / filename
    out_path.write_text(md_text, encoding="utf-8")
    return str(out_path)

