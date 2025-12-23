"""
app.py

Private Chatbot with Local LLM

Two tabs:
1) General Chat (Ollama) with dynamic model switching + streaming
2) Chat with Documents (LlamaIndex + Qdrant) with:
   - configurable chunking/top_k
   - persona/system prompt (via chat_mode="context")
   - source chunks display
   - live status line (retrieving/thinking + elapsed seconds)
   - collection strategy:
       * Fixed collection  (persistent library)
       * Unique per upload (avoid mixing)
   - optional reset (delete collection before ingest)
"""

import os
import re
import shutil
import time
import hashlib
import traceback
import concurrent.futures

import gradio as gr
import requests
import qdrant_client

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.llms import ChatMessage
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter


# =============================================================================
# Global configuration
# =============================================================================

Settings.llm = Ollama(model="llama3.2:3b", request_timeout=300.0)

def _init_embed_model():
    """
    NOTE:
    第一次跑会下载 embedding 模型权重（正常现象），之后会走缓存。
    如果你经常重装环境/清缓存，就会反复下载。
    """
    try:
        # multilingual (better if you ask in Chinese)
        return resolve_embed_model("local:BAAI/bge-small-en-v1.5")   # bge-m3
    except Exception:
        print("[WARN] local:BAAI/bge-m3 not available, fallback to bge-small-en-v1.5")
        return resolve_embed_model("local:BAAI/bge-small-en-v1.5")

Settings.embed_model = _init_embed_model()

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


# =============================================================================
# Helpers
# =============================================================================

def _safe_slug(s: str) -> str:
    s = (s or "default").strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "default"


def get_ollama_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        data = r.json()
        models = data.get("models", [])
        return [m.get("name") for m in models if m.get("name")]
    except requests.exceptions.RequestException:
        print("[WARN] Ollama not reachable at http://localhost:11434. Model list empty.")
        return []


def _extract_text(x) -> str:
    """
    Normalize Gradio inputs into plain string.

    - str -> str
    - dict like {"text": "..."} -> "..."
    - list of blocks [{"type":"text","text":"..."}] -> joined string
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # common: {"text": "...", "type": "text"}
        if "text" in x and isinstance(x["text"], str):
            return x["text"]
        return str(x)
    if isinstance(x, list):
        parts = []
        for blk in x:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                if "text" in blk and isinstance(blk["text"], str):
                    parts.append(blk["text"])
        return "".join(parts)
    return str(x)


def _history_to_chatmessages(history):
    """
    Convert Gradio ChatInterface history to LlamaIndex ChatMessage list.
    history example: [{"role":"user","content":"..."}, ...]
    """
    msgs = []
    history = history or []
    if history and isinstance(history[0], dict):
        for h in history:
            role = h.get("role", "user")
            text = _extract_text(h.get("content")).strip()
            if text:
                msgs.append(ChatMessage(role=role, content=text))
    return msgs


def _format_sources(resp) -> str:
    """
    Format retrieved evidence chunks into Markdown.

    resp.source_nodes: List[NodeWithScore]
      - nws.node: Node
      - nws.score: float
      - node.metadata: may include file_name/filename/source
      - node.get_content(): chunk text
    """
    source_nodes = getattr(resp, "source_nodes", None) or []
    md = "### Retrieved Sources\n\n"

    if not source_nodes:
        md += "_No source nodes returned._"
        return md

    for i, nws in enumerate(source_nodes):
        node = getattr(nws, "node", nws)
        score = getattr(nws, "score", None)

        meta = getattr(node, "metadata", {}) or {}
        file_name = meta.get("file_name") or meta.get("filename") or meta.get("source") or "N/A"

        if hasattr(node, "get_content"):
            chunk_text = node.get_content() or ""
        else:
            chunk_text = getattr(node, "text", "") or ""

        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        md += f"**Source {i+1}** — `{file_name}` (score: {score_str})\n\n"
        md += f"```text\n{chunk_text.strip()}\n```\n\n"

    return md


# =============================================================================
# Tab 1: General Chat (streaming)
# =============================================================================

def chat_with_llm(message, history, system_prompt, model_name):
    if not model_name:
        raise gr.Warning("No model selected. Please choose a model from the dropdown.")

    temp_llm = Ollama(model=model_name, request_timeout=300.0)

    msg = _extract_text(message)
    messages = [ChatMessage(role="system", content=_extract_text(system_prompt))]
    messages.extend(_history_to_chatmessages(history))
    messages.append(ChatMessage(role="user", content=msg))

    response = ""
    for r in temp_llm.stream_chat(messages):
        response += (r.delta or "")
        yield response


# =============================================================================
# Tab 2: Documents - Build KB (Qdrant) + Chat with Sources + Persona
# =============================================================================

def handle_file_processing_stateful(
    files,
    chunk_size,
    chunk_overlap,
    top_k,
    system_prompt,
    kb_name,
    collection_mode,
    reset_collection,
):
    if not files:
        raise gr.Warning("No files uploaded. Please upload documents first.")

    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)
    top_k = int(top_k)

    # chunking settings for ingestion
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # copy files into temp folder and compute stable hash
    temp_dir = "temp_docs_for_processing"
    os.makedirs(temp_dir, exist_ok=True)

    sha1 = hashlib.sha1()
    temp_paths = []
    for f in files:
        src_path = f.name if hasattr(f, "name") else str(f)
        base = os.path.basename(src_path)
        dst_path = os.path.join(temp_dir, base)

        with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
            while True:
                buf = src.read(1024 * 1024)
                if not buf:
                    break
                sha1.update(buf)
                dst.write(buf)

        temp_paths.append(dst_path)

    file_hash = sha1.hexdigest()[:12]
    kb_slug = _safe_slug(_extract_text(kb_name))

    # collection naming
    if collection_mode == "Fixed collection":
        collection_name = f"private_chatbot_{kb_slug}"
    else:
        collection_name = f"private_chatbot_{kb_slug}_{file_hash}"

    # load docs
    loader = SimpleDirectoryReader(input_files=temp_paths)
    documents = loader.load_data()

    # connect qdrant
    client = qdrant_client.QdrantClient(host="localhost", port=6333)

    if reset_collection:
        try:
            client.delete_collection(collection_name=collection_name)
            print(f"[Qdrant] deleted collection: {collection_name}")
        except Exception:
            pass

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # build index + write embeddings
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # cleanup temp copies
    for p in temp_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    # persona support (IMPORTANT)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        similarity_top_k=top_k,
        system_prompt=_extract_text(system_prompt),
        verbose=True,
    )

    kb_info_md = (
        f"✅ **Knowledge base ready**\n\n"
        f"- Collection: `{collection_name}`\n"
        f"- Mode: **{collection_mode}**\n"
        f"- chunk_size={chunk_size}, overlap={chunk_overlap}, top_k={top_k}\n"
        f"- reset={bool(reset_collection)}\n"
    )

    return chat_engine, kb_info_md


def chat_with_sources(message, history, chat_engine):
    """
    Outputs:
      (chat_history, sources_md, cleared_textbox, status_md)

    - chat_history: list of {"role":"user/assistant", "content": "..."}
    - sources_md: Markdown of retrieved chunks
    - cleared_textbox: "" (clears input)
    - status_md: progress line with elapsed seconds
    """
    t0 = time.perf_counter()

    question = _extract_text(message).strip()
    history = list(history or [])

    source_text = ""
    status_text = ""

    if not question:
        yield history, source_text, "", status_text
        return

    # ✅ 防呆：避免你现在遇到的 "str has no attribute chat"
    if chat_engine is None or not hasattr(chat_engine, "chat"):
        raise gr.Warning(
            "Chat engine is not ready (state is invalid). "
            "Please click 'Create Knowledge Base' again. "
            "If this keeps happening, your click outputs order is wrong."
        )

    # show user message immediately
    history.append({"role": "user", "content": question})
    yield history, source_text, "", f"🔎 Retrieving… ({time.perf_counter()-t0:.1f}s)"

    # run heavy call in background thread so seconds can update
    def _run():
        return chat_engine.chat(question)

    future = _EXECUTOR.submit(_run)

    while not future.done():
        elapsed = time.perf_counter() - t0
        yield history, source_text, "", f"🤔 Thinking… ({elapsed:.1f}s)"
        time.sleep(0.25)

    try:
        resp = future.result()
    except Exception as e:
        tb = traceback.format_exc()
        history.append({"role": "assistant", "content": f"❌ Error: {type(e).__name__}: {e}"})
        source_text = f"### Error\n\n**{type(e).__name__}:** {e}\n\n```text\n{tb}\n```"
        yield history, source_text, "", ""
        return

    answer_text = getattr(resp, "response", None) or str(resp)
    history.append({"role": "assistant", "content": answer_text})

    try:
        source_text = _format_sources(resp)
    except Exception as e:
        tb = traceback.format_exc()
        source_text = f"### Retrieved Sources (formatting failed)\n\n**{type(e).__name__}:** {e}\n\n```text\n{tb}\n```"

    elapsed = time.perf_counter() - t0
    yield history, source_text, "", f"✅ Done ({elapsed:.1f}s)"
    yield history, source_text, "", ""


# =============================================================================
# UI
# =============================================================================

with gr.Blocks(title="Private Chatbot with Local LLM") as demo:
    gr.Markdown("# Private Chatbot with Local LLM")

    # ------------------ Tab 1: General Chat ------------------
    with gr.Tab("General Chat"):
        model_list = get_ollama_models()

        model_dropdown = gr.Dropdown(
            label="Select a Model",
            choices=model_list,
            value=(model_list[0] if model_list else None),
            interactive=True,
        )

        system_prompt_box = gr.Textbox(
            label="System Prompt",
            value="You are a helpful and friendly assistant.",
            lines=3,
        )

        general_chatbot = gr.Chatbot(height=500, label="General Chat")
        general_textbox = gr.Textbox(placeholder="Ask me anything...", container=False)

        gr.ChatInterface(
            fn=chat_with_llm,
            chatbot=general_chatbot,
            textbox=general_textbox,
            additional_inputs=[system_prompt_box, model_dropdown],
        )

        gr.ClearButton([general_chatbot, general_textbox], value="Clear Chat")

    # ------------------ Tab 2: Chat with Documents ------------------
    with gr.Tab("Chat with Documents"):
        chat_engine_state = gr.State(None)

        with gr.Row():
            # Left panel
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="Upload documents to create a knowledge base",
                    file_count="multiple",
                )
                process_button = gr.Button("Create Knowledge Base", variant="primary")

                with gr.Accordion("Advanced RAG Settings", open=True):
                    chunk_size_slider = gr.Slider(128, 2048, value=512, step=64, label="Chunk Size")
                    chunk_overlap_slider = gr.Slider(0, 512, value=50, step=16, label="Chunk Overlap")
                    top_k_slider = gr.Slider(1, 10, value=3, step=1, label="Top-K")

                    kb_name_box = gr.Textbox(
                        label="Knowledge Base Name",
                        value="default",
                    )

                    collection_mode_radio = gr.Radio(
                        choices=["Unique per upload", "Fixed collection"],
                        value="Fixed collection",
                        label="Collection Strategy",
                        info="Unique per upload avoids mixing old docs. Fixed collection builds a persistent library.",
                    )

                    reset_collection_cb = gr.Checkbox(
                        label="Reset collection (delete existing vectors before ingest)",
                        value=True,
                    )

                    system_prompt_docs = gr.Textbox(
                        label="Document Assistant System Prompt",
                        value="Answer strictly using the provided documents. If the documents do not contain the answer, say so.",
                        lines=4,
                    )

                kb_info_md = gr.Markdown("")

            # Right panel
            with gr.Column(scale=3):
                chatbot_docs = gr.Chatbot(height=500, label="Chat with Your Documents")
                status_md = gr.Markdown("")

                question_box_docs = gr.Textbox(
                    label="Ask a question",
                    placeholder="Ask something about your uploaded documents...",
                )

                with gr.Accordion("Retrieved Sources", open=False):
                    source_markdown = gr.Markdown("")

                question_box_docs.submit(
                    fn=chat_with_sources,
                    inputs=[question_box_docs, chatbot_docs, chat_engine_state],
                    outputs=[chatbot_docs, source_markdown, question_box_docs, status_md],
                )

                gr.ClearButton(
                    [chatbot_docs, question_box_docs, source_markdown, status_md],
                    value="Clear",
                )

        # ✅ 关键：outputs 顺序必须是 [chat_engine_state, kb_info_md]
        process_button.click(
            fn=handle_file_processing_stateful,
            inputs=[
                file_upload,
                chunk_size_slider,
                chunk_overlap_slider,
                top_k_slider,
                system_prompt_docs,
                kb_name_box,
                collection_mode_radio,
                reset_collection_cb,
            ],
            outputs=[chat_engine_state, kb_info_md],
            show_progress="full",
        ).then(
            # Clear chat UI after rebuilding KB (recommended)
            fn=lambda: ([], "", "", ""),
            inputs=None,
            outputs=[chatbot_docs, source_markdown, question_box_docs, status_md],
        )


if __name__ == "__main__":
    demo.launch(
        share=False,
        debug=True,
        show_error=True,
        theme=gr.themes.Soft(),
    )
