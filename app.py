"""
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
import inspect

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

import datetime
import tempfile
from pathlib import Path

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
        return resolve_embed_model("local:BAAI/bge-small-en-v1.5")  # bge-m3
    except Exception:
        print("[WARN] local:BAAI/bge-m3 not available, fallback to bge-small-en-v1.5")
        return resolve_embed_model("local:BAAI/bge-small-en-v1.5")


Settings.embed_model = _init_embed_model()

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# =============================================================================
# Helpers
# =============================================================================

from private_chatbot.utils import (
    _safe_slug,
    _extract_text,
    _history_to_chatmessages,
    format_chat_history_as_markdown,
    write_markdown_to_tempfile,
)

from private_chatbot.services import get_ollama_models


def refresh_ollama_models():
    models = get_ollama_models()
    if not models:
        ui_warn("Ollama server not reachable. Start Ollama, then click 'Refresh models'.")
        return dd_update(choices=[], value=None)
    ui_info(f"Found {len(models)} Ollama models.")
    return dd_update(choices=models, value=models[0])


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
# UI-safe notifications + version-safe updates
# =============================================================================

def ui_info(msg: str):
    try:
        gr.Info(msg)
    except Exception:
        print("[INFO]", msg)


def ui_warn(msg: str):
    try:
        gr.Warning(msg)
    except Exception:
        print("[WARN]", msg)


def ui_error(msg: str, e: Exception | None = None):
    if e is not None:
        msg = f"{msg}\n\nDetails: {type(e).__name__}: {e}"
    raise gr.Error(msg)


def dd_update(**kwargs):
    """
    Version-safe dropdown update:
    - older: gr.Dropdown.update(...)
    - newer: gr.update(...)
    """
    if hasattr(gr.Dropdown, "update"):
        return gr.Dropdown.update(**kwargs)
    return gr.update(**kwargs)


# =============================================================================
# Tab 1: General Chat (streaming)
# =============================================================================

def chat_with_llm(message, history, system_prompt, model_name):
    try:
        if not model_name:
            ui_warn("No model selected. Please choose a model from the dropdown.")
            yield "⚠️ No model selected."
            return

        temp_llm = Ollama(model=model_name, request_timeout=300.0)

        msg = _extract_text(message)
        messages = [ChatMessage(role="system", content=_extract_text(system_prompt))]
        messages.extend(_history_to_chatmessages(history))
        messages.append(ChatMessage(role="user", content=msg))

        response = ""
        for r in temp_llm.stream_chat(messages):
            response += (r.delta or "")
            yield response

    except Exception as e:
        print(f"[ERROR] chat_with_llm failed: {e}")
        print(traceback.format_exc())
        yield f"❌ An error occurred: {type(e).__name__}: {e}\n\n(See terminal for full traceback.)"


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
        ui_warn("No files uploaded. Please upload documents first.")
        return None, "", ""

    temp_dir = "temp_docs_for_processing"
    temp_paths = []

    try:
        chunk_size = int(chunk_size)
        chunk_overlap = int(chunk_overlap)
        top_k = int(top_k)

        if chunk_overlap >= chunk_size:
            ui_warn("chunk_overlap should be smaller than chunk_size. Adjusting overlap.")
            chunk_overlap = max(0, chunk_size // 4)

        Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        os.makedirs(temp_dir, exist_ok=True)
        sha1 = hashlib.sha1()

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

        if collection_mode == "Fixed collection":
            collection_name = f"private_chatbot_{kb_slug}"
        else:
            collection_name = f"private_chatbot_{kb_slug}_{file_hash}"

        ui_info("Reading documents…")
        loader = SimpleDirectoryReader(input_files=temp_paths)
        documents = loader.load_data()

        ui_info("Connecting to Qdrant…")
        client = qdrant_client.QdrantClient(host="localhost", port=6333)

        try:
            client.get_collections()
        except Exception as e:
            ui_error("Failed to connect to Qdrant. Please ensure Qdrant is running (Docker container up).", e)

        if reset_collection:
            try:
                client.delete_collection(collection_name=collection_name)
                print(f"[Qdrant] deleted collection: {collection_name}")
            except Exception:
                pass

        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        ui_info("Building vector index…")
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        chat_engine = index.as_chat_engine(
            chat_mode="context",
            similarity_top_k=top_k,
            system_prompt=_extract_text(system_prompt),
            verbose=True,
        )

        ui_info("Knowledge base created successfully!")

        kb_info_md = (
            f"✅ **Knowledge base ready**\n\n"
            f"- Collection: `{collection_name}`\n"
            f"- Mode: **{collection_mode}**\n"
            f"- chunk_size={chunk_size}, overlap={chunk_overlap}, top_k={top_k}\n"
            f"- reset={bool(reset_collection)}\n"
        )
        return chat_engine, kb_info_md, collection_name

    except gr.Error:
        raise
    except Exception as e:
        print(f"[ERROR] Error during file processing: {e}")
        print(traceback.format_exc())
        ui_error(
            "Failed to process documents. Please ensure files are valid and Qdrant is running.",
            e,
        )
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


def clear_knowledge_base(collection_name: str):
    try:
        if not collection_name:
            gr.Warning("Knowledge base is already empty (no active collection).")
            return None, "", "", [], "", ""

        client = qdrant_client.QdrantClient(host="localhost", port=6333)

        cols = client.get_collections()
        existing = {c.name for c in cols.collections}

        if collection_name in existing:
            client.delete_collection(collection_name=collection_name)
            gr.Info(f"Knowledge base cleared: {collection_name}")
        else:
            gr.Warning("Collection not found in Qdrant. It may already be empty.")

        chat_engine = None
        kb_info_md = ""
        collection_name = ""
        chatbot_history = []
        sources_md = ""
        status_md = ""
        return chat_engine, kb_info_md, collection_name, chatbot_history, sources_md, status_md

    except Exception as e:
        print(f"[ERROR] Error clearing knowledge base: {e}")
        print(traceback.format_exc())
        raise gr.Error(f"Failed to clear knowledge base. Is Qdrant running? Details: {type(e).__name__}: {e}")


def chat_with_sources(message, history, chat_engine):
    t0 = time.perf_counter()

    question = _extract_text(message).strip()
    history = list(history or [])

    if not question:
        yield history, "", "", "", gr.update(visible=False, value=None)
        return

    if chat_engine is None or not hasattr(chat_engine, "chat"):
        ui_warn("Knowledge base not created yet. Please click 'Create Knowledge Base' first.")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "⚠️ Knowledge base not ready. Please create it first."})
        yield history, "No sources (KB not ready).", "", "", gr.update(visible=False, value=None)
        return

    history.append({"role": "user", "content": question})
    yield history, "", "", f"🔎 Retrieving… ({time.perf_counter()-t0:.1f}s)", gr.update(visible=False, value=None)

    def _run():
        return chat_engine.chat(question)

    future = _EXECUTOR.submit(_run)

    while not future.done():
        elapsed = time.perf_counter() - t0
        yield history, "", "", f"🤔 Thinking… ({elapsed:.1f}s)", gr.update(visible=False, value=None)
        time.sleep(0.25)

    try:
        resp = future.result()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] chat_engine.chat failed: {e}\n{tb}")
        history.append({"role": "assistant", "content": f"❌ Error: {type(e).__name__}: {e}"})
        source_text = f"### Error\n\n**{type(e).__name__}:** {e}\n\n```text\n{tb}\n```"
        yield history, source_text, "", "", gr.update(visible=False, value=None)
        return

    answer_text = getattr(resp, "response", None) or str(resp)
    history.append({"role": "assistant", "content": answer_text})

    try:
        source_text = _format_sources(resp)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[WARN] source formatting failed: {e}\n{tb}")
        source_text = f"### Retrieved Sources (formatting failed)\n\n**{type(e).__name__}:** {e}\n\n```text\n{tb}\n```"

    md_text = format_chat_history_as_markdown(history)
    md_path = write_markdown_to_tempfile(md_text, filename="chat_history.md")

    elapsed = time.perf_counter() - t0
    yield history, source_text, "", f"✅ Done ({elapsed:.1f}s)", gr.update(visible=True, value=md_path)
    yield history, source_text, "", "", gr.update(visible=True, value=md_path)


def verify_services_running():
    missing = []

    try:
        requests.get("http://localhost:11434", timeout=1.5)
    except Exception:
        missing.append("Ollama (http://localhost:11434)")

    try:
        client = qdrant_client.QdrantClient(host="localhost", port=6333)
        client.get_collections()
    except Exception:
        missing.append("Qdrant (localhost:6333)")

    if missing:
        print("[WARN] Missing services: " + ", ".join(missing) + ". App may not function correctly.")


# =============================================================================
# UI
# =============================================================================

verify_services_running()

with gr.Blocks(title="Private Chatbot with Local LLM") as demo:
    gr.Markdown("# Private Chatbot with Local LLM")

    # ------------------ Tab 1: General Chat ------------------
    with gr.Tab("General Chat"):
        model_dropdown = gr.Dropdown(
            label="Select a Model",
            choices=[],
            value=None,
            interactive=True,
        )
        refresh_models_btn = gr.Button("Refresh models")
        refresh_models_btn.click(fn=refresh_ollama_models, inputs=None, outputs=model_dropdown)

        demo.load(fn=refresh_ollama_models, inputs=None, outputs=model_dropdown)

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
        collection_name_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="Upload documents to create a knowledge base",
                    file_count="multiple",
                )
                with gr.Row():
                    process_button = gr.Button("Create Knowledge Base", variant="primary")
                    clear_button = gr.Button("Clear Knowledge Base", variant="stop")

                with gr.Accordion("Advanced RAG Settings", open=False):
                    chunk_size_slider = gr.Slider(128, 2048, value=512, step=64, label="Chunk Size")
                    chunk_overlap_slider = gr.Slider(0, 512, value=50, step=16, label="Chunk Overlap")
                    top_k_slider = gr.Slider(1, 10, value=3, step=1, label="Top-K")

                    kb_name_box = gr.Textbox(label="Knowledge Base Name", value="default")

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

            with gr.Column(scale=3):
                chatbot_docs = gr.Chatbot(height=500, label="Chat with Your Documents")
                status_md = gr.Markdown("")

                question_box_docs = gr.Textbox(
                    label="Ask a question",
                    placeholder="Ask something about your uploaded documents...",
                )

                with gr.Accordion("Retrieved Sources", open=False):
                    source_markdown = gr.Markdown("")
                    download_button = gr.DownloadButton(
                        label="Download Chat History",
                        visible=False,
                        value=None,
                    )

                question_box_docs.submit(
                    fn=chat_with_sources,
                    inputs=[question_box_docs, chatbot_docs, chat_engine_state],
                    outputs=[chatbot_docs, source_markdown, question_box_docs, status_md, download_button],
                )

                gr.ClearButton(
                    [chatbot_docs, question_box_docs, source_markdown, status_md],
                    value="Clear",
                )

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
            outputs=[chat_engine_state, kb_info_md, collection_name_state],
            show_progress="full",
        ).then(
            fn=lambda: ([], "", "", "", gr.update(visible=False, value=None)),
            inputs=None,
            outputs=[chatbot_docs, source_markdown, question_box_docs, status_md, download_button],
        )

        clear_button.click(
            fn=clear_knowledge_base,
            inputs=[collection_name_state],
            outputs=[chat_engine_state, kb_info_md, collection_name_state, chatbot_docs, source_markdown, status_md],
            show_progress="full",
        )

# =============================================================================
# Unified entrypoint (merged from your launcher wrapper)
# =============================================================================

def _launch_demo():
    # Match your wrapper behavior (PORT + 0.0.0.0), but keep compatibility across gradio versions.
    port = int(os.getenv("PORT", "7860"))

    # If queue exists in your gradio version, enabling it can improve responsiveness.
    try:
        demo.queue()
    except Exception:
        pass

    sig = inspect.signature(demo.launch)
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": port,
        "show_error": True,
        # Keep your previous theme/debug defaults when supported
        "share": False,
        "debug": True,
        "theme": gr.themes.Soft(),
    }

    # ssr_mode exists only in some versions; add it only if supported
    if "ssr_mode" in sig.parameters:
        launch_kwargs["ssr_mode"] = False

    # Remove kwargs not supported by current gradio version
    launch_kwargs = {k: v for k, v in launch_kwargs.items() if k in sig.parameters}
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    _launch_demo()
