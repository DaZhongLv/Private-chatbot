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
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Ollama not reachable at http://localhost:11434/api/tags: {e}")  # :contentReference[oaicite:10]{index=10}
        return []
    except Exception as e:
        print(f"[WARN] Unexpected error while fetching Ollama models: {e}")
        return []

def refresh_ollama_models():
    models = get_ollama_models()
    if not models:
        ui_warn("Ollama server not reachable. Start Ollama, then click 'Refresh models'.")
        return dd_update(choices=[], value=None)
    ui_info(f"Found {len(models)} Ollama models.")
    return dd_update(choices=models, value=models[0])



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
    Accept either:
    1) messages format: [{"role":"user","content":"..."}, ...]
    2) tuples format:   [("user msg","assistant msg"), ...]  or [["u","a"], ...]
    """
    msgs = []
    history = history or []

    if not history:
        return msgs

    # messages format
    if isinstance(history[0], dict):
        for h in history:
            role = h.get("role", "user")
            text = _extract_text(h.get("content")).strip()
            if text:
                msgs.append(ChatMessage(role=role, content=text))
        return msgs

    # tuples/list-of-2 format
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


def format_chat_history_as_markdown(history_messages):
    """
    history_messages: list of dicts, each like {"role": "user"/"assistant", "content": "..."}
    Returns: markdown string
    """
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


def write_markdown_to_tempfile(md_text: str, filename: str = "chat_history.md") -> str:
    """
    Write markdown to an absolute path, return that path for DownloadButton.
    Use absolute path to avoid relative-path issues in some Gradio 6 setups. :contentReference[oaicite:2]{index=2}
    """
    tmp_dir = Path(tempfile.gettempdir())
    out_path = tmp_dir / filename
    out_path.write_text(md_text, encoding="utf-8")
    return str(out_path)


# =============================================================================
# UI-safe notifications + version-safe updates
# =============================================================================

def ui_info(msg: str):
    try:
        gr.Info(msg)  # called directly :contentReference[oaicite:6]{index=6}
    except Exception:
        print("[INFO]", msg)

def ui_warn(msg: str):
    try:
        gr.Warning(msg)  # called directly :contentReference[oaicite:7]{index=7}
    except Exception:
        print("[WARN]", msg)

def ui_error(msg: str, e: Exception | None = None):
    # gr.Error must be raised :contentReference[oaicite:8]{index=8}
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
        return gr.Dropdown.update(**kwargs)  # widely used pattern :contentReference[oaicite:9]{index=9}
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
        # 对 streaming：用文本提示更稳（某些版本 raise 会导致前端卡住）
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

        # chunking settings
        Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # copy files into temp folder and compute stable hash
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
        loader = SimpleDirectoryReader(input_files=temp_paths)  # :contentReference[oaicite:11]{index=11}
        documents = loader.load_data()

        # connect qdrant (validate availability early)
        ui_info("Connecting to Qdrant…")
        client = qdrant_client.QdrantClient(host="localhost", port=6333)

        # quick ping: if qdrant is down this should fail fast
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

        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)  # :contentReference[oaicite:12]{index=12}
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        ui_info("Building vector index…")
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        chat_engine = index.as_chat_engine(
            chat_mode="context",
            similarity_top_k=top_k,
            system_prompt=_extract_text(system_prompt),
            verbose=True,
        )

        ui_info("Knowledge base created successfully!")  # :contentReference[oaicite:13]{index=13}

        kb_info_md = (
            f"✅ **Knowledge base ready**\n\n"
            f"- Collection: `{collection_name}`\n"
            f"- Mode: **{collection_mode}**\n"
            f"- chunk_size={chunk_size}, overlap={chunk_overlap}, top_k={top_k}\n"
            f"- reset={bool(reset_collection)}\n"
        )
        return chat_engine, kb_info_md, collection_name

    except gr.Error:
        # 你自己 raise 的 gr.Error，直接往上抛就行
        raise
    except Exception as e:
        print(f"[ERROR] Error during file processing: {e}")
        print(traceback.format_exc())
        ui_error(
            "Failed to process documents. Please ensure files are valid and Qdrant is running.",
            e,
        )
    finally:
        # cleanup temp copies
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


def clear_knowledge_base(collection_name: str):
    """
    Deletes the current Qdrant collection and clears UI/state.
    """
    try:
        # 1) 没有正在跟踪的 collection：说明还没创建过 KB 或者已清空
        if not collection_name:
            gr.Warning("Knowledge base is already empty (no active collection).")  # queue enabled -> modal :contentReference[oaicite:3]{index=3}
            return None, "", "", [], "", ""

        # 2) 连接 Qdrant
        client = qdrant_client.QdrantClient(host="localhost", port=6333)

        # 3) 判断 collection 是否存在
        cols = client.get_collections()
        existing = {c.name for c in cols.collections}

        if collection_name in existing:
            # 删除整个 collection（清空所有向量与payload）:contentReference[oaicite:4]{index=4}
            client.delete_collection(collection_name=collection_name)
            gr.Info(f"Knowledge base cleared: {collection_name}")  # :contentReference[oaicite:5]{index=5}
        else:
            gr.Warning("Collection not found in Qdrant. It may already be empty.")

        # 4) 清 UI + 清 state
        # 返回顺序要和 clear_button.click(outputs=[...]) 完全一致
        chat_engine = None
        kb_info_md = ""
        collection_name = ""
        chatbot_history = []     # Chatbot value 清空
        sources_md = ""
        status_md = ""
        return chat_engine, kb_info_md, collection_name, chatbot_history, sources_md, status_md

    except Exception as e:
        print(f"[ERROR] Error clearing knowledge base: {e}")
        print(traceback.format_exc())
        # raise gr.Error 会在 UI 弹红框 :contentReference[oaicite:6]{index=6}
        raise gr.Error(f"Failed to clear knowledge base. Is Qdrant running? Details: {type(e).__name__}: {e}")



def chat_with_sources(message, history, chat_engine):
    """
    Gradio 6 messages format:
      history = [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]
    Yields:
      (history, sources_md, "", status_md, download_button_update)
    """
    t0 = time.perf_counter()

    question = _extract_text(message).strip()
    history = list(history or [])

    # 0) 空输入
    if not question:
        yield history, "", "", "", gr.update(visible=False, value=None)
        return

    # 1) KB 未就绪
    if chat_engine is None or not hasattr(chat_engine, "chat"):
        ui_warn("Knowledge base not created yet. Please click 'Create Knowledge Base' first.")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "⚠️ Knowledge base not ready. Please create it first."})
        # 这里也可以允许下载（会下载当前对话提示），看你喜好；我这里先隐藏
        yield history, "No sources (KB not ready).", "", "", gr.update(visible=False, value=None)
        return

    # 2) 先显示用户问题
    history.append({"role": "user", "content": question})
    yield history, "", "", f"🔎 Retrieving… ({time.perf_counter()-t0:.1f}s)", gr.update(visible=False, value=None)

    # 3) 后台线程跑 llamaindex
    def _run():
        return chat_engine.chat(question)

    future = _EXECUTOR.submit(_run)

    while not future.done():
        elapsed = time.perf_counter() - t0
        yield history, "", "", f"🤔 Thinking… ({elapsed:.1f}s)", gr.update(visible=False, value=None)
        time.sleep(0.25)

    # 4) 拿结果
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

    # 5) sources
    try:
        source_text = _format_sources(resp)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[WARN] source formatting failed: {e}\n{tb}")
        source_text = f"### Retrieved Sources (formatting failed)\n\n**{type(e).__name__}:** {e}\n\n```text\n{tb}\n```"

    # 6) Day18: 导出 markdown -> 写临时文件 -> 给 DownloadButton
    md_text = format_chat_history_as_markdown(history)
    md_path = write_markdown_to_tempfile(md_text, filename="chat_history.md")

    elapsed = time.perf_counter() - t0
    # DownloadButton 更新：显示 + 指向文件路径
    yield history, source_text, "", f"✅ Done ({elapsed:.1f}s)", gr.update(visible=True, value=md_path)
    yield history, source_text, "", "", gr.update(visible=True, value=md_path)


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
            # Left panel
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="Upload documents to create a knowledge base",
                    file_count="multiple",
                )
                with gr.Row():
                    process_button = gr.Button("Create Knowledge Base", variant="primary")
                    clear_button   = gr.Button("Clear Knowledge Base", variant="stop")


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
            outputs=[chat_engine_state, kb_info_md, collection_name_state],  # ✅ 多一个 state
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




if __name__ == "__main__":
    demo.queue() 
    demo.launch(
        share=False,
        debug=True,
        show_error=True,
        theme=gr.themes.Soft(),
    )
