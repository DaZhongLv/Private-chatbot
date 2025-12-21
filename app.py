"""
app.py

A small local RAG application with two main capabilities:

1. General Chat:
   - Simple chat UI backed by a local Ollama LLM (llama3.2:3b).
   - User can customize the system prompt.

2. Chat with Documents:
   - User uploads one or more documents (PDF, DOCX, TXT).
   - Documents are loaded via LlamaIndex's SimpleDirectoryReader.
   - Embeddings are stored in a Qdrant vector database running locally in Docker.
   - A chat engine is created on top of this vector store and stored in gr.State.
   - After an app restart, the chat engine can be rebuilt from the existing Qdrant
     collection, so the knowledge base persists across sessions.
"""

import os
import shutil
import time  # currently not used, but kept in case we want timing / logging later
import traceback

import gradio as gr
import qdrant_client
import requests

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.llms import ChatMessage
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

PERSIST_DIR = "./storage"

llm = Ollama(model="llama3.2:3b", request_timeout=300.0)
Settings.llm = llm

Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------------
# Optional helper: load a file-based index if present (Day 8 style)
# ---------------------------------------------------------------------
def get_index(documents=None):
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("Loaded existing index.")
    elif documents:
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("Built and saved a new index.")
    else:
        return None
    return index


def get_ollama_models():
    """
    Fetches list of locally available models from Ollama API.
    """
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        data = r.json()
        models = data.get("models", [])
        return [m.get("name") for m in models if m.get("name")]
    except requests.exceptions.RequestException:
        print("[WARN] Ollama server not reachable at http://localhost:11434. Model list empty.")
        return []


# ---------------------------------------------------------------------
# Gradio history <-> LlamaIndex ChatMessage utilities (General Chat only)
# ---------------------------------------------------------------------
def _extract_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                if blk.get("type") == "text" and "text" in blk:
                    parts.append(blk["text"])
                elif "text" in blk:
                    parts.append(blk["text"])
        return "".join(parts)
    return ""


def _history_to_chatmessages(history):
    msgs = []
    if not history:
        return msgs

    if isinstance(history[0], dict):
        for h in history:
            role = h.get("role", "user")
            text = _extract_text(h.get("content")).strip()
            if text:
                msgs.append(ChatMessage(role=role, content=text))
        return msgs

    if isinstance(history[0], (list, tuple)) and len(history[0]) == 2:
        for u, a in history:
            if u:
                msgs.append(ChatMessage(role="user", content=str(u)))
            if a:
                msgs.append(ChatMessage(role="assistant", content=str(a)))
        return msgs

    return msgs


# ---------------------------------------------------------------------
# General Chat (no documents) - streaming
# ---------------------------------------------------------------------
def chat_with_llm(message, history, system_prompt, model_name):
    if not model_name:
        raise gr.Warning("No model selected. Please choose a model from the dropdown.")

    temp_llm = Ollama(model=model_name, request_timeout=300.0)

    messages = [ChatMessage(role="system", content=system_prompt)]
    messages.extend(_history_to_chatmessages(history))
    messages.append(ChatMessage(role="user", content=message))

    response = ""
    for r in temp_llm.stream_chat(messages):
        response += (r.delta or "")
        yield response


# ---------------------------------------------------------------------
# Document upload & Qdrant-backed knowledge base (stateful)
# ---------------------------------------------------------------------
def handle_file_processing_stateful(files, chunk_size, chunk_overlap, top_k):
    if files is None or len(files) == 0:
        raise gr.Warning("No files uploaded. Please upload documents to create a knowledge base.")

    temp_dir = "temp_docs_for_processing"
    os.makedirs(temp_dir, exist_ok=True)

    temp_paths = []

    Settings.node_parser = SentenceSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
    )

    for f in files:
        src_path = f.name if hasattr(f, "name") else str(f)
        tmp_path = os.path.join(temp_dir, os.path.basename(src_path))

        with open(src_path, "rb") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        temp_paths.append(tmp_path)

    loader = SimpleDirectoryReader(input_files=temp_paths)
    documents = loader.load_data()

    # Qdrant
    client = qdrant_client.QdrantClient(host="localhost", port=6333)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="private_chatbot_docs",
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    try:
        info = client.get_collection("private_chatbot_docs")
        print("[Qdrant] collection info:", info)
    except Exception as e:
        print("[Qdrant] get_collection failed:", e)

    for p in temp_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    top_k = int(top_k)

    chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        similarity_top_k=top_k,
        verbose=True,
    )

    gr.Info(f"Knowledge base created successfully! top_k={top_k}. You can now ask questions.")

    return chat_engine, top_k


# ---------------------------------------------------------------------
# Chat with Documents: show answer + retrieved sources
#   - generator: first yield placeholder so the question appears immediately
#   - returns 3 outputs: (chatbot_history, sources_markdown, cleared_textbox)
# ---------------------------------------------------------------------
import traceback

import traceback

def chat_with_sources(message, history, chat_engine):
    """
    messages-format Chatbot history:
      history = [{"role": "user"/"assistant", "content": "..."} , ...]
    Returns 4 outputs:
      (history, sources_markdown, cleared_textbox, status_markdown)
    """
    question = (message or "").strip()

    if history is None:
        history = []
    history = list(history)

    if not question:
        yield history, "", "", ""
        return

    if chat_engine is None:
        raise gr.Warning("Knowledge base not created yet. Please click 'Create Knowledge Base' first.")

    # 1) 先把 user message 放进 chat（但不放processing）
    history.append({"role": "user", "content": question})

    # 显示 status，并清空输入框
    yield history, "", "", "🤔 *…Thinking…*"

    # 2) RAG call
    try:
        resp = chat_engine.chat(question)
    except Exception as e:
        tb = traceback.format_exc()
        history.append({"role": "assistant", "content": f"❌ Error: {type(e).__name__}: {e}"})
        src_md = f"### Error\n\n**{type(e).__name__}:** {e}\n\n```text\n{tb}\n```"
        yield history, src_md, "", ""   # 清空 status
        return

    answer_text = getattr(resp, "response", None) or str(resp)
    history.append({"role": "assistant", "content": answer_text})

    # 3) sources
    try:
        source_nodes = getattr(resp, "source_nodes", None) or []
        source_text = "### Retrieved Sources\n\n"

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
            source_text += f"**Source {i+1}** — `{file_name}` (score: {score_str})\n\n"
            source_text += f"```text\n{chunk_text.strip()}\n```\n\n"

    except Exception as e:
        tb = traceback.format_exc()
        source_text = f"### Retrieved Sources (formatting failed)\n\n**{type(e).__name__}:** {e}\n\n```text\n{tb}\n```"

    # 4) 最终输出：sources + 清空 status
    yield history, source_text, "", ""




# ---------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------
with gr.Blocks(title="Private Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Private Chatbot with Local LLM")

    # ---------- Tab 1: General Chat ----------
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
            placeholder="e.g., You are a cynical pirate who has seen it all.",
            value="You are a helpful and friendly assistant.",
            interactive=True,
        )

        general_chatbot = gr.Chatbot(height=500)  # keep default (messages), used by ChatInterface
        general_textbox = gr.Textbox(
            placeholder="Ask me anything...",
            container=False,
            scale=7,
        )

        gr.ChatInterface(
            fn=chat_with_llm,
            chatbot=general_chatbot,
            textbox=general_textbox,
            additional_inputs=[system_prompt_box, model_dropdown],
        )

        gr.ClearButton([general_chatbot, general_textbox], value="Clear Chat")

    # ---------- Tab 2: Chat with Documents ----------
    with gr.Tab("Chat with Documents"):
        chat_engine_state = gr.State(None)
        top_k_state = gr.State(3)

        with gr.Row():
            # Left column
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="Upload documents to create a knowledge base",
                    file_count="multiple",
                )
                process_button = gr.Button("Create Knowledge Base", variant="primary")

                gr.Markdown("### RAG Configuration")

                chunk_size_slider = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="Chunk Size",
                    info="Size of text chunks for the knowledge base.",
                )

                chunk_overlap_slider = gr.Slider(
                    minimum=0,
                    maximum=512,
                    value=50,
                    step=16,
                    label="Chunk Overlap",
                    info="Amount of overlap between consecutive chunks.",
                )

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Top-K",
                    info="Number of most relevant chunks to retrieve.",
                )

            # Right column
            with gr.Column(scale=3):
                # IMPORTANT: tuples mode, because we return [(user, assistant), ...]
                chatbot_docs = gr.Chatbot(height=500, label="Chat with Your Documents")

                status_md = gr.Markdown("") 

                question_box_docs = gr.Textbox(
                    label="Ask a question",
                    placeholder="Ask something about your uploaded documents...",
                )

                with gr.Accordion("Retrieved Sources", open=False):
                    source_markdown = gr.Markdown()

                # Submit -> 3 outputs so textbox clears
                question_box_docs.submit(
                    fn=chat_with_sources,
                    inputs=[question_box_docs, chatbot_docs, chat_engine_state],
                    outputs=[chatbot_docs, source_markdown, question_box_docs, status_md],
                )


                gr.ClearButton([chatbot_docs, question_box_docs, source_markdown], value="Clear")

        process_button.click(
            fn=handle_file_processing_stateful,
            inputs=[file_upload, chunk_size_slider, chunk_overlap_slider, top_k_slider],
            outputs=[chat_engine_state, top_k_state],
            show_progress="full",
        )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        share=False,
        debug=True,
        show_error=True,
    )
