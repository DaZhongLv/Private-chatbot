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

# Qdrant integration:
# We use a local Qdrant vector database (running in Docker) to store document embeddings.
# Compared to local file-based indexes, Qdrant gives us:
#   - persistent storage across restarts
#   - a dedicated, scalable vector search engine
#   - a clean separation between the app (Gradio + LlamaIndex) and the vector store backend

import os
import shutil
import gradio as gr
import time  # currently not used, but kept in case we want timing / logging later
import qdrant_client

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.llms import ChatMessage, MessageRole
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

# Local folder for (optional) file-based LlamaIndex storage.
# Note: in this version we primarily rely on Qdrant for vector storage,
# but this directory can still be used for experiments with file-based indices.
PERSIST_DIR = "./storage"

# Global LLM + embedding configuration for LlamaIndex.
# This tells LlamaIndex which LLM to call and which embedding model to use.
llm = Ollama(model="llama3.2:3b", request_timeout=300.0)
Settings.llm = llm

# Embedding model is loaded via LlamaIndex's helper.
# Using a local BGE model that should be available through your environment.
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------------
# Optional helper: load a file-based index if present (Day 8 style)
# ---------------------------------------------------------------------
def get_index(documents=None):
    """
    Load a VectorStoreIndex from file-based storage if it exists,
    otherwise build it from the provided documents and persist it.

    Parameters
    ----------
    documents : Optional[List[Document]]
        List of LlamaIndex Document objects to build an index from
        if no persisted index is found.

    Returns
    -------
    VectorStoreIndex or None
        - Existing index loaded from PERSIST_DIR, or
        - Newly built index (also persisted), or
        - None if there is no existing index and no documents.
    """
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


# ---------------------------------------------------------------------
# Gradio history <-> LlamaIndex ChatMessage utilities
# ---------------------------------------------------------------------
def _extract_text(content):
    """
    Extract plain text from a Gradio 6 'content' field.

    In Gradio 6, message content can be:
      - a plain string, or
      - a list of blocks like [{"type": "text", "text": "..."}].
    This helper normalizes both cases to a single text string.
    """
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
    """
    Convert a Gradio ChatInterface history object into a list
    of LlamaIndex ChatMessage objects.

    Gradio 6 can represent history in two main ways:
      1. List[dict] with keys {"role": ..., "content": ...}
      2. Legacy list of (user, assistant) tuples.

    This function handles both formats and returns a flat list like:
      [ChatMessage(role="user", ...),
       ChatMessage(role="assistant", ...),
       ...]
    """
    msgs = []
    if not history:
        return msgs

    # New Gradio 6 "messages" format: list of {"role": ..., "content": ...}
    if isinstance(history[0], dict):
        for h in history:
            role = h.get("role", "user")
            text = _extract_text(h.get("content")).strip()
            if text:
                msgs.append(ChatMessage(role=role, content=text))
        return msgs

    # Fallback: legacy list of (user, assistant) tuples
    if isinstance(history[0], (list, tuple)) and len(history[0]) == 2:
        for u, a in history:
            if u:
                msgs.append(ChatMessage(role="user", content=str(u)))
            if a:
                msgs.append(ChatMessage(role="assistant", content=str(a)))
        return msgs

    return msgs


# ---------------------------------------------------------------------
# General Chat (no documents)
# ---------------------------------------------------------------------
def chat_with_llm(message, history, system_prompt):
    """
    Streaming chat handler for the 'General Chat' tab.

    Parameters
    ----------
    message : str
        Latest user message from the textbox.
    history : list
        Chat history as provided by Gradio ChatInterface.
    system_prompt : str
        System prompt text from the "System Prompt" textbox.

    Yields
    ------
    str
        Partial response text chunks, to be streamed back to the UI.
    """
    # 1) Start with system prompt
    messages = [ChatMessage(role="system", content=system_prompt)]

    # 2) Append previous conversation history
    messages.extend(_history_to_chatmessages(history))

    # 3) Append latest user message
    messages.append(ChatMessage(role="user", content=message))

    # 4) Stream the LLM response back to Gradio
    response = ""
    for r in llm.stream_chat(messages):
        response += (r.delta or "")
        yield response


# ---------------------------------------------------------------------
# Document upload & Qdrant-backed knowledge base (stateful)
# ---------------------------------------------------------------------
def handle_file_processing_stateful(files, chunk_size, chunk_overlap):
    """
    Process uploaded files, build a Qdrant-backed index, and return a chat engine.

    This function is called when the user clicks "Create Knowledge Base" in
    the "Chat with Documents" tab. The returned chat_engine object is stored
    in a Gradio State component, giving each browser session its own engine.

    Parameters
    ----------
    files : list[gradio.NamedString] or None
        Uploaded files from the Gradio File component. In Gradio 6, each file
        is represented by a NamedString whose `.name` is the temporary path.

    Returns
    -------
    BaseChatEngine
        A LlamaIndex chat engine configured to use the Qdrant-backed index.
    """
    if files is None or len(files) == 0:
        # Gradio will surface this as a UI warning dialog.
        raise gr.Warning("No files uploaded. Please upload documents to create a knowledge base.")

    # Gradio 6: files is a list of NamedString objects without .read(),
    # so we need to reopen them from their `.name` paths and copy them.
    temp_dir = "temp_docs_for_processing"
    os.makedirs(temp_dir, exist_ok=True)

    temp_paths = []

    Settings.node_parser = SentenceSplitter(
    chunk_size=int(chunk_size),
    chunk_overlap=int(chunk_overlap),
    )

    # Copy each uploaded file into our own temporary directory,
    # so that LlamaIndex can safely read them.
    for f in files:
        src_path = f.name if hasattr(f, "name") else str(f)
        tmp_path = os.path.join(temp_dir, os.path.basename(src_path))

        with open(src_path, "rb") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        temp_paths.append(tmp_path)

    # Load the copied files as LlamaIndex Document objects.
    loader = SimpleDirectoryReader(input_files=temp_paths)
    documents = loader.load_data()

    # NOTE: This was the Day 8 approach (pure in-memory index) and is no longer
    # strictly needed now that we write to Qdrant. We leave it here as a
    # reference for the original implementation.
    # index = VectorStoreIndex.from_documents(documents)

    # --- Qdrant integration (Day 9) ---

    # 1) Connect to the local Qdrant instance (running in Docker).
    client = qdrant_client.QdrantClient(host="localhost", port=6333)

    # 2) Create a QdrantVectorStore. The collection_name identifies the
    #    "knowledge base" for this app. Using the same name later lets us
    #    rebuild the index and chat engine from existing embeddings.
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="private_chatbot_docs",
    )

    # 3) Build a StorageContext that points to Qdrant as the vector store.
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4) Build a VectorStoreIndex on top of the Qdrant vector store.
    #    This call:
    #      - computes embeddings for the documents
    #      - writes them into the Qdrant collection
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    try:
        info = client.get_collection("private_chatbot_docs")
        print("[Qdrant] collection info:", info)
    except Exception as e:
        print("[Qdrant] get_collection failed:", e)

    # Optional: clean up our own temporary copies. The original uploads
    # are managed by Gradio; we only remove the copies in temp_dir.
    for p in temp_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    gr.Info("Knowledge base created successfully! You can now ask questions.")

    # Return a chat engine built on top of this index.
    # ChatInterface will store this object inside gr.State.
    return index.as_chat_engine(chat_mode="condense_question", verbose=True)


def chat_with_document_stateful(message, history, chat_engine):
    """
    Streaming chat handler for the 'Chat with Documents' tab.

    This function uses the chat_engine stored in gr.State. If the engine is
    missing (e.g., after an app restart), it will attempt to reconstruct a
    new engine from the existing Qdrant collection.

    Parameters
    ----------
    message : str
        Latest user question from the textbox.
    history : list
        Conversation history from Gradio ChatInterface (ignored here, since
        LlamaIndex maintains its own internal chat history).
    chat_engine : Optional[BaseChatEngine]
        Engine previously created by handle_file_processing_stateful and stored
        in gr.State, or None at the beginning of a session.

    Yields
    ------
    str
        Partial response text chunks, streamed back to the Gradio UI.
    """
    # If chat_engine is None, try to rebuild it from the Qdrant vector store.
    if chat_engine is None:
        try:
            client = qdrant_client.QdrantClient(host="localhost", port=6333)
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="private_chatbot_docs",  # must match the name used above
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Rebuild the index from the existing Qdrant collection.
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
            )

            chat_engine = index.as_chat_engine(
                chat_mode="condense_question",
                verbose=True,
            )
            print("Rebuilt chat_engine from existing Qdrant collection.")

        except Exception as e:
            # If Qdrant doesn't have this collection yet (or is unavailable),
            # ask the user to create the knowledge base first.
            raise gr.Warning(
                "Knowledge base not ready yet. Please upload documents and click "
                "'Create Knowledge Base'. "
                f"(Details: {e})"
            )

    question = (message or "").strip()
    if not question:
        # Empty input: nothing to answer, but ChatInterface expects a generator.
        yield ""
        return

    # Use LlamaIndex's streaming API to generate the answer.
    response_stream = chat_engine.stream_chat(question)

    response = ""
    for r in response_stream.response_gen:
        response += r
        yield response


# ---------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------
with gr.Blocks(title="Private Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Private Chatbot with Local LLM")

    # ---------- Tab 1: General Chat ----------
    with gr.Tab("General Chat"):
        # User-configurable system prompt for the general chat LLM.
        system_prompt_box = gr.Textbox(
            label="System Prompt",
            placeholder="e.g., You are a cynical pirate who has seen it all.",
            value="You are a helpful and friendly assistant.",
            interactive=True,
        )

        general_chatbot = gr.Chatbot(height=500)
        general_textbox = gr.Textbox(
            placeholder="Ask me anything...",
            container=False,
            scale=7,
        )

        # ChatInterface wires the UI components to the chat_with_llm function.
        gr.ChatInterface(
            fn=chat_with_llm,
            chatbot=general_chatbot,
            textbox=general_textbox,
            additional_inputs=[system_prompt_box],
        )

        gr.ClearButton([general_chatbot, general_textbox], value="Clear Chat")

    # ---------- Tab 2: Chat with Documents ----------
    with gr.Tab("Chat with Documents"):
        # Hidden state object that holds the chat_engine for this browser session.
        chat_engine_state = gr.State(None)

        with gr.Row():
            # Left column: upload + "Create Knowledge Base" button
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="Upload documents to create a knowledge base",
                    file_count="multiple",  # allow multi-document knowledge base
                )
                process_button = gr.Button("Create Knowledge Base", variant="primary")

                gr.Markdown("### RAG Configuration")

                chunk_size_slider = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="Chunk Size",
                    info="Size of text chunks for the knowledge base."
                )

                chunk_overlap_slider = gr.Slider(
                    minimum=0,
                    maximum=512,
                    value=50,
                    step=16,
                    label="Chunk Overlap",
                    info="Amount of overlap between consecutive chunks."
                )

            # Right column: chat UI grounded in the uploaded documents
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=chat_with_document_stateful,
                    additional_inputs=[chat_engine_state],
                    chatbot=gr.Chatbot(height=500, label="Chat with Your Documents"),
                )

        # When the user clicks "Create Knowledge Base":
        #   - handle_file_processing_stateful is called with the uploaded files
        #   - it returns a chat_engine
        #   - Gradio stores that engine value inside chat_engine_state
        process_button.click(
            fn=handle_file_processing_stateful,
            inputs=[file_upload, chunk_size_slider, chunk_overlap_slider],
            outputs=[chat_engine_state],
            show_progress="full",
        )


if __name__ == "__main__":
    # Launch the Gradio app. The theme is configured here (Gradio 6 style).
    demo.launch(
        theme=gr.themes.Soft(),
        share=False,
    )
