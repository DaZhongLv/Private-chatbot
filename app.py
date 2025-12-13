import os
import shutil
import gradio as gr
import time
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext, load_index_from_storage


PERSIST_DIR = "./storage"
# chat_engine = None

# ---- 全局 LLM & Embedding 设置 ----
llm = Ollama(model="llama3.2:3b", request_timeout=300.0)
Settings.llm = llm
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")


def get_index(documents=None):
    """
    Loads the index from storage if it exists, otherwise builds it from documents.
    If no documents are provided and the index doesn't exist, returns None.
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



def _extract_text(content):
    # Gradio 6: content 可能是 str 或 [{"type":"text","text":"..."}] 这样的块
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

    # ✅ Gradio 6 的 messages 格式
    if isinstance(history[0], dict):
        for h in history:
            role = h.get("role", "user")
            text = _extract_text(h.get("content")).strip()
            if text:
                msgs.append(ChatMessage(role=role, content=text))
        return msgs

    # ✅ 兜底：老式 tuples/list 格式
    if isinstance(history[0], (list, tuple)) and len(history[0]) == 2:
        for u, a in history:
            if u:
                msgs.append(ChatMessage(role="user", content=str(u)))
            if a:
                msgs.append(ChatMessage(role="assistant", content=str(a)))
        return msgs

    return msgs

def chat_with_llm(message, history, system_prompt):
    # 1) system prompt
    messages = [ChatMessage(role="system", content=system_prompt)]

    # 2) history -> ChatMessage
    messages.extend(_history_to_chatmessages(history))

    # 3) latest user message
    messages.append(ChatMessage(role="user", content=message))

    # 4) stream chat
    response = ""
    for r in llm.stream_chat(messages):
        response += (r.delta or "")
        yield response



def handle_file_processing_stateful(files):
    """
    Processes uploaded files, creates an index, and returns a chat engine.
    这个返回值会被存进 gr.State（每个会话一份）。
    """
    if files is None or len(files) == 0:
        # Gradio 会弹 warning 对话框
        raise gr.Warning("No files uploaded. Please upload documents to create a knowledge base.")

    # Gradio 6: files 是一个 NamedString 列表，没有 .read，只能用 .name 路径
    temp_dir = "temp_docs_for_processing"
    os.makedirs(temp_dir, exist_ok=True)

    temp_paths = []

    
    # 把上传的每个文件复制一份到临时目录
    for f in files:
        src_path = f.name if hasattr(f, "name") else str(f)
        tmp_path = os.path.join(temp_dir, os.path.basename(src_path))

        with open(src_path, "rb") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        temp_paths.append(tmp_path)

    # 用这些临时文件构建文档列表
    loader = SimpleDirectoryReader(input_files=temp_paths)
    documents = loader.load_data()

    # Day 8：每次重新建 index（还不做持久化）
    index = VectorStoreIndex.from_documents(documents)

    # ========= 这里开始改成 Qdrant 版本 =========
    # 1) 连接本地 Qdrant（Docker 容器）
    client = qdrant_client.QdrantClient(host="localhost", port=6333)

    # 2) 建一个 QdrantVectorStore，collection_name 可以自己起，
    #    但后面要用同一个名字才能复用数据
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="private_chatbot_docs",
    )

    # 3) 用这个 vector_store 构建 storage_context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4) 建索引，LlamaIndex 会自动把 embeddings 写入 Qdrant
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    # ========= Qdrant 部分结束 =========

    gr.Info("Knowledge base created successfully! You can now ask questions.")

    # 把 chat_engine 返回给 gr.State
    return index.as_chat_engine(chat_mode="condense_question", verbose=True)


def chat_with_document_stateful(message, history, chat_engine):
    """
    Uses the chat engine stored in gr.State to respond to the user.
    这是给 ChatInterface 用的：输入 (message, history, chat_engine)，输出流式文本。
    """
    if chat_engine is None:
        try:
            client = qdrant_client.QdrantClient(host="localhost", port=6333)
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="private_chatbot_docs",  # 要和 handle_file_processing_stateful 里一致
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 从已有的向量库重建索引
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
            # 如果 Qdrant 还没有这个 collection，就提示用户先创建
            raise gr.Warning(
                f"Knowledge base not ready yet. Please upload documents and click 'Create Knowledge Base'. "
                f"(Details: {e})"
            )

    question = (message or "").strip()
    if not question:
        # 空消息就不回答
        yield ""
        return

    # 和课程要求一样，用 stream_chat 流式输出
    response_stream = chat_engine.stream_chat(question)

    response = ""
    for r in response_stream.response_gen:
        response += r
        yield response



with gr.Blocks(title="Private Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Private Chatbot with Local LLM")

    with gr.Tab("General Chat"):
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

        gr.ChatInterface(
            fn=chat_with_llm,
            chatbot=general_chatbot,
            textbox=general_textbox,
            additional_inputs=[system_prompt_box],
        )


        gr.ClearButton([general_chatbot, general_textbox], value="Clear Chat")

    with gr.Tab("Chat with Documents"):
    # 这个隐藏组件用来存 chat_engine，对每个浏览器会话是独立的
        chat_engine_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="Upload documents to create a knowledge base",
                    file_count="multiple",   # 多文件
                )
                process_button = gr.Button("Create Knowledge Base", variant="primary")

            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=chat_with_document_stateful,
                    additional_inputs=[chat_engine_state],
                    chatbot=gr.Chatbot(height=500, label="Chat with Your Documents"),
                )

        # 按钮：处理文件 -> 返回 chat_engine -> 存进 chat_engine_state
        process_button.click(
            fn=handle_file_processing_stateful,
            inputs=[file_upload],
            outputs=[chat_engine_state],
            show_progress="full",
        )




if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        share=False,
    )
