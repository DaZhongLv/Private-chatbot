import os
import shutil
import gradio as gr
import time
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext, load_index_from_storage


PERSIST_DIR = "./storage"
chat_engine = None

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


def handle_file_upload(file):
    """
    Handles the file upload, creates an index, and initializes the chat engine.
    """
    global chat_engine

    if file is None:
        return "Please upload a file."

    try:
        # Gradio 6: file 是 NamedString，file.name 是实际文件路径
        src_path = file.name if hasattr(file, "name") else str(file)

        # 临时目录：只复制副本，不动原文件
        temp_dir = "temp_docs"
        os.makedirs(temp_dir, exist_ok=True)

        temp_file_path = os.path.join(temp_dir, os.path.basename(src_path))

        # 把 src_path 复制一份到 temp_docs 里
        with open(src_path, "rb") as src, open(temp_file_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        # 用临时目录读取文档（和课程写法一致）
        loader = SimpleDirectoryReader(input_dir=temp_dir)
        documents = loader.load_data()

        # 用 Day 7 的 get_index 来加载 / 构建持久化索引
        index = get_index(documents)

        # 用 index 初始化全局 chat_engine（这是课程要求的关键）
        chat_engine = index.as_chat_engine(
            chat_mode="condense_question",
            verbose=True,
        )

        # 清理我们自己复制的临时文件
        try:
            os.remove(temp_file_path)
        except OSError:
            pass

        return (
            f"File '{os.path.basename(src_path)}' processed successfully. "
            "You can now ask questions."
        )

    except Exception as e:
        return f"An error occurred during file processing: {e}"



def chat_with_document(message, history):
    """
    Handles the chat interaction using the global chat engine.
    这个函数专门给 ChatInterface 用：输入 (message, history)，输出流式文本。
    """
    global chat_engine

    question = (message or "").strip()
    if not question:
        # 没问题就直接提示
        yield "Please ask a question first."
        return

    if chat_engine is None:
        index = get_index()  # 不传 documents，只是尝试从磁盘加载
        if index is not None:
            chat_engine = index.as_chat_engine(
                chat_mode="condense_question",
                verbose=True,
            )
            print("Loaded chat_engine from persisted index.")
        else:
            # 磁盘上也没有 index，那就真的需要先上传文件
            yield "Please upload and process a document first."
            return

    try:
        # 用 chat_engine 做流式对话（和课程一样）
        response_stream = chat_engine.stream_chat(question)

        response = ""
        for r in response_stream.response_gen:
            response += r
            yield response

    except Exception as e:
        yield f"An error occurred: {e}"



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
        with gr.Row():
            file_upload = gr.File(
                label="Upload documents (PDF, DOCX, TXT)"
            )
            process_button = gr.Button("Process Document(s)")
            status_box = gr.Textbox(label="Status", interactive=False)

        # 这里的 chatbot / textbox 交给 ChatInterface 统一管理
        doc_chatbot = gr.Chatbot(height=400, label="Chat")
        doc_textbox = gr.Textbox(
            label="Ask a question",
            placeholder="Ask something about your document...",
            container=False,
        )

        # 文档聊天：用 chat_engine + stream_chat
        gr.ChatInterface(
            fn=chat_with_document,
            chatbot=doc_chatbot,
            textbox=doc_textbox,
        )

        # 处理文件：单独一个按钮调用 handle_file_upload
        process_button.click(
            fn=handle_file_upload,
            inputs=[file_upload],
            outputs=[status_box],
        )




if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        share=False,
    )
