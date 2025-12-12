import os
import shutil
import gradio as gr
import time
from llama_index.llms.ollama import Ollama

# This import is needed to structure the chat history correctly
from llama_index.core.llms import ChatMessage, MessageRole
# Configure the global settings for LlamaIndex
# This sets the LLM and the embedding model for all subsequent operations.
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

# ---- 全局 LLM & Embedding 设置 ----
llm = Ollama(model="llama3:8b", request_timeout=300.0)
Settings.llm = llm

Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")



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


# ChatInterface 的 fn 需要 (message, history, *additional_inputs)
def process_document_and_chat(message, history, file):
    # message = 用户输入的问题
    question = (message or "").strip()

    if not question:
        yield "Please ask a question first."
        return

    if file is None:
        yield "Please upload a document first."
        return

    try:
        # Gradio 6: file 是 NamedString，file.name 是实际文件路径（通常在 /tmp 目录）
        if hasattr(file, "name"):
            src_path = file.name
        else:
            src_path = str(file)

        # ---- 自己建临时目录 & 临时文件（真正要给 LlamaIndex 读的就是这个）----
        temp_dir = "temp_docs"
        os.makedirs(temp_dir, exist_ok=True)

        # 用原文件名复制一份到 temp_docs 里
        temp_file_path = os.path.join(temp_dir, os.path.basename(src_path))
        with open(src_path, "rb") as src, open(temp_file_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        # 用临时文件喂给 SimpleDirectoryReader
        loader = SimpleDirectoryReader(input_files=[temp_file_path])
        documents = loader.load_data()

        # 建索引
        index = VectorStoreIndex.from_documents(documents)

        # 聊天引擎
        chat_engine = index.as_chat_engine(
            chat_mode="condense_question",
            verbose=True,
        )

        # 流式回答
        response_stream = chat_engine.stream_chat(question)
        response = ""
        for r in response_stream.response_gen:
            response += r
            yield response

        # 只删我们自己复制出来的那一份，不动原始文件
        try:
            os.remove(temp_file_path)
        except OSError:
            pass  # 删不掉也没关系，下次还能被覆盖

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
        # 上传文件组件
        file_upload = gr.File(label="Upload your PDF, DOCX, or TXT file")

        # 用 ChatInterface 自己来画聊天框和输入框
        doc_interface = gr.ChatInterface(
            fn=process_document_and_chat,
            additional_inputs=[file_upload],  # 作为第三个参数传给 fn
            chatbot=gr.Chatbot(height=400),
            textbox=gr.Textbox(
                label="Ask a question about your document",
                placeholder="Ask a question about your document",
                container=False,
            ),
        )

        # 清空按钮：清聊天 + 输入框 + 上传文件
        gr.ClearButton(
            [doc_interface.chatbot, doc_interface.textbox, file_upload],
            value="Clear Doc Chat",
        )



if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        share=False,
    )
