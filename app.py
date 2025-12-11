import gradio as gr
import time
from llama_index.llms.ollama import Ollama


# Initialize the Ollama LLM
# Llama 3 8B is a great starting model.
# It balances performance and resource requirements.
llm = Ollama(model="llama3:8b", request_timeout=60.0)


def chat_with_llm(message, history):
    response = ""
    for r in llm.stream_complete(message):
        response += r.delta or ""
        yield response


# ChatInterface 的 fn 需要 (message, history, *additional_inputs)
def mock_document_response(message, history, file, question):
    if file is not None and question:
        file_name = file.name.split("/")[-1]
        bot_message = (
            f"I have received your file '{file_name}' and your question: "
            f"'{question}'. I am not smart enough to answer yet."
        )
        response = ""
        for char in bot_message:
            response += char
            time.sleep(0.02)
            yield response
    else:
        yield "Please upload a file and ask a question."


with gr.Blocks(title="Private Chatbot") as demo:
    gr.Markdown("# Private Chatbot with Local LLM")

    with gr.Tab("General Chat"):
        # 显式创建组件（不传 type）
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
            # ✅ 删掉 type="messages"
        )

        gr.ClearButton([general_chatbot, general_textbox], value="Clear Chat")

    with gr.Tab("Chat with Documents"):
        file_upload = gr.File(label="Upload your PDF, DOCX, or TXT file")
        question_box = gr.Textbox(label="Ask a question about your document")

        doc_chatbot = gr.Chatbot(height=400)

        # 你这里如果不想让用户输入额外 message，可以保留隐藏
        doc_textbox = gr.Textbox(
            placeholder="Optional message...",
            container=False,
            visible=False,
        )

        gr.ChatInterface(
            fn=mock_document_response,
            additional_inputs=[file_upload, question_box],
            chatbot=doc_chatbot,
            textbox=doc_textbox,
            # ✅ 删掉 type="messages"
        )

        gr.ClearButton([doc_chatbot, file_upload, question_box], value="Clear Doc Chat")


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        share=False,
    )
