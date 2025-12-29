# Private Chatbot with Local LLMs and RAG

A secure, offline-first desktop application that allows you to chat with powerful language models and your own documents, ensuring 100% data privacy.

![App Screenshot GIF](./app_demo.gif)

## ✨ Features

- **Local-First AI:** Runs entirely on your machine. No data ever leaves your computer.
- **Model Flexibility:** Supports any model compatible with Ollama (Llama 3, Phi-3, Mistral, etc.). Switch between models with a dropdown menu.
- **Chat with Your Documents:** Upload PDFs, DOCX, or TXT files to create a searchable knowledge base and get answers based on their content.
- **Advanced RAG Pipeline:** Powered by LlamaIndex and a Qdrant vector database for efficient, state-of-the-art retrieval.
- **Configurable & Transparent:** Interactively tune RAG parameters like chunk size and retrieval count. View the source chunks used for each answer to ensure accuracy.
- **Professional UI:** Clean, intuitive interface built with Gradio, featuring knowledge base management and chat history export.


## 🛠️ Tech Stack

- **Language:** Python
- **UI Framework:** Gradio
- **RAG Framework:** LlamaIndex
- **LLM Serving:** Ollama
- **Vector Database:** Qdrant (via Docker)
- **Document Parsers:** `pypdf`, `docx2txt`


## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

- Python 3.9+
- Docker Desktop (must be running)
- Ollama (must be running)

### Installation

. **Clone the repository:**
```bash
git clone [https://github.com/DaZhongLv/private-chatbot.git\](https://github.com/DaZhongLv/private-chatbot.git)
cd private-chatbot
```
. **Create and activate a virtual environment:**
```bash
# For Windows
python -m venv venv
.\\venv\\Scripts\\activate

# For macOS/Linux  
python3 \-m venv venv  
source venv/bin/activate  
```

. **Install dependencies:**
```bash
pip install -r requirements.txt
```
. **Run dependent services:**
- **Start Qdrant:**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
- **Download an Ollama model** (if you haven't already):
```bash
ollama pull llama3:8b
```
. **Run the application:**
```bash
python app.py
```
The application will now be running at `http://127.0.0.1:7860\`.  



