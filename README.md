# Private Chatbot with Local LLM

This project is a **privacy-preserving RAG chatbot** that runs entirely on your local machine.  
It has two main features:

1. **General Chat** — talk to a local LLM through a simple Gradio UI.
2. **Chat with Documents** — upload PDFs / DOCX / TXT files and ask questions based on their content.
   Document embeddings are stored in a **Qdrant vector database** running in Docker, so the knowledge base
   persists across restarts.

Core stack:

- **Ollama** – runs the local LLMs (e.g. `llama3.2:3b`)
- **LlamaIndex** – handles document ingestion, chunking and retrieval
- **Qdrant** – vector database for persistent embeddings
- **Gradio 6** – web UI for chatting with the model and your documents

---

## Prerequisites / Dependencies

Before running the app, make sure you have:

1. **Python 3.11** (and `venv`)
2. **Ollama** installed and running  
   - Install from the official website  
   - Make sure the model you use in `app.py` is available, e.g.:
     ```bash
     ollama run llama3.2:3b
     ```
3. **Docker Desktop** installed and running  
   This is required to run **Qdrant**, the vector database used for document storage.:contentReference[oaicite:1]{index=1}  

4. **Qdrant Docker container** running locally:  

   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

