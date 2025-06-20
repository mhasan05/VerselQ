# Business-Assistant-ChatGPT

**AI-powered PDF Q&A Assistant for Business Books**

---

## Overview

Business-Assistant-ChatGPT is a FastAPI-based backend that enables users (especially admins) to upload business books in PDF format, process them into searchable knowledge chunks, and interact with an AI assistant that answers questions **strictly using only the content from the uploaded books**. No external or internet-based knowledge is ever used in answers.

---

## Features

- **Admin Book Upload:** Upload PDF business books, which are processed and chunked for semantic search.
- **Strict Book-Only AI Chat:** The AI answers questions using only the uploaded books. If no answer is found, it clearly states so.
- **Session-based Chat:** Users can create, reuse, and manage chat sessions. Each session keeps its own conversation history.
- **Book Management:** List, search, and delete uploaded books (admin only).
- **Citations & References:** Every AI answer includes exact citations from the books.
- **No Hallucination:** The AI will never make up information or use general knowledge.
- **REST API:** Well-structured endpoints for chat, session management, book upload, and more.
- **CORS Support:** Ready for integration with modern frontends (React, Vue, etc.).

---

## Tech Stack

- **Backend:** FastAPI (Python)
- **Database:** SQLAlchemy (SQLite by default)
- **Vector Store:** ChromaDB
- **Embeddings:** SentenceTransformers
- **AI Model:** OpenAI GPT (via API)
- **PDF Parsing:** PyPDF2
- **Environment:** Python 3.10+

---

## Setup & Installation

1. **Clone the repository**
    ```bash
    git clone <your-repo-url>
    cd Business-Assistant-ChatGPT
    ```

2. **Create a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure environment variables**

    Create a `.env` file in the root directory. Example:
    ```
    JWT_SECRET=your-256-bit-secret
    VECTOR_STORE_PATH=./vector_store
    API_VERSION=1.0.0
    DEBUG_MODE=True
    MAX_UPLOAD_SIZE=52428800
    ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://127.0.0.1:5173
    OPENAI_API_KEY=your-openai-key
    ```

5. **Run database migrations (if using Alembic)**
    ```bash
    alembic upgrade head
    ```

6. **Start the FastAPI server**
    ```bash
    uvicorn app.api:app --reload
    ```

---

## API Endpoints

### Book Management

- `POST /ai/upload/`  
  Upload a PDF book (admin only).

- `GET /ai/book_list/?email=...`  
  List all accessible books for a user.

- `DELETE /ai/delete_books/`  
  Delete one or more books (admin only).

### Chat & Sessions

- `POST /ai/chat_session/`  
  Create a new chat session.

- `POST /ai/chat/`  
  Ask a question. The AI will answer using only uploaded books.

- `GET /ai/all_chat/?session_id=...`  
  Get all chat messages for a session.

- `GET /ai/all_sessions/?email=...`  
  List all sessions for a user, grouped by date.

- `GET /ai/search/?q=...&email=...`  
  Search sessions by title.

### Model & Vector Store

- `POST /ai/reset_model/`  
  Reset the knowledge base for a user.

---

## Usage Notes

- **Strict Book-Only Answers:**  
  The AI will never use internet or general knowledge. If the answer is not in the uploaded books, it will say so.

- **Admin Role:**  
  Only users marked as `is_admin=True` in the database can upload or delete books.

- **Session Reuse:**  
  Blank sessions (with no chat) are automatically reused for new conversations.

- **Citations:**  
  Every answer includes citations in the format:  
  `[Book Title | Chapter Name/Number | Page X]`

---

## Development

- All backend code is in the `app/` directory.
- Main entrypoint: [`app/api.py`](app/api.py)
- Models: [`app/models.py`](app/models.py)
- AI logic: [`app/ai_assistant.py`](app/ai_assistant.py)

---

## License

MIT License

---

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI](https://openai.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [PyPDF2](https://pypdf2.readthedocs.io/)

---

## Contact

For questions or support, please open an issue or contact the maintainer.
