# AI PDF Knowledge Base Chatbot Like ChatGpt

This project provides two complementary AI systems:

1. **AI Model** - A document processing system for uploading and querying PDFs with conversation history.
2. **ChatAI** - A ChatGPT-like experience with memory, conversation sessions, and PDF knowledge base integration.

## Overview

- Django-based REST API
- OpenAI integration (using GPT-4o)
- PDF document processing
- Vector storage with FAISS
- JWT authentication

## Key Features

### AI Model (Original System)
- PDF document processing and text extraction
- Vector embeddings for semantic search
- Simple conversation history with the knowledge base
- Session-based memory functionality

### ChatAI (New System)
- ChatGPT-like conversational experience
- Persistent chat sessions with full conversation memory
- PDF knowledge base integration
- Automatic title generation for conversations
- Search functionality across conversation history
- Flexible memory management with context control
- Session reset capabilities

## Installation and Setup

1. Clone the repository
2. Set up a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

5. Run migrations:
   ```
   python manage.py migrate
   ```

6. Start the development server:
   ```
   python manage.py runserver
   ```

## API Endpoints

### Authentication
- `POST /accounts/login/`: Log in and get JWT token
- `POST /accounts/signup/`: Signup

### AI Model (Original System)
- `POST /ai/upload/`: Upload PDFs to the knowledge base
-  `POST /ai/chat_session/`: Create new chat sessions
- `POST /ai/chat/`: Send a question and get an answer from the knowledge base
- `GET /ai/all_sessions/`: Get all chat sessions
- `POST /ai/reset_model/`: Reset the knowledge base (admin only)


## Technologies Used

- Django and Django REST Framework
- OpenAI API
- LangChain for conversational AI
- FAISS for vector storage and similarity search
- PostgreSQL database
- JWT authentication 
