from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr
import logging
from datetime import datetime, timedelta
import uuid
import os
from dotenv import load_dotenv

from app.models import Base, User, ChatSession, ChatMessage, Book
from app.ai_assistant import AIAssistant, AIAssistantError
from app.database import get_db, engine

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="PDF Q&A AI Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins=os.getenv("ALLOWED_ORIGINS").split(","),  # In production, replace with specific origins
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://192.168.10.99:5173",
        "http://192.168.10.12:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Pydantic models for request/response
class EmailRequest(BaseModel):
    email: EmailStr

class SessionCreateResponse(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query_text: str
    email: EmailStr

class ChatResponseData(BaseModel):
    response_id: str
    query_text: str
    response_text: str

class ChatResponse(BaseModel):
    session_id: str
    data: List[ChatResponseData]

class SessionInfo(BaseModel):
    title: str
    session_id: str
    session_start: datetime

class SessionsByDateResponse(BaseModel):
    today: Optional[List[SessionInfo]] = []
    yesterday: Optional[List[SessionInfo]] = []
    last_week: Optional[List[SessionInfo]] = []
    last_month: Optional[List[SessionInfo]] = []
    last_year: Optional[List[SessionInfo]] = []

class BookInfo(BaseModel):
    email: str  # admin's email
    book_id: str
    book_name: str
    upload_on: datetime
    author: Optional[str] = None
    category: Optional[str] = None
    is_public: bool = True
    class Config:
        from_attributes = True

class BookListResponse(BaseModel):
    books: List[BookInfo]
    class Config:
        from_attributes = True

class DeleteBooksRequest(BaseModel):
    email: EmailStr
    book_ids: List[str]

@app.post("/ai/chat_session/", response_model=SessionCreateResponse)
async def create_chat_session(request: EmailRequest, db: Session = Depends(get_db)):
    """
    Create a new chat session for a user.
    
    Args:
        request: EmailRequest containing the user's email
        db: Database session
        
    Returns:
        SessionCreateResponse containing the created session ID
    """
    try:
        # Check if user exists, if not create
        user = db.query(User).filter(User.email == request.email).first()
        if not user:
            user = User(email=request.email)
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Created new user with email: {request.email}")
        
        # Create new chat session
        session = ChatSession(
            email=request.email,
            title="New Consultation",
            created_at=datetime.utcnow()
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        logger.info(f"Created new session with ID: {session.id}")
        
        return SessionCreateResponse(session_id=str(session.id))
        
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.post("/ai/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Process a user message and return the AI's response with context.
    If no session_id is provided, creates a new session automatically.
    
    Args:
        request: ChatRequest containing optional session_id, query text and email
        db: Database session
        
    Returns:
        ChatResponse containing the AI's response and session data
    """
    try:
        # Input validation
        if not request.query_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": "Query text cannot be empty"
                }
            )

        session_id = request.session_id
        
        # Handle session creation/validation
        if not session_id:
            # Create or get user
            user = db.query(User).filter(User.email == request.email).first()
            if not user:
                user = User(email=request.email)
                db.add(user)
                db.commit()
                db.refresh(user)
                logger.info(f"Created new user with email: {request.email}")
            
            # Create new chat session with generated title
            initial_title = (
                request.query_text[:47] + "..." 
                if len(request.query_text) > 50 
                else request.query_text
            )
            session = ChatSession(
                email=request.email,
                title=initial_title,
                created_at=datetime.utcnow()
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = str(session.id)
            logger.info(f"Created new session {session_id} for {request.email}")
        else:
            # Validate existing session and ownership
            session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.email == request.email
            ).first()
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "status": "error",
                        "message": "Session not found or access denied"
                    }
                )
        
        # Initialize AI assistant and process message
        ai_assistant = AIAssistant(db)
        response = ai_assistant.process_message(
            session_id=session_id,
            query_text=request.query_text,
            email=request.email
        )
        
        # Store the conversation
        chat_message = ChatMessage(
            session_id=session_id,
            query_text=request.query_text,
            response_text=response,
            timestamp=datetime.utcnow()
        )
        db.add(chat_message)
        db.commit()
        db.refresh(chat_message)
        
        # Update session title if needed
        if not db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id,
            ChatMessage.id != chat_message.id
        ).count():
            ai_assistant.update_session_title(session_id, request.query_text)
        
        return ChatResponse(
            session_id=session_id,
            data=[ChatResponseData(
                response_id=str(chat_message.id),
                query_text=request.query_text,
                response_text=response
            )]
        )
        
    except HTTPException:
        raise
    except AIAssistantError as e:
        logger.error(f"Error processing chat for {request.email}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to process message",
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat for {request.email}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Internal server error",
                "error": "An unexpected error occurred"
            }
        )

@app.get("/ai/all_chat/", response_model=ChatResponse)
async def get_all_chat(session_id: str, db: Session = Depends(get_db)):
    """
    Get all chat messages for a specific session.
    
    Args:
        session_id: The session ID to retrieve messages for
        db: Database session
        
    Returns:
        ChatResponse containing all messages in the session
    """
    try:
        # Validate session exists
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get all messages for the session
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp.asc()).all()
        
        data = [
            ChatResponseData(
                response_id=str(msg.id),
                query_text=msg.query_text,
                response_text=msg.response_text
            )
            for msg in messages
        ]
        
        return ChatResponse(session_id=session_id, data=data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting all chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/ai/all_sessions/", response_model=SessionsByDateResponse)
async def get_all_sessions(email: EmailStr, db: Session = Depends(get_db)):
    """
    Get all sessions for a specific user, grouped by date.
    
    Args:
        email: Email address to get sessions for
        db: Database session
        
    Returns:
        SessionsByDateResponse with sessions grouped by date
    """
    try:
        # Get current time for date calculations
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_start = today_start - timedelta(days=1)
        week_start = today_start - timedelta(days=7)
        month_start = today_start - timedelta(days=30)
        year_start = today_start - timedelta(days=365)
        
        # Query all sessions for the user
        sessions = db.query(ChatSession).filter(
            ChatSession.email == email
        ).order_by(ChatSession.created_at.desc()).all()
        
        # Group sessions by date
        grouped_sessions = {
            "today": [],
            "yesterday": [],
            "last_week": [],
            "last_month": [],
            "last_year": []
        }
        
        for session in sessions:
            session_info = SessionInfo(
                title=session.title,
                session_id=str(session.id),
                session_start=session.created_at
            )
            
            if session.created_at >= today_start:
                grouped_sessions["today"].append(session_info)
            elif session.created_at >= yesterday_start:
                grouped_sessions["yesterday"].append(session_info)
            elif session.created_at >= week_start:
                grouped_sessions["last_week"].append(session_info)
            elif session.created_at >= month_start:
                grouped_sessions["last_month"].append(session_info)
            elif session.created_at >= year_start:
                grouped_sessions["last_year"].append(session_info)
        
        return SessionsByDateResponse(**grouped_sessions)
        
    except Exception as e:
        logger.error(f"Error getting all sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ai/upload/")
async def upload_book(
    email: EmailStr = Form(...),
    file: UploadFile = File(...),
    author: str = Form(None),
    description: str = Form(None),
    category: str = Form(None),
    is_public: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Upload a PDF book (admin only)."""
    try:
        # Verify admin access
        user = db.query(User).filter(User.email == email).first()
        if not user or not user.is_admin:
            raise HTTPException(
                status_code=403,
                detail="Only administrators can upload books"
            )
        
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )
        
        ai_assistant = AIAssistant(db)
        book_id = await ai_assistant.upload_and_process_book(
            email=email,
            file=file,
            filename=file.filename
        )
        
        # Update book metadata
        book = db.query(Book).filter(Book.id == book_id).first()
        if book:
            book.author = author
            book.description = description
            book.category = category
            book.is_public = is_public
            db.commit()
        
        return {
            "status": "success",
            "message": "Book uploaded and processed successfully",
            "book_id": book_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/book_list/", response_model=BookListResponse)
async def get_book_list(email: EmailStr, db: Session = Depends(get_db)):
    """Get list of accessible books (public books and admin's own books)."""
    try:
        books = db.query(Book).filter(
            (Book.is_public == True) | (Book.admin_email == email)
        ).order_by(Book.upload_on.desc()).all()
        
        book_list = [
            BookInfo(
                email=book.admin_email,
                book_id=str(book.id),
                book_name=book.book_name,
                upload_on=book.upload_on,
                author=book.author,
                category=book.category,
                is_public=book.is_public
            )
            for book in books
        ]
        
        return BookListResponse(books=book_list)
        
    except Exception as e:
        logger.error(f"Error getting book list: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/ai/delete_books/")
async def delete_books(request: DeleteBooksRequest, db: Session = Depends(get_db)):
    try:
        ai_assistant = AIAssistant(db)
        
        # Delete books and update knowledge base
        deleted_count = ai_assistant.delete_books(request.email, request.book_ids)
        
        # Verify consistency after deletion
        consistency_check = ai_assistant.verify_knowledge_base_consistency(request.email)
        
        if not consistency_check["is_consistent"]:
            logger.warning(f"Inconsistency detected after deletion for {request.email}: {consistency_check}")
        
        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} book(s)",
            "consistency_check": consistency_check
        }
        
    except AIAssistantError as e:
        logger.error(f"Error deleting books: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to delete books",
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.post("/ai/reset_model/", response_model=Dict[str, Any])
async def reset_model(request: EmailRequest, db: Session = Depends(get_db)):
    """
    Reset the entire model and knowledge base for a user.
    
    Args:
        request: EmailRequest containing user's email
        db: Database session
        
    Returns:
        Dict containing success message and reset statistics
    """
    try:
        # Validate email format
        if not request.email:
            raise HTTPException(
                status_code=400,
                detail="Email is required"
            )
        
        # Initialize AI assistant
        ai_assistant = AIAssistant(db)
        
        # Reset the model and get statistics
        reset_result = ai_assistant.reset_user_model(request.email)
        
        return {
            "status": "success",
            "message": "Model reset successfully. You can now upload new books.",
            "email": request.email,
            "reset_time": datetime.utcnow().isoformat() + "Z",
            "statistics": reset_result["details"]
        }
        
    except AIAssistantError as e:
        logger.error(f"Error resetting model for {request.email}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to reset model",
                "error": str(e)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model reset for {request.email}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Internal server error",
                "error": "An unexpected error occurred during model reset"
            }
        )

@app.get("/ai/search/")
async def search_sessions(q: str, email: EmailStr, db: Session = Depends(get_db)):
    """
    Search sessions by title for a specific user.
    
    Args:
        q: Search query string
        email: User's email address
        db: Database session
        
    Returns:
        List of matching sessions
    """
    try:
        sessions = db.query(ChatSession).filter(
            ChatSession.email == email,
            ChatSession.title.ilike(f"%{q}%")
        ).order_by(ChatSession.created_at.desc()).all()
        
        results = [
            {
                "title": session.title,
                "session_id": str(session.id),
                "session_start": session.created_at.isoformat() + "Z"
            }
            for session in sessions
        ]
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error searching sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)