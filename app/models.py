from sqlalchemy import Column, String, Boolean, DateTime, JSON, ForeignKey, Text, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_admin = Column(Boolean, default=False)
    
    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    books = relationship("Book", back_populates="admin", cascade="all, delete-orphan")

    def is_admin_user(self) -> bool:
        return self.is_admin

class ChatSession(Base):
    """Chat session model for storing conversation sessions"""
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, ForeignKey("users.email"), nullable=False, index=True)
    title = Column(String, default="New Consultation")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    """Chat message model for storing conversation messages"""
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    references = Column(Text)  # Store book references used in the response
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")

class Book(Base):
    """Book model for admin-uploaded documents"""
    __tablename__ = "books"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    admin_email = Column(String, ForeignKey("users.email", ondelete="CASCADE"), nullable=False, index=True)
    book_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_on = Column(DateTime, default=datetime.utcnow, index=True)
    processed = Column(Boolean, default=False)
    page_count = Column(Integer, default=0)
    is_public = Column(Boolean, default=True)
    author = Column(String)  # Added author field
    description = Column(Text)  # Added description field
    category = Column(String)  # Added category field
    
    # Relationships
    admin = relationship("User", back_populates="books")
    chunks = relationship("BookChunk", back_populates="book", cascade="all, delete-orphan")

class BookChunk(Base):
    """Model for storing processed book chunks with embeddings"""
    __tablename__ = "book_chunks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    book_id = Column(String, ForeignKey("books.id", ondelete="CASCADE"))
    chunk_text = Column(Text, nullable=False)
    page_number = Column(Integer)
    chunk_index = Column(Integer)
    embedding_id = Column(String, unique=True)
    chapter_name = Column(String)
    section_name = Column(String)
    relevance_score = Column(Float)  # Added for tracking search relevance
    last_accessed = Column(DateTime)  # Added for tracking usage
    
    # Relationship
    book = relationship("Book", back_populates="chunks")