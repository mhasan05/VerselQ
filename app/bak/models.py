from sqlalchemy import Column, String, Boolean, DateTime, JSON, ForeignKey, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """User model for storing user information"""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    books = relationship("Book", back_populates="user", cascade="all, delete-orphan")

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
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")

class Book(Base):
    """Book model for storing uploaded PDF books"""
    __tablename__ = "books"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, ForeignKey("users.email"), nullable=False, index=True)
    book_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)  # Path where the PDF is stored
    upload_on = Column(DateTime, default=datetime.utcnow, index=True)
    processed = Column(Boolean, default=False)  # Whether the book has been processed for RAG
    page_count = Column(Integer, default=0)  # Number of pages in the book
    
    # Relationships
    user = relationship("User", back_populates="books")
    chunks = relationship("BookChunk", back_populates="book", cascade="all, delete-orphan")

class BookChunk(Base):
    """Book chunk model for storing processed text chunks for RAG"""
    __tablename__ = "book_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    book_id = Column(String, ForeignKey("books.id"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    chapter_name = Column(String, nullable=True)
    chunk_index = Column(Integer, nullable=False)  # Order of chunk in the book
    embedding_id = Column(String, nullable=True)  # ID in vector database
    
    # Relationships
    book = relationship("Book", back_populates="chunks")