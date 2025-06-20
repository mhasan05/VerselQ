import os
import logging
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile
import shutil

from fastapi import UploadFile
from sqlalchemy.orm import Session
import openai
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import PyPDF2
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from dotenv import load_dotenv

from app.models import User, ChatSession, ChatMessage, Book, BookChunk

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.download('punkt_tab')
except LookupError:
    nltk.download('punkt')
    

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AIAssistantError(Exception):
    """Custom exception for AI Assistant errors"""
    pass

class AIAssistant:
    def __init__(self, db_session: Session):
        """
        Initialize the AI Assistant with database session and necessary services.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        
        # Initialize OpenAI client
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise AIAssistantError("OpenAI API key not found in environment variables")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise AIAssistantError(f"Failed to initialize embedding model: {e}")
        
        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise AIAssistantError(f"Failed to initialize vector database: {e}")
        
        # Directory for storing uploaded PDFs
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)

    def get_or_create_collection(self, email: str):
        """
        Get or create a ChromaDB collection for a specific user.
        
        Args:
            email: User's email address
            
        Returns:
            ChromaDB collection object
        """
        collection_name = f"user_{email.replace('@', '_').replace('.', '_')}"
        try:
            return self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"user_email": email}
            )
        except Exception as e:
            logger.error(f"Error accessing collection for {email}: {e}")
            raise AIAssistantError(f"Failed to access user collection: {e}")

    async def upload_and_process_book(self, email: str, file: UploadFile, filename: str) -> str:
        """
        Upload and process a PDF book for a user.
        
        Args:
            email: User's email address
            file: Uploaded PDF file
            filename: Original filename
            
        Returns:
            str: Book ID of the processed book
        """
        try:
            # Generate unique book ID and file path
            book_id = str(uuid.uuid4())
            file_path = os.path.join(self.upload_dir, f"{book_id}_{filename}")
            
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Extract text from PDF
            text_content, page_count = self._extract_pdf_text(file_path)
            
            # Create book record in database
            book = Book(
                id=book_id,
                email=email,
                book_name=filename,
                file_path=file_path,
                page_count=page_count,
                processed=False
            )
            self.db.add(book)
            self.db.commit()
            self.db.refresh(book)
            
            # Process text into chunks and store in vector database
            await self._process_and_store_chunks(book_id, email, text_content, filename)
            
            # Mark book as processed
            book.processed = True
            self.db.commit()
            
            logger.info(f"Successfully processed book {filename} for user {email}")
            return book_id
            
        except Exception as e:
            logger.error(f"Error processing book {filename}: {e}")
            # Clean up on error
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            raise AIAssistantError(f"Failed to process book: {e}")

    def _extract_pdf_text(self, file_path: str) -> tuple[str, int]:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            tuple: (extracted_text, page_count)
        """
        try:
            text_content = ""
            page_count = 0
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content += f"\n[Page {page_num}]\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        continue
            
            if not text_content.strip():
                raise AIAssistantError("No text could be extracted from the PDF")
            
            return text_content, page_count
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise AIAssistantError(f"Failed to extract text from PDF: {e}")

    async def _process_and_store_chunks(self, book_id: str, email: str, text_content: str, book_name: str):
        """
        Process text content into chunks and store in vector database.
        
        Args:
            book_id: Unique book identifier
            email: User's email address
            text_content: Extracted text from the book
            book_name: Name of the book
        """
        try:
            # Get user's collection
            collection = self.get_or_create_collection(email)
            
            # Split text into chunks
            chunks = self._create_text_chunks(text_content)
            
            # Process each chunk
            chunk_texts = []
            chunk_embeddings = []
            chunk_metadatas = []
            chunk_ids = []
            
            for i, chunk_data in enumerate(chunks):
                chunk_text = chunk_data['text']
                page_number = chunk_data.get('page_number')
                
                # Generate embedding
                embedding = self.embedding_model.encode(chunk_text).tolist()
                
                # Create unique ID for the chunk
                chunk_embedding_id = f"{book_id}_chunk_{i}"
                
                # Prepare metadata
                metadata = {
                    "book_id": book_id,
                    "book_name": book_name,
                    "chunk_index": i,
                    "page_number": page_number or 0,
                    "user_email": email
                }
                
                chunk_texts.append(chunk_text)
                chunk_embeddings.append(embedding)
                chunk_metadatas.append(metadata)
                chunk_ids.append(chunk_embedding_id)
                
                # Store chunk in database
                book_chunk = BookChunk(
                    book_id=book_id,
                    chunk_text=chunk_text,
                    page_number=page_number,
                    chunk_index=i,
                    embedding_id=chunk_embedding_id
                )
                self.db.add(book_chunk)
            
            # Store in vector database
            collection.add(
                documents=chunk_texts,
                embeddings=chunk_embeddings,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            self.db.commit()
            logger.info(f"Stored {len(chunks)} chunks for book {book_name}")
            
        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            raise AIAssistantError(f"Failed to process book chunks: {e}")

    def _create_text_chunks(self, text_content: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Split text content into overlapping chunks.
        
        Args:
            text_content: The full text content
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        chunks = []
        current_page = 1
        
        # Split by pages first
        page_sections = re.split(r'\[Page (\d+)\]', text_content)
        
        for i in range(1, len(page_sections), 2):
            if i + 1 < len(page_sections):
                page_num = int(page_sections[i])
                page_text = page_sections[i + 1].strip()
                
                if not page_text:
                    continue
                
                # Split page text into sentences for better chunk boundaries
                sentences = sent_tokenize(page_text)
                
                current_chunk = ""
                for sentence in sentences:
                    # Check if adding this sentence would exceed chunk size
                    if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                        # Save current chunk
                        chunks.append({
                            'text': current_chunk.strip(),
                            'page_number': page_num
                        })
                        
                        # Start new chunk with overlap
                        if overlap > 0:
                            words = current_chunk.split()
                            overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
                            current_chunk = " ".join(overlap_words) + " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk += " " + sentence
                
                # Add remaining text as final chunk for this page
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'page_number': page_num
                    })
        
        # If no page markers found, split the entire text
        if not chunks:
            sentences = sent_tokenize(text_content)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'page_number': None
                    })
                    
                    if overlap > 0:
                        words = current_chunk.split()
                        overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
                        current_chunk = " ".join(overlap_words) + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk += " " + sentence
            
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'page_number': None
                })
        
        return chunks
    
    def _build_context_from_chunks(self, chunks: List[Dict]) -> tuple[str, List[Dict]]:
        """
        Build context text and references from relevant chunks.
        
        Args:
            chunks: List of relevant text chunks with metadata
            
        Returns:
            tuple: (context_text, references)
        """
        context_text = ""
        references = []
        
        # Group chunks by book
        books_context = {}
        for chunk in chunks:
            book_name = chunk['book_name']
            if book_name not in books_context:
                books_context[book_name] = []
            books_context[book_name].append(chunk)
        
        # Build context text with clear book separation
        if books_context:
            context_text = "\n\nRelevant information from your uploaded books:\n"
            
            for book_name, book_chunks in books_context.items():
                context_text += f"\n📚 From '{book_name}':\n"
                
                for chunk in book_chunks:
                    section_info = []
                    
                    if chunk.get('chapter'):
                        section_info.append(f"Chapter: {chunk['chapter']}")
                    if chunk.get('page_number'):
                        section_info.append(f"Page {chunk['page_number']}")
                    
                    if section_info:
                        context_text += f"[{' | '.join(section_info)}]\n"
                    
                    context_text += f"{chunk['text']}\n\n"
                    
                    # Add to references
                    references.append({
                        'book_name': book_name,
                        'chapter': chunk.get('chapter'),
                        'page_number': chunk.get('page_number'),
                        'relevance_score': chunk.get('relevance_score', 0)
                    })
        
        # Sort references by relevance score
        references.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return context_text, references
    
    def _retrieve_relevant_context(self, email: str, query: str, n_results: int = 10) -> List[Dict]:
        """
        Retrieve relevant context from the user's knowledge base across all books.
        
        Args:
            email: User's email address
            query: User's query
            n_results: Number of relevant chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            collection = self.get_or_create_collection(email)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search for similar chunks
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            relevant_chunks = []
            seen_books = set()  # Track unique books
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    book_name = metadata.get('book_name', 'Unknown')
                    relevance_score = 1 - distance
                    
                    # Only include if relevance score is good enough
                    if relevance_score > 0.6:
                        # Extract chapter information
                        chapter_info = None
                        chapter_match = re.search(r'Chapter [\d\w]+:?\s*([^\.]+)', doc)
                        if chapter_match:
                            chapter_info = chapter_match.group(1).strip()
                        
                        chunk_data = {
                            'text': doc,
                            'book_name': book_name,
                            'page_number': metadata.get('page_number', 0),
                            'chapter': chapter_info,
                            'relevance_score': relevance_score
                        }
                        
                        relevant_chunks.append(chunk_data)
                        seen_books.add(book_name)
                
                # Sort chunks by relevance score and book
                relevant_chunks.sort(key=lambda x: (-x['relevance_score'], x['book_name']))
                
                # Group chunks by book
                grouped_chunks = []
                for book in seen_books:
                    book_chunks = [c for c in relevant_chunks if c['book_name'] == book]
                    if book_chunks:
                        # Take top 2-3 most relevant chunks per book
                        grouped_chunks.extend(book_chunks[:3])
                
                return grouped_chunks[:n_results]
                
            return []
                
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def process_message(self, session_id: str, query_text: str, email: str) -> str:
        try:
            # Get conversation history
            previous_messages = self.db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.timestamp.desc()).limit(5).all()
            
            # Retrieve relevant context from all books
            relevant_chunks = self._retrieve_relevant_context(email, query_text, n_results=15)
            
            # Build context from chunks
            context_text, references = self._build_context_from_chunks(relevant_chunks)
            
            # Enhanced system prompt for multi-book synthesis
            system_prompt = """You are a professional business consultant AI assistant. Your responses should:
    1. Synthesize insights from ALL provided book sources
    2. Compare and contrast different viewpoints from different books
    3. Format references clearly for each insight:
    [Book Name | Chapter | Page X]: <insight>
    4. Structure response with clear sections
    5. Highlight when multiple books support the same insight
    6. If different books have contrasting views, acknowledge this
    7. Never make up information not found in the provided books"""

            # Prepare messages for OpenAI
            messages = [{"role": "system", "content": system_prompt}]
            
            # Create conversation history using a proper loop
            conversation_history = []
            for msg in reversed(previous_messages):
                conversation_history.extend([
                    {"role": "user", "content": msg.query_text},
                    {"role": "assistant", "content": msg.response_text}
                ])
            
            # Add conversation history to messages (last 6 messages)
            messages.extend(conversation_history[:6])
            
            # Add current query with context
            user_message = f"{query_text}\n\nPlease synthesize insights from these sources:{context_text}"
            messages.append({"role": "user", "content": user_message})
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Add references section
            if references:
                ai_response = self._add_references_section(ai_response, references)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise AIAssistantError(f"Failed to process message: {e}")
    
    def update_session_title(self, session_id: str, query_text: str) -> None:
        """
        Update the chat session title based on the first message.
        
        Args:
            session_id: ID of the session to update
            query_text: First message text to base title on
            
        Raises:
            AIAssistantError: If update fails
        """
        try:
            # Generate a concise title from the query
            title = (
                query_text[:47] + "..." 
                if len(query_text) > 50 
                else query_text
            )
            
            # Update session title
            session = self.db.query(ChatSession).filter(
                ChatSession.id == session_id
            ).first()
            
            if session:
                session.title = title
                session.last_updated = datetime.utcnow()
                self.db.commit()
                logger.info(f"Updated title for session {session_id}: {title}")
            else:
                raise AIAssistantError(f"Session not found: {session_id}")
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update session title: {str(e)}")
            raise AIAssistantError(f"Failed to update session title: {str(e)}")
        
    def delete_books(self, email: str, book_ids: List[str]) -> int:
        """
        Delete books and their associated data from the knowledge base.
        
        Args:
            email: User's email address
            book_ids: List of book IDs to delete
            
        Returns:
            Number of books successfully deleted
        """
        try:
            collection = self.get_or_create_collection(email)
            deleted_count = 0
            
            for book_id in book_ids:
                # Get book from database
                book = self.db.query(Book).filter(
                    Book.id == book_id,
                    Book.email == email
                ).first()
                
                if book:
                    # Delete from vector database
                    try:
                        # Get all chunk IDs for this book
                        chunk_ids = [f"{book_id}_chunk_{i}" for i in range(100)]  # Estimate max chunks
                        collection.delete(ids=chunk_ids)
                    except Exception as e:
                        logger.warning(f"Error deleting from vector DB: {e}")
                    
                    # Delete physical file
                    if os.path.exists(book.file_path):
                        os.remove(book.file_path)
                    
                    # Delete from database (cascades to chunks)
                    self.db.delete(book)
                    deleted_count += 1
            
            self.db.commit()
            logger.info(f"Successfully deleted {deleted_count} books for {email}")
            return deleted_count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting books: {e}")
            raise AIAssistantError(f"Failed to delete books: {e}")