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
        Upload and process a PDF book for an admin user.
        
        Args:
            email: Admin's email address
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
                admin_email=email,  # Changed from email to admin_email
                book_name=filename,
                file_path=file_path,
                page_count=page_count,
                processed=False,
                is_public=True  # Default to public
            )
            self.db.add(book)
            self.db.commit()
            self.db.refresh(book)
            
            # Process text into chunks and store in vector database
            await self._process_and_store_chunks(book_id, email, text_content, filename)
            
            # Mark book as processed
            book.processed = True
            self.db.commit()
            
            logger.info(f"Successfully processed book {filename} for admin {email}")
            return book_id
            
        except Exception as e:
            logger.error(f"Error processing book {filename}: {e}")
            # Clean up on error
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            self.db.rollback()
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
        Retrieve relevant context from all public books and admin's books.
        """
        try:
            collection = self.get_or_create_collection(email)
            
            # Get access to public books and admin's books
            books = self.db.query(Book).filter(
                (Book.is_public == True) | (Book.admin_email == email)
            ).all()
            
            # Get query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search for similar chunks
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            relevant_chunks = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    book_id = metadata.get('book_id')
                    # Only include chunks from accessible books
                    if any(book.id == book_id for book in books):
                        relevance_score = 1 - distance
                        chunk_data = {
                            'text': doc,
                            'book_name': metadata.get('book_name', 'Unknown'),
                            'page_number': metadata.get('page_number', 0),
                            'chapter_name': metadata.get('chapter_name'),
                            'section_name': metadata.get('section_name'),
                            'relevance_score': relevance_score
                        }
                        relevant_chunks.append(chunk_data)
            
            return sorted(relevant_chunks, key=lambda x: -x['relevance_score'])[:n_results]
            
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
            
                
            system_prompt = """
            You are an AI that ONLY uses information from uploaded books. This is your most critical rule.

            STRICT REQUIREMENTS:
            1. NEVER use any external knowledge or information not present in the provided book excerpts
            2. NEVER make assumptions or fill gaps in knowledge
            3. If the books don't contain relevant information, explicitly state this
            4. Only cite and use information from the specific excerpts provided
            5. Do not draw from general knowledge, even if it seems relevant
            7. Every single statement must have a book citation
            8. Do not provide book links beside the citations

            When you cannot answer:
            - Clearly state that the uploaded books don't contain information to answer this question
            - Don't try to be helpful with general knowledge
            - Don't make educated guesses
            - Simply acknowledge the limitation

            Format your response:
            1. First, state your confidence in answering based on available book content
            2. Only proceed if you have relevant book citations
            3. Each statement must link directly to a book citation
            4. Conclude with clear references to book sources
            5. Citation format: [Book Title | Chapter Name/Number | Page X]

            If you're unsure or the books don't cover it, respond with:
            "I cannot provide an answer as this topic is not covered in the uploaded books. I am only able to share information directly from the books in the knowledge base."
        """

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

    def _cleanup_orphaned_vectors(self, email: str) -> None:
        """Clean up any orphaned vectors that don't have corresponding database records."""
        try:
            collection = self.get_or_create_collection(email)
            
            # Get all valid embedding IDs from database
            valid_chunks = (
                self.db.query(BookChunk)
                .join(Book)
                .filter(
                    (Book.admin_email == email) | (Book.is_public == True)
                )
                .all()
            )
            valid_embedding_ids = {chunk.embedding_id for chunk in valid_chunks if chunk.embedding_id}
            
            # Get all vectors from ChromaDB
            try:
                all_vectors = collection.get()
                if all_vectors and all_vectors['ids']:
                    vector_ids = set(all_vectors['ids'])
                    
                    # Find orphaned vectors (exist in ChromaDB but not in database)
                    orphaned_ids = vector_ids - valid_embedding_ids
                    
                    if orphaned_ids:
                        # Delete orphaned vectors in batches
                        batch_size = 100
                        orphaned_list = list(orphaned_ids)
                        
                        for i in range(0, len(orphaned_list), batch_size):
                            batch = orphaned_list[i:i + batch_size]
                            collection.delete(ids=batch)
                        
                        logger.info(f"Cleaned up {len(orphaned_ids)} orphaned vectors for user {email}")
                    
            except Exception as e:
                logger.error(f"Error during vector cleanup: {e}")
                
        except Exception as e:
            logger.error(f"Error during orphaned vector cleanup: {e}")
            # Don't raise - this is a cleanup operation

    def reset_user_collection(self, email: str) -> Dict[str, Any]:
        """Reset/clear all vectors for a specific user."""
        try:
            collection_name = f"user_{email.replace('@', '_').replace('.', '_')}"
            
            # Delete the entire collection
            try:
                self.chroma_client.delete_collection(name=collection_name)
                logger.info(f"Deleted collection {collection_name}")
            except Exception as e:
                logger.warning(f"Collection {collection_name} might not exist: {e}")
            
            # Recreate empty collection
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"user_email": email}
            )
            
            return {
                "status": "success",
                "message": f"Reset collection for user {email}",
                "collection_name": collection_name
            }
            
        except Exception as e:
            logger.error(f"Error resetting user collection: {e}")
            raise AIAssistantError(f"Failed to reset collection: {e}")

    def cleanup_all_orphaned_vectors(self, email: str) -> Dict[str, Any]:
        """Clean up all orphaned vectors for a user."""
        try:
            collection = self.get_or_create_collection(email)
            
            # Get all valid embedding IDs from database
            valid_chunks = (
                self.db.query(BookChunk)
                .join(Book)
                .filter(
                    (Book.admin_email == email) | (Book.is_public == True)
                )
                .all()
            )
            valid_embedding_ids = {chunk.embedding_id for chunk in valid_chunks if chunk.embedding_id}
            
            # Get all vectors from ChromaDB
            all_vectors = collection.get()
            vector_ids = set(all_vectors['ids']) if all_vectors and all_vectors['ids'] else set()
            
            # Find orphaned vectors
            orphaned_ids = vector_ids - valid_embedding_ids
            
            if orphaned_ids:
                # Delete orphaned vectors in batches
                batch_size = 100
                orphaned_list = list(orphaned_ids)
                deleted_count = 0
                
                for i in range(0, len(orphaned_list), batch_size):
                    batch = orphaned_list[i:i + batch_size]
                    try:
                        collection.delete(ids=batch)
                        deleted_count += len(batch)
                    except Exception as e:
                        logger.error(f"Error deleting batch: {e}")
                
                logger.info(f"Cleaned up {deleted_count} orphaned vectors")
            
            return {
                "total_vectors": len(vector_ids),
                "valid_vectors": len(valid_embedding_ids),
                "orphaned_vectors": len(orphaned_ids),
                "deleted_count": len(orphaned_ids) if orphaned_ids else 0
            }
            
        except Exception as e:
            logger.error(f"Error cleaning orphaned vectors: {e}")
            raise AIAssistantError(f"Failed to cleanup vectors: {e}")

    def get_collection_stats(self, email: str) -> Dict[str, Any]:
        """Get statistics about a user's collection."""
        try:
            collection = self.get_or_create_collection(email)
            
            # Get collection info
            all_vectors = collection.get()
            vector_count = len(all_vectors['ids']) if all_vectors and all_vectors['ids'] else 0
            
            # Get database info
            books = self.db.query(Book).filter(
                (Book.admin_email == email) | (Book.is_public == True)
            ).all()
            
            chunks = self.db.query(BookChunk).filter(
                BookChunk.book_id.in_([book.id for book in books])
            ).all()
            
            return {
                "collection_name": f"user_{email.replace('@', '_').replace('.', '_')}",
                "vector_count": vector_count,
                "books_count": len(books),
                "chunks_count": len(chunks),
                "books": [{"id": book.id, "name": book.book_name} for book in books]
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise AIAssistantError(f"Failed to get stats: {e}")

    def cleanup_chroma_db_standalone():
        """Standalone script to clean ChromaDB - run this separately."""
        import chromadb
        from chromadb.config import Settings
        import os
        import shutil
        
        # Option 1: Delete entire ChromaDB directory
        chroma_dir = "./chroma_db"
        if os.path.exists(chroma_dir):
            try:
                shutil.rmtree(chroma_dir)
                print(f"Deleted entire ChromaDB directory: {chroma_dir}")
                
                # Recreate empty directory
                os.makedirs(chroma_dir, exist_ok=True)
                print("Created new empty ChromaDB directory")
            except Exception as e:
                print(f"Error deleting ChromaDB directory: {e}")
        
        # Option 2: Delete specific collections
        try:
            client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )
            
            # List all collections
            collections = client.list_collections()
            print(f"Found {len(collections)} collections")
            
            for collection in collections:
                try:
                    client.delete_collection(name=collection.name)
                    print(f"Deleted collection: {collection.name}")
                except Exception as e:
                    print(f"Error deleting collection {collection.name}: {e}")
                    
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")

    def sync_and_cleanup_database():
        """Sync database and vector store, removing inconsistencies."""
        from app.database import get_db
        from app.models import Book, BookChunk
        
        db = next(get_db())
        
        try:
            # Get all users who have books
            users = db.query(Book.admin_email).distinct().all()
            
            for (email,) in users:
                print(f"Cleaning up for user: {email}")
                
                ai_assistant = AIAssistant(db)
                
                # Reset user collection
                result = ai_assistant.reset_user_collection(email)
                print(f"Reset result: {result}")
                
                # Reprocess all books for this user
                books = db.query(Book).filter(Book.admin_email == email).all()
                
                for book in books:
                    if book.processed:
                        print(f"Reprocessing book: {book.book_name}")
                        
                        # Extract text again
                        try:
                            text_content, _ = ai_assistant._extract_pdf_text(book.file_path)
                            
                            # Reprocess and store chunks
                            ai_assistant._process_and_store_chunks(
                                book.id, email, text_content, book.book_name
                            )
                            
                            print(f"Successfully reprocessed: {book.book_name}")
                        except Exception as e:
                            print(f"Error reprocessing {book.book_name}: {e}")
            
            print("Database sync completed")
            
        except Exception as e:
            print(f"Error during sync: {e}")
        finally:
            db.close()

    def quick_reset_chroma():
        """Quick reset - deletes ChromaDB and recreates empty structure."""
        import os
        import shutil
        
        chroma_path = "./chroma_db"
        
        print("Stopping application first...")
        
        # Delete ChromaDB directory
        if os.path.exists(chroma_path):
            try:
                shutil.rmtree(chroma_path)
                print(f"✅ Deleted ChromaDB directory: {chroma_path}")
            except Exception as e:
                print(f"❌ Error deleting directory: {e}")
                return False
        
        # Recreate directory
        try:
            os.makedirs(chroma_path, exist_ok=True)
            print(f"✅ Created new ChromaDB directory: {chroma_path}")
            
            # Test initialization
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            print("✅ ChromaDB initialized successfully")
            print("🔄 Restart your application to begin fresh")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating new ChromaDB: {e}")
            return False
    
    def verify_knowledge_base_consistency(self, email: str) -> dict:
        """
        Verify the consistency between application database and vector store
        
        Args:
            email: Admin email to check books for
            
        Returns:
            Dict containing consistency check results
        """
        try:
            # Get all books from application database
            db_books = (self.db.query(Book)
                    .filter((Book.admin_email == email) | (Book.is_public == True))
                    .all())
            db_book_ids = set(str(book.id) for book in db_books)
            
            # Get all chunks from database
            db_chunks = (self.db.query(BookChunk)
                        .join(Book)
                        .filter((Book.admin_email == email) | (Book.is_public == True))
                        .all())
            chunk_book_ids = set(str(chunk.book_id) for chunk in db_chunks)
            
            # Get embeddings from vector store using get_or_create_collection
            collection = self.get_or_create_collection(email)
            vector_ids = set()
            if collection is not None:
                all_vectors = collection.get()
                if all_vectors and all_vectors['ids']:
                    vector_ids = set(str(id) for id in all_vectors['ids'])
            
            # Check consistency
            is_consistent = (db_book_ids == chunk_book_ids) and all(
                chunk.embedding_id in vector_ids for chunk in db_chunks if chunk.embedding_id
            )
            
            return {
                "is_consistent": is_consistent,
                "details": {
                    "database_books": len(db_book_ids),
                    "database_chunks": len(db_chunks),
                    "vector_embeddings": len(vector_ids),
                    "orphaned_chunks": len(chunk_book_ids - db_book_ids),
                    "missing_embeddings": sum(1 for chunk in db_chunks 
                                        if chunk.embedding_id and chunk.embedding_id not in vector_ids)
                }
            }
            
        except Exception as e:
            logger.error(f"Error verifying knowledge base consistency: {str(e)}")
            return {
                "is_consistent": False,
                "details": {
                    "error": str(e)
                }
            }
    def delete_books(self, email: str, book_ids: List[str]) -> int:
        """
        Delete books and their associated data
        
        Args:
            email: Admin email requesting deletion
            book_ids: List of book IDs to delete
            
        Returns:
            Number of books deleted
        """
        deleted_count = 0
        
        try:
            # Get collection once for all operations
            collection = self.get_or_create_collection(email)
            
            for book_id in book_ids:
                # Verify book exists and user has permission
                book = (self.db.query(Book)
                    .filter(Book.id == book_id, Book.admin_email == email)
                    .first())
                
                if not book:
                    continue
                    
                # Delete physical file
                if os.path.exists(book.file_path):
                    os.remove(book.file_path)
                    logger.info(f"Deleted physical file: {book.file_path}")
                
                # Get chunks to delete from vector store
                chunks = self.db.query(BookChunk).filter(BookChunk.book_id == book_id).all()
                chunk_ids = [chunk.id for chunk in chunks]
                embedding_ids = [chunk.embedding_id for chunk in chunks if chunk.embedding_id]
                
                # Delete from vector store
                if collection is not None and embedding_ids:
                    collection.delete(ids=embedding_ids)
                
                # Delete chunks from database
                if chunk_ids:
                    self.db.query(BookChunk).filter(BookChunk.id.in_(chunk_ids)).delete()
                
                # Delete book record
                self.db.query(Book).filter(Book.id == book_id).delete()
                deleted_count += 1
                
                logger.info(f"Deleted book {book.book_name} (ID: {book_id})")
                
            self.db.commit()
            return deleted_count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting books: {str(e)}")
            raise AIAssistantError(f"Failed to delete books: {str(e)}")