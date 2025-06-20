from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment variable with a fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:root@localhost:5432/document_assistant"
)

# Create SQLAlchemy engine with proper connection settings
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=True
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

def get_db():
    """
    Get database session.
    
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()