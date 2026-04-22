from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# We store SQLite in ./db_data, ensure it exists or SQLAlchemy will create it
os.makedirs("./db_data", exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db_data/adaptive_rag.db")

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
