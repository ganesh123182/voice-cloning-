from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DB_DIR = os.path.dirname(os.path.abspath(__file__))
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'voice_enrollment.db')}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def init_and_migrate_db():
    from sqlalchemy import text
    Base.metadata.create_all(bind=engine)
    try:
        with engine.connect() as conn:
            res = conn.execute(text("PRAGMA table_info(voice_enrollments)"))
            cols = [row[1] for row in res.fetchall()]
            if cols:
                if "audio_filepath" not in cols:
                    conn.execute(text("ALTER TABLE voice_enrollments ADD COLUMN audio_filepath VARCHAR"))
                if "voice_hash" not in cols:
                    conn.execute(text("ALTER TABLE voice_enrollments ADD COLUMN voice_hash VARCHAR"))
                conn.commit()
    except Exception as e:
        print(f"[DB] Migration check notice: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
