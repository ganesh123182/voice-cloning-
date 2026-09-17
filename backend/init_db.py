from backend.database import engine, Base
from backend.db_models import User, VoiceEnrollment

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")

if __name__ == "__main__":
    init_db()
