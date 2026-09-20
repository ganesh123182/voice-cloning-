from backend.database import engine, Base, init_and_migrate_db
from backend.db_models import User, VoiceEnrollment

def init_db():
    init_and_migrate_db()
    print("Database tables created and migrated.")

if __name__ == "__main__":
    init_db()
