import uuid
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Integer
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    status = Column(String, default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class VoiceProfile(Base):
    __tablename__ = "voice_profiles"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    model_version = Column(String, default="1.0")
    embedding_version = Column(String, default="1.0")
    embedding = Column(String, nullable=True)
    profile_hash = Column(String, nullable=True)
    blockchain_tx_id = Column(String, nullable=True)
    quality_score = Column(Float, default=0.0)
    status = Column(String, default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    revoked_at = Column(DateTime(timezone=True), nullable=True)

class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    call_type = Column(String, default="cellular")
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)
    consent_version = Column(String, default="1.0")
    audio_mode = Column(String, default="speakerphone")
    result = Column(String, nullable=True)
    risk_score = Column(Float, default=0.0)
