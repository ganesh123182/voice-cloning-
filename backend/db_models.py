from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True) # UUID
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    full_name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    
    enrollments = relationship("VoiceEnrollment", back_populates="user")

class VoiceEnrollment(Base):
    __tablename__ = "voice_enrollments"

    enrollment_id = Column(String, primary_key=True, index=True) # UUID
    user_id = Column(String, ForeignKey("users.id"))
    enrollment_version = Column(Integer, default=1)
    model_version = Column(String, default="ecapa-tdnn-voxceleb")
    evidence_hash = Column(String, nullable=False)
    voice_hash = Column(String, nullable=True)
    audio_filepath = Column(String, nullable=True)
    blockchain_tx_hash = Column(String, nullable=True)
    blockchain_network = Column(String, nullable=True)
    status = Column(String, default="ENROLLMENT_CREATED") # BLOCKCHAIN_PENDING, BLOCKCHAIN_CONFIRMED, BLOCKCHAIN_FAILED, NOT_CONFIGURED
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), onupdate=lambda: datetime.datetime.now(datetime.timezone.utc))

    user = relationship("User", back_populates="enrollments")
