# sensor-ai/models.py
from sqlalchemy import Column, Integer, String, DateTime, Boolean
import datetime
from sqlalchemy.sql import func
from database_rdb import Base

class AiModel(Base):
    __tablename__ = "ai_models"

    id = Column(Integer, primary_key=True, index=True)
    sensor_type = Column(String(50))
    model_type = Column(String(50))
    status = Column(String(20))
    file_path = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    # soft delete를 위한 컬럼 추가
    is_deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)