# sensor-ai/models.py
from sqlalchemy import Column, Integer, String, DateTime
import datetime
from database_rdb import Base

class AiModel(Base):
    __tablename__ = "ai_models"

    id = Column(Integer, primary_key=True, index=True)
    sensor_type = Column(String(50))
    model_type = Column(String(50))
    status = Column(String(20))
    file_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)