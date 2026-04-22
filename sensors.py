from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float
from sqlalchemy.sql import func
from database_rdb import Base

class Sensor(Base): # 클래스명은 파이썬 관례상 PascalCase(대문자 시작)를 추천합니다.
    __tablename__ = "sensors"

    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)
    sampling_rate = Column(Integer)
    
    threshold_min = Column(Float, nullable=True) 
    threshold_max = Column(Float, nullable=True)
    location = Column(String(200))

    physics_k = Column(Float, default=0.5)
    physics_c = Column(Float, default=0.01)
    physics_m = Column(Float, default=1.0)
    ambient_temp = Column(Float, default=25.0)
    
    recommended_k = Column(Float, nullable=True)
    recommended_c = Column(Float, nullable=True)
    recommended_threshold = Column(Float, nullable=True)
    
    is_active = Column(Boolean, default=True)
    last_calibrated_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())