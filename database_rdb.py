from sqlalchemy import create_engine, Column, Integer, Float, String, Text, CHAR, TIMESTAMP, text, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings
from datetime import datetime, timedelta, timezone

engine = create_engine(settings.mariadb_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
KST = timezone(timedelta(hours=9))

class ModelRegistry(Base):
    __tablename__ = "tb_model_registry"

    MODEL_ID = Column(Integer, primary_key=True, autoincrement=True)
    MAC_ADDR = Column(String(50), nullable=False)
    MODEL_TYPE = Column(String(20), nullable=False)
    VERSION = Column(Integer, nullable=False)
    FILE_PATH = Column(String(200), nullable=False)
    TRAIN_SAMPLES = Column(Integer)
    THRESHOLD_MEAN = Column(Float)
    THRESHOLD_STD = Column(Float)
    STATUS = Column(String(20), default="CANDIDATE")
    
    # 학습 시점의 상세 평가 지표 (JSON 형식)
    # 예: {"accuracy": 0.98, "max_dist": 1.2, "dist_distribution": [...]}
    EVAL_METRICS = Column(JSON, nullable=True) 

    # MEMO
    MEMO = Column(Text, nullable=True)
    
    REG_DT = Column(DateTime, default=lambda: datetime.now(KST))

# 🌟 신규 생성: 예측 로그 모델
class PredictionLog(Base):
    __tablename__ = "tb_prediction_log"

    LOG_ID = Column(Integer, primary_key=True, autoincrement=True)
    MODEL_ID = Column(Integer, nullable=False) # 💡 나중에 외래키 설정 가능
    MAC_ADDR = Column(String(50), nullable=False)
    PROBABILITY = Column(Float, nullable=False)
    RESULT = Column(CHAR(1), nullable=False)
    REG_DT = Column(DateTime, default=lambda: datetime.now(KST))

# tb_sensor_data 테이블 모델 정의
class SensorData(Base):
    __tablename__ = 'tb_sensor_data'
    SEQ = Column(Integer, primary_key=True, autoincrement=True, comment='시퀀스')
    MAC_ADDR = Column(String(16), primary_key=True, comment='맥주소')
    BATTERY_RMIN = Column(String(50), nullable=False, default="100", comment='배터리잔량')
    SENSOR_DATA = Column(Text, nullable=False, comment='센서자료')
    LEAK_PRBBLT = Column(String(50), nullable=False, default="0", comment='누출확률')
    REG_DT = Column(TIMESTAMP, default=lambda: datetime.now(KST), comment='등록일시')
    LEAK_YN = Column(CHAR(1), default="N", comment='누출여부')

# DB 세션을 가져오는 의존성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()