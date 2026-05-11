from sqlalchemy import create_engine, Column, Integer, Float, String, Text, CHAR, TIMESTAMP, text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings
from datetime import datetime, timedelta, timezone

engine = create_engine(settings.mariadb_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ModelRegistry(Base):
    __tablename__ = "tb_model_registry"

    MODEL_ID = Column(Integer, primary_key=True, autoincrement=True, index=True)
    MAC_ADDR = Column(String(50), index=True, nullable=False) # 센서 ID
    MODEL_TYPE = Column(String(20), nullable=False)           # 'all' 또는 'few'
    VERSION = Column(Integer, nullable=False)                 # 버전 관리 (1, 2, 3...)
    FILE_PATH = Column(String(200), nullable=False)           # 실제 .pt 파일 경로
    TRAIN_SAMPLES = Column(Integer)                           # 학습에 사용된 데이터 수
    THRESHOLD_MEAN = Column(Float)                            # 정상 거리 평균
    THRESHOLD_STD = Column(Float)                             # 정상 거리 표준편차
    STATUS = Column(String(20), default="CANDIDATE")          # ACTIVE, INACTIVE, CANDIDATE
    REG_DT = Column(DateTime, default=datetime.now)           # 생성일시

KST = timezone(timedelta(hours=9))

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