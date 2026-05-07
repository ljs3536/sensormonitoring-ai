from sqlalchemy import create_engine, Column, Integer, String, Text, CHAR, TIMESTAMP, text
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

engine = create_engine(settings.mariadb_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# tb_sensor_data 테이블 모델 정의
class SensorData(Base):
    __tablename__ = 'tb_sensor_data'
    SEQ = Column(Integer, primary_key=True, autoincrement=True, comment='시퀀스')
    MAC_ADDR = Column(String(16), primary_key=True, comment='맥주소')
    BATTERY_RMIN = Column(String(50), nullable=False, default="100", comment='배터리잔량')
    SENSOR_DATA = Column(Text, nullable=False, comment='센서자료')
    LEAK_PRBBLT = Column(String(50), nullable=False, default="0", comment='누출확률')
    REG_DT = Column(TIMESTAMP, server_default=text('CURRENT_TIMESTAMP'), comment='등록일시')
    LEAK_YN = Column(CHAR(1), default="N", comment='누출여부')

# DB 세션을 가져오는 의존성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()