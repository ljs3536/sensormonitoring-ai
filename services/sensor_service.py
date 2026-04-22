# services/sensor_service.py
import datetime
from sqlalchemy.orm import Session
from sensors import Sensor
from sqlalchemy.sql import func

class SensorService:
    @staticmethod
    def get_sensor_meta(db: Session, sensor_id: str):
        sensor_record = db.query(Sensor).filter(Sensor.id == sensor_id).first()
        if sensor_record:
            return {
                "sampling_rate": sensor_record.sampling_rate,
                "k": sensor_record.physics_k or 0.5,
                "c": sensor_record.physics_c or 0.01
            }
        return None

    @staticmethod
    def get_all_sensor_metadata(db: Session, sensor_type: str) -> dict:
        """해당 타입의 모든 센서 물리 정보를 딕셔너리로 반환 (학습용)"""
        sensors = db.query(Sensor).filter(Sensor.type == sensor_type).all()
        meta_map = {}
        for s in sensors:
            meta_map[s.id] = {
                "k": float(s.physics_k) if s.physics_k else 0.5,
                "c": float(s.physics_c) if s.physics_c else 0.01,
                "sampling_rate": s.sampling_rate or 1000
            }
        return meta_map
    
    @staticmethod
    def update_recommended_params(db: Session, sensor_id: str, rec_k: float, rec_c: float, rec_thresh: float):
        sensor = db.query(Sensor).filter(Sensor.id == sensor_id).first()
        if sensor:
            sensor.recommended_k = rec_k
            sensor.recommended_c = rec_c
            sensor.recommended_threshold = rec_thresh # 🌟 추가
            sensor.last_calibrated_at = func.now()
            db.commit()
            return True
        return False