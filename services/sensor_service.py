# services/sensor_service.py
import datetime
from sqlalchemy.orm import Session
from sensors import Sensor

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