# services/model_service.py
import datetime
from sqlalchemy.orm import Session
from models import AiModel
from typing import List

class ModelService:
    @staticmethod
    def create_training_record(db: Session, sensor_type: str, model_type: str) -> AiModel:
        new_model = AiModel(
            sensor_type=sensor_type,
            model_type=model_type,
            status="TRAINING"
        )
        db.add(new_model)
        db.commit()
        db.refresh(new_model)
        return new_model
    
    @staticmethod
    def get_model(db: Session, model_id: int) -> AiModel:
        return db.query(AiModel).filter(
            AiModel.id == model_id,
            AiModel.is_deleted == False
        ).first()
    
    @staticmethod
    def get_ready_model(db: Session, model_id: int) -> AiModel:
        return db.query(AiModel).filter(
            AiModel.id == model_id,
            AiModel.status == "READY",
            AiModel.is_deleted == False
        ).first()

    @staticmethod
    def get_all_models(db: Session, sensor_type: str = None) -> List[AiModel]:
        # 1. 기본 쿼리: 삭제되지 않은 모델만
        query = db.query(AiModel).filter(AiModel.is_deleted == False)
        
        # 2. 동적 필터: sensor_type이 파라미터로 넘어왔을 때만 조건 추가
        if sensor_type:
            query = query.filter(AiModel.sensor_type == sensor_type)
            
        return query.order_by(AiModel.created_at.desc()).all()
    
    @staticmethod
    def get_latest_pinn_model(db: Session, sensor_type: str) -> AiModel:
        return db.query(AiModel).filter(
            AiModel.sensor_type == sensor_type,
            AiModel.model_type.ilike("pinn_cnnlstmautoencoder"),
            AiModel.status == "READY",
            AiModel.is_deleted == False
        ).order_by(AiModel.created_at.desc()).first()

    @staticmethod
    def soft_delete_model(db: Session, model_id: int):
        model_record = db.query(AiModel).filter(
            AiModel.id == model_id,
            AiModel.is_deleted == False
        ).first()
        
        if not model_record:
            return False
            
        model_record.is_deleted = True
        model_record.deleted_at = datetime.datetime.now()
        model_record.status = "DELETED" 
        db.commit()
        return True

