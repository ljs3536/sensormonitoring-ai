# sensor-ai/routers/proto_router.py
from fastapi import APIRouter, BackgroundTasks, Depends, Body, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
import numpy as np
import datetime
from sqlalchemy import func
# 프로젝트 구조에 맞게 Import 경로를 수정해 주세요.
from database_rdb import get_db, SensorData, SessionLocal, ModelRegistry, PredictionLog
from architectures.prototypical import PrototypicalLeakDetector 
from architectures.fewshot_prototypical import FewShotPrototypicalDetector
import torch
from sklearn.decomposition import PCA
from services.proto_service import train_proto_model_internal
router = APIRouter(prefix="/ai", tags=["Proto"])

class LeakPredictRequest(BaseModel):
    features: List[List[float]] 
    file_path: str
    model_type: str


@router.post("/proto/train")
async def train_leak_model(sensor_id: str, model_type: str, update_mode: str, days: int = 7, auto_activate: bool = True, background_tasks: BackgroundTasks = None):
    
    background_tasks.add_task(train_proto_model_internal, sensor_id, model_type,update_mode, days, auto_activate)
    return {"message": "누출 모델(Prototypical) 학습이 시작되었습니다."}


@router.post("/proto/predict")
async def predict_leak_model(payload: LeakPredictRequest):
    try:
        X_test = np.array(payload.features) 
        
        # 백엔드에서 넘겨준 model_type에 따라 아키텍처 선택
        if payload.model_type == "all":
            detector = PrototypicalLeakDetector()
        elif payload.model_type == "few":
            detector = FewShotPrototypicalDetector()
        else:
            raise ValueError("알 수 없는 모델 타입입니다.")

        # 🌟 백엔드가 넘겨준 정확한 파일 경로로 모델 로드!
        detector.load(payload.file_path) 
        
        results = detector.predict(X_test)
        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")