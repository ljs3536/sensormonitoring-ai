from fastapi import APIRouter, BackgroundTasks, Depends, Body, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
import numpy as np
import datetime
from sqlalchemy import func
# 프로젝트 구조에 맞게 Import 경로를 수정해 주세요.
from database_rdb import get_db, SensorData, SessionLocal, ModelRegistry
from architectures.prototypical import PrototypicalLeakDetector 
from architectures.fewshot_prototypical import FewShotPrototypicalDetector
router = APIRouter(prefix="/ai", tags=["Proto"])

class LeakPredictRequest(BaseModel):
    features: List[List[float]] 
    file_path: str
    model_type: str

@router.post("/proto/train")
async def train_leak_model(sensor_id: str, model_type: str, days: int = 7, background_tasks: BackgroundTasks = None):
    def task(s_id, train_days):
        db_session = SessionLocal() 
        try:
            print(f"🚀 [AI] {s_id} 누출 모델 학습 시작...")
            
            # 1. 데이터 조회 및 파싱 (기존 코드와 동일)
            time_threshold = datetime.datetime.now() - datetime.timedelta(days=train_days)
            records = db_session.query(SensorData).filter(
                SensorData.MAC_ADDR == s_id, SensorData.REG_DT >= time_threshold,
                SensorData.LEAK_YN == "N"
            ).all()

            X, y = [], []
            for r in records:
                if r.SENSOR_DATA:
                    X.append([float(val) for val in r.SENSOR_DATA.split('|')])
                    y.append(1 if r.LEAK_YN == 'Y' else 0)
            
            X_train, y_train = np.array(X), np.array(y)
            
            # 🌟 2. 아키텍처 모델 호출 및 학습
            if model_type == "all":
                detector = PrototypicalLeakDetector()
                detector.fit(X_train, y_train, epochs=50)
            elif model_type == "few":
                detector = FewShotPrototypicalDetector()
                detector.fit(X_train, y_train, epochs=50) 
            
            # 🌟 3. 학습 완료된 모델 파일 저장 (중복 호출 제거 및 파일명 적용)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # 파일명 예시: piezo_01_all_20260508_143000.pt
            file_name = f"{s_id}_{model_type}_{timestamp}" 
            
            # 여기서 딱 한 번만 저장합니다! (내부 저장 로직에 .pt가 붙도록 되어 있다면 확장자는 뺍니다)
            saved_path = detector.save(file_name) 

            # 🌟 4. [NEW] DB에 모델 메타데이터 기록 (버전 자동 증가)
            # 이제 func.max가 정상적으로 작동합니다!
            max_version = db_session.query(func.max(ModelRegistry.VERSION)).filter(
                ModelRegistry.MAC_ADDR == s_id
            ).scalar() or 0
            
            new_version = max_version + 1

            # 임계값 통계 추출
            t_mean = detector.thresholds[0].get('mean', 0.0)
            t_std = detector.thresholds[0].get('std', 0.0)

            new_model_record = ModelRegistry(
                MAC_ADDR=s_id,
                MODEL_TYPE=model_type,
                VERSION=new_version,
                FILE_PATH=saved_path,
                TRAIN_SAMPLES=len(X_train),
                THRESHOLD_MEAN=t_mean,
                THRESHOLD_STD=t_std,
                STATUS="CANDIDATE" # 👈 처음엔 무조건 대기 상태!
            )
            
            db_session.add(new_model_record)
            db_session.commit()
            
            print(f"✅ [AI] {s_id} 모델(v{new_version}) DB 등록 성공!")

        except Exception as e:
            print(f"❌ [AI] 누출 모델 학습 실패: {e}")
        finally:
            db_session.close()

    background_tasks.add_task(task, sensor_id, days)
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