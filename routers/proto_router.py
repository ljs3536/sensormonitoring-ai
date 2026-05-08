from fastapi import APIRouter, BackgroundTasks, Depends, Body, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
import numpy as np
import datetime

# 프로젝트 구조에 맞게 Import 경로를 수정해 주세요.
from database_rdb import get_db, SensorData, SessionLocal
from architectures.prototypical import PrototypicalLeakDetector # 🌟 추가됨!
from architectures.fewshot_prototypical import FewShotPrototypicalDetector
router = APIRouter(prefix="/ai", tags=["Proto"])

class LeakPredictRequest(BaseModel):
    features: List[List[float]] 

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
                detector.fit(X_train, y_train, epochs=50) # 50번 반복 학습
            elif model_type == "few":
                detector = FewShotPrototypicalDetector()
                detector.fit(X_train, y_train, epochs=50) 
            # 🌟 3. 학습 완료된 모델 저장
            saved_path = detector.save(s_id)
            print(f"✅ [AI] {s_id} 누출 모델 학습 및 저장 성공! ({saved_path})")

        except Exception as e:
            print(f"❌ [AI] 누출 모델 학습 실패: {e}")
        finally:
            db_session.close()

    background_tasks.add_task(task, sensor_id, days)
    return {"message": "누출 모델(Prototypical) 학습이 시작되었습니다."}


@router.post("/proto/predict")
async def predict_leak_model(sensor_id: str, model_type: str, payload: LeakPredictRequest):
    print(sensor_id)
    try:
        # 1. 프론트엔드/백엔드에서 받은 배열 (예: N개 데이터 x 128차원)
        X_test = np.array(payload.features) 
        
        # 🌟 2. 아키텍처 모델 로드 및 예측 진행
        if model_type == "all":
            detector = PrototypicalLeakDetector()
            detector.load(sensor_id) # 저장된 모델 파일 불러오기
        elif model_type == "few":
            detector = FewShotPrototypicalDetector()
            detector.load(sensor_id)
        # 실제 모델 추론 실행!
        results = detector.predict(X_test)

        return {"predictions": results}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="모델이 학습되지 않았습니다. 먼저 모델을 갱신해주세요.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")