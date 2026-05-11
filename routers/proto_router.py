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

router = APIRouter(prefix="/ai", tags=["Proto"])

class LeakPredictRequest(BaseModel):
    features: List[List[float]] 
    file_path: str
    model_type: str


def train_proto_model_internal(sensor_id: str, model_type: str, days: int, auto_activate: bool):
    db_session = SessionLocal() 
    try:
        print(f"🚀 [AI] {sensor_id} 누출 모델 학습 시작...")
        
        # 1. 데이터 조회 및 파싱 (기존 코드와 동일)
        time_threshold = datetime.datetime.now() - datetime.timedelta(days=days)
        records = db_session.query(SensorData).filter(
            SensorData.MAC_ADDR == sensor_id, SensorData.REG_DT >= time_threshold,
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
        file_name = f"{sensor_id}_{model_type}_{timestamp}" 
        
        # 여기서 딱 한 번만 저장합니다! (내부 저장 로직에 .pt가 붙도록 되어 있다면 확장자는 뺍니다)
        saved_path = detector.save(file_name) 

        # 🌟 4. [NEW] DB에 모델 메타데이터 기록 (버전 자동 증가)
        # 이제 func.max가 정상적으로 작동합니다!
        max_version = db_session.query(func.max(ModelRegistry.VERSION)).filter(
            ModelRegistry.MAC_ADDR == sensor_id
        ).scalar() or 0
        
        new_version = max_version + 1

        # 임계값 통계 추출
        t_mean = detector.thresholds[0].get('mean', 0.0)
        t_std = detector.thresholds[0].get('std', 0.0)

        # 1. 학습 데이터를 AI 모델이 읽을 수 있게 텐서로 변환
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(detector.device)
        
        with torch.no_grad():
            # 2. 모델을 통과시켜 데이터들의 위치(임베딩) 추출
            embeds = detector.model(X_tensor)
            # 3. 데이터들의 중심점(평균 위치) 계산
            center = torch.mean(embeds, dim=0)
            # 4. 각 데이터들이 중심에서 얼마나 떨어져 있는지 '거리' 계산
            dists_np = torch.norm(embeds - center, dim=1).cpu().numpy()
            
        # 5. 차트를 그리기 위해 거리를 20개의 구간(막대)으로 쪼개기
        counts, bin_edges = np.histogram(dists_np, bins=20)

        # 통계학적 3-Sigma 한계선(이 선을 넘으면 누출)
        anomaly_limit = t_mean + (3.0 * t_std)

        # [NEW] 320차원 임베딩을 2D 좌표로 축소 (화면 그리기 용도)
        pca = PCA(n_components=2)
        embeds_np = embeds.cpu().numpy()
        
        # 데이터 개수가 2개 이상일 때만 PCA 수행 가능
        if len(embeds_np) >= 2:
            points_2d = pca.fit_transform(embeds_np)
            center_2d = pca.transform([center.cpu().numpy()])[0]
        else:
            points_2d = [[0.0, 0.0]] * len(embeds_np)
            center_2d = [0.0, 0.0]

        eval_metrics = {
            "input_dimension": X_train.shape[1] if len(X_train) > 0 else 0,
            "train_samples": len(X_train),
            "threshold_mean": float(t_mean),
            "threshold_std": float(t_std),
            "anomaly_limit_3sigma": float(anomaly_limit),
            # 밀집도 점수 (표준편차가 작을수록 클러스터가 단단하게 뭉쳐있어 성능이 좋음)
            "tightness_score": float(1.0 / (t_std + 1e-6)), 
            "model_type": model_type,
            "train_dist_hist": {
                "counts": counts.tolist(), # 예: [10, 50, 100, 5, 1 ...] (막대 높이)
                "bins": bin_edges.tolist() # 예: [0.0, 0.1, 0.2 ...] (X축 기준점)
            },
            "pca_2d_points": points_2d.tolist(), # 예: [[-1.2, 0.5], [2.1, -0.3], ...]
            "pca_2d_center": center_2d.tolist()
        }

        # [핵심 변경 사항] is_auto_activate 가 True면 즉시 ACTIVE로 설정
        new_status = "ACTIVE" if auto_activate else "CANDIDATE"

        # 만약 새 모델을 ACTIVE로 만들 거라면, 기존 ACTIVE 모델들을 INACTIVE로 강등시켜야 함!
        if new_status == "ACTIVE":
            db_session.query(ModelRegistry).filter(
                ModelRegistry.MAC_ADDR == sensor_id,
                ModelRegistry.STATUS == "ACTIVE"
            ).update({"STATUS": "INACTIVE"})

        new_model_record = ModelRegistry(
            MAC_ADDR=sensor_id,
            MODEL_TYPE=model_type,
            VERSION=new_version,
            FILE_PATH=saved_path,
            TRAIN_SAMPLES=len(X_train),
            THRESHOLD_MEAN=t_mean,
            THRESHOLD_STD=t_std,
            STATUS=new_status,
            EVAL_METRICS=eval_metrics
        )
        
        db_session.add(new_model_record)
        db_session.commit()

        #  백테스팅 (최근 3일 치 데이터로 모의고사 쳐서 로그 남기기)
        test_threshold = datetime.datetime.now() - datetime.timedelta(days=3)
        test_records = db_session.query(SensorData).filter(
            SensorData.MAC_ADDR == sensor_id, 
            SensorData.REG_DT >= test_threshold
        ).all()

        if test_records:
            X_test = np.array([[float(val) for val in r.SENSOR_DATA.split('|')] for r in test_records if r.SENSOR_DATA])
            
            if len(X_test) > 0:
                # 방금 학습한 모델로 즉시 예측!
                test_results = detector.predict(X_test)
                
                # 예측 결과를 tb_prediction_log에 '모의고사' 기록으로 대량 삽입
                logs_to_insert = []
                for res in test_results:
                    logs_to_insert.append(
                        PredictionLog(
                            MODEL_ID=new_model_record.MODEL_ID,
                            MAC_ADDR=sensor_id,
                            PROBABILITY=round(res.get("prob", 0) * 100, 2),
                            RESULT=res.get("is_leak", "N"),
                            # 모의고사임을 표시하고 싶다면 별도 컬럼(IS_BACKTEST)을 두거나 메모를 남겨도 좋습니다.
                        )
                    )
                db_session.bulk_save_objects(logs_to_insert)
                db_session.commit()
                print(f"✅ [AI] 백테스팅 완료: {len(logs_to_insert)}건의 로그 생성됨.")

        
        print(f"✅ [AI] {sensor_id} 모델(v{new_version}) DB 등록 성공!")

    except Exception as e:
        print(f"❌ [AI] 누출 모델 학습 실패: {e}")
    finally:
        db_session.close()


@router.post("/proto/train")
async def train_leak_model(sensor_id: str, model_type: str, days: int = 7, auto_activate: bool = True, background_tasks: BackgroundTasks = None):
    
    background_tasks.add_task(train_proto_model_internal, sensor_id, model_type, days, auto_activate)
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