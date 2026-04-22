# sensor-ai/main.py
# 서비스 엔트리 포인트 (FastAPI 라우팅)
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import AIStore
import numpy as np
import os
import datetime
from train_engine import run_unsupervised_training, run_supervised_training
from predict_engine import run_unsupervised_inference, run_supervised_inference, run_pinn_inference

# RDB 연동 모듈 임포트
from database_rdb import get_db, SessionLocal
from models import AiModel
from sensors import Sensor
# 스케줄러 추가
from scheduler import scheduler
from architectures.calibration_pinn import InversePINN_CalibratorTrainer
from services.model_service import ModelService
from services.sensor_service import SensorService

app = FastAPI()
influx_store = AIStore() # DB 클라이언트 인스턴스 생성

# 모델 저장 경로 설정
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    if not scheduler.running:
        scheduler.start()
        print("--- [SYSTEM] 영구 삭제 스케줄러가 가동되었습니다. (매일 새벽 3시) ---")

@app.on_event("shutdown")
async def shutdown_event():
    if scheduler.running:
        scheduler.shutdown()
        print("--- [SYSTEM] 스케줄러가 종료되었습니다. ---")

@app.post("/train")
async def train_model(
    sensor_type: str, 
    model_type: str = "AutoEncoder", # 기본값 설정
    days: int = 7, 
    sensor_id: str = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db) # MariaDB 세션 주입 (이게 있어야 db.add가 작동합니다!)
):
    # 1. 학습 시작 전 DB에 'TRAINING' 상태로 레코드 생성
    # Service 호출로 DB 생성 로직 위임
    new_model = ModelService.create_training_record(db, sensor_type, model_type)

    # 2. 백그라운드 작업 정의
    def task(m_id, s_type, m_type, train_days):
        # 백그라운드 쓰레드이므로 별도의 DB 세션을 열어야 합니다.
        db_session = SessionLocal()
        model_record = db_session.query(AiModel).filter(AiModel.id == m_id).first()
        
        try:
            # 학습 엔진 실행 (완료 후 생성된 tflite 파일의 절대/상대 경로 반환)
            if model_type.lower() in ["autoencoder","cnnlstmautoencoder", "pinn_cnnlstmautoencoder"]:
                file_path = run_unsupervised_training(s_type, m_type, train_days, sensor_id)
            elif model_type.lower() in ["cnnlstm_classifier","spectrogram_cnn"]:
                file_path = run_supervised_training(s_type, m_type, train_days, sensor_id)
            # 성공 시 DB 업데이트
            model_record.status = "READY"
            model_record.file_path = file_path
            db_session.commit()
            print(f"--- [SUCCESS] 모델 {m_id} 학습 완료 ---")
            
        except Exception as e:
            # 실패 시 ERROR 상태로 업데이트
            print(f"--- [ERROR] 모델 {m_id} 학습 실패: {e} ---")
            model_record.status = "ERROR"
            db_session.commit()
        finally:
            db_session.close()

    # 3. 백그라운드 실행
    background_tasks.add_task(task, new_model.id, sensor_type, model_type, days)
    return {
        "message": f"{sensor_type} {model_type} 학습이 시작되었습니다.", 
        "model_id": new_model.id
    }


@app.post("/predict")
async def predict(sensor_type: str, model_id: int, sensor_id: str = None, data: list = Body(...), db: Session = Depends(get_db)):

    # 1. DB에서 모델 정보(파일 경로) 조회
    model_record = ModelService.get_ready_model(db, model_id)
    
    if not model_record:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다.")
    if model_record.status != "READY":
        raise HTTPException(status_code=400, detail="모델이 아직 학습 중이거나 에러 상태입니다.")
    
    model_type_lower = model_record.model_type.lower()
    print(f"--- [PREDICT] 선택된 모델: {model_type_lower} ---")

    try:
        # 2-1. PINN 모델 전용 분기 (물리 정보 주입 필요)
        if model_type_lower == "pinn_cnnlstmautoencoder":
            sensor_meta = None
            if sensor_id:
                sensor_record = db.query(Sensor).filter(Sensor.id == sensor_id).first()
                if sensor_record:
                    # DB에서 해당 센서의 물리 상수 가져오기
                    sensor_meta = {
                        "sampling_rate": sensor_record.sampling_rate,
                        "k": sensor_record.physics_k or 0.5,
                        "c": sensor_record.physics_c or 0.01 # 임시 감쇠 계수
                    }
            # PINN 전용 엔진 호출
            result_dict = run_pinn_inference(sensor_type, model_record.file_path, data, sensor_meta)

        # 2-2. 일반 비지도 학습 모델 분기
        elif model_type_lower in ["autoencoder", "cnnlstmautoencoder"]:
            result_dict = run_unsupervised_inference(sensor_type, model_record.file_path, model_type_lower, data)
            
        # 2-3. 일반 지도 학습 모델 분기
        elif model_type_lower in ["cnnlstm_classifier", "spectrogram_cnn"]:
            result_dict = run_supervised_inference(sensor_type, model_record.file_path, model_type_lower, data)
        
        else:
            raise ValueError("지원하지 않는 모델 아키텍처입니다.")
        
        return result_dict
    except Exception as e:
        print(f" [PREDICT ERROR]: {str(e)}") 
        return {"error": str(e)}

# API 엔드포인트 예시 스케치
@app.post("/auto_tune")
async def auto_tune_sensor(sensor_id: str, sensor_type: str = "piezo", days: int = 7, db: Session = Depends(get_db)):
    # 1. InfluxDB에서 해당 센서의 최근 정상 데이터(df)를 가져온다.
    df = influx_store.fetch_normal_data_from_influx(sensor_type=sensor_type, days=days, sensor_id=sensor_id)
    
    if df.empty:
        return {"status": "error", "message": "정상(normal) 데이터가 충분하지 않아 최적화를 진행할 수 없습니다."}
    
    latest_model = ModelService.get_latest_pinn_model(db, sensor_type)
    pretrained_path = latest_model.file_path if latest_model else None

    trainer = InversePINN_CalibratorTrainer(sensor_type=sensor_type)
    best_k, best_c, best_thresh = trainer.calibrate(df, pretrained_path)
    
    SensorService.update_recommended_params(db, sensor_id, best_k, best_c, best_thresh)
    
    return {
        "status": "success",
        "suggested_k": best_k,
        "suggested_c": best_c,
        "suggested_threshold": best_thresh,
        "message": "AI 최적화가 완료되어 추천 값이 센서 정보에 저장되었습니다."
    }
    
@app.get("/models")
async def get_all_models(sensor_type: str = None, db: Session = Depends(get_db)):
    """등록된 모든 모델 목록을 가져옵니다."""
    models = ModelService.get_all_models(db, sensor_type)
    return models

@app.delete("/models/{model_id}")
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    """특정 모델의 DB 기록과 실제 파일을 삭제합니다."""
    success = ModelService.soft_delete_model(db, model_id)
    if not success:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없거나 이미 삭제되었습니다.")
    return {"message": "모델이 안전하게 삭제되었습니다."}

@app.get("/status")
async def get_status(db: Session = Depends(get_db)):
    # 시스템 상태 체크용 
    return {"status": "ai ready"}