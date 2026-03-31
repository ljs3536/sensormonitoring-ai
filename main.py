# sensor-ai/main.py
# 서비스 엔트리 포인트 (FastAPI 라우팅)
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import AIStore 
import numpy as np
import os

from train_engine import run_unsupervised_training, run_supervised_training
from predict_engine import run_unsupervised_inference, run_supervised_inference

# RDB 연동 모듈 임포트
from database_rdb import get_db, SessionLocal
from models import AiModel

app = FastAPI()
influx_store = AIStore() # DB 클라이언트 인스턴스 생성

# 모델 저장 경로 설정
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


@app.post("/train")
async def train_model(
    sensor_type: str, 
    model_type: str = "AutoEncoder", # 기본값 설정
    days: int = 7, 
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db) # MariaDB 세션 주입 (이게 있어야 db.add가 작동합니다!)
):
    # 1. 학습 시작 전 DB에 'TRAINING' 상태로 레코드 생성
    new_model = AiModel(
        sensor_type=sensor_type,
        model_type=model_type,
        status="TRAINING"
    )
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    model_id = new_model.id # 생성된 ID 가져오기

    # 2. 백그라운드 작업 정의
    def task(m_id, s_type, m_type, train_days):
        # 백그라운드 쓰레드이므로 별도의 DB 세션을 열어야 합니다.
        db_session = SessionLocal()
        model_record = db_session.query(AiModel).filter(AiModel.id == m_id).first()
        
        try:
            # 학습 엔진 실행 (완료 후 생성된 tflite 파일의 절대/상대 경로 반환)
            if model_type.lower() in ["autoencoder","cnnlstmautoencoder"]:
                file_path = run_unsupervised_training(s_type, m_type, train_days)
            elif model_type.lower() in ["cnnlstm_classifier"]:
                file_path = run_supervised_training(s_type, m_type, train_days)
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
    background_tasks.add_task(task, model_id, sensor_type, model_type, days)
    return {
        "message": f"{sensor_type} {model_type} 학습이 시작되었습니다.", 
        "model_id": model_id
    }


@app.post("/predict")
async def predict(model_id: int, data: list = Body(...), db: Session = Depends(get_db)):
    """
    이제 predict는 sensor_type 대신 model_id를 받습니다!
    """
    # 1. DB에서 모델 정보(파일 경로) 조회
    model_record = db.query(AiModel).filter(AiModel.id == model_id).first()
    
    if not model_record:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다.")
    if model_record.status != "READY":
        raise HTTPException(status_code=400, detail="모델이 아직 학습 중이거나 에러 상태입니다.")

    try:
        # 2. 파일 경로를 예측 엔진에 넘겨서 추론 실행
        # run_inference 함수도 파라미터를 file_path를 받도록 수정해야 합니다.
        if model_record.model_type.lower() in ["autoencoder","cnnlstmautoencoder"]:
                result_dict = run_unsupervised_inference(model_record.file_path, model_record.model_type, data)
        elif model_record.model_type.lower() in ["cnnlstm_classifier"]:
            result_dict = run_supervised_inference(model_record.file_path, model_record.model_type, data)
        
        print("결과 : ",result_dict)
        return result_dict
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/models")
async def get_all_models(sensor_type: str = None, db: Session = Depends(get_db)):
    """등록된 모든 모델 목록을 가져옵니다."""
    query = db.query(AiModel).order_by(AiModel.created_at.desc())
    if sensor_type:
        query = query.filter(AiModel.sensor_type == sensor_type)
    return query.all()

@app.delete("/models/{model_id}")
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    """특정 모델의 DB 기록과 실제 파일을 삭제합니다."""
    model_record = db.query(AiModel).filter(AiModel.id == model_id).first()
    
    # 파일 경로 미리 복사 (삭제 후에 레코드 접근이 안될 수 있음)
    file_path = model_record.file_path
    mapping_path = file_path.replace(".pt","_mapping.json") if file_path else None

    if not model_record:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다.")
    try:
        # 1️. DB 레코드 삭제 (아직 commit 안 함, '대기' 상태)
        db.delete(model_record)
        
        # 2️. 실제 파일 삭제 시도
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        if mapping_path and os.path.exists(mapping_path):
            os.remove(mapping_path)

        # 3️. 모든 과정이 무사히 끝나면 DB에 최종 반영 (Commit)
        db.commit()
        print(f"--- [SUCCESS] 모델 {model_id} 관련 모든 자원 삭제 완료 ---")

    except Exception as e:
        # ❌ 도중에 에러가 발생하면 DB 조작을 취소(Rollback)
        db.rollback()
        print(f"--- [ROLLBACK] 삭제 중 오류 발생, DB 상태를 복구합니다: {e} ---")
        raise HTTPException(status_code=500, detail="삭제 중 서버 오류가 발생했습니다.")

    return {"message": f"모델 {model_id}이(가) 성공적으로 삭제되었습니다."}

@app.get("/status")
async def get_status(db: Session = Depends(get_db)):
    # 시스템 상태 체크용 (최근 동작 중인 모델이 있는지 반환)
    is_training = db.query(AiModel).filter(AiModel.status == "TRAINING").first() is not None
    return {"status": "training" if is_training else "ready"}