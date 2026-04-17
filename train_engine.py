# sensor-ai/train_engine.py
import torch
import numpy as np
from database import AIStore

# 비지도 학습용 임포트
from architectures.autoencoder import AutoEncoderTrainer 
from architectures.cnnlstmautoencoder import CNNLSTMAutoEncoderTrainer

# 지도 학습용 임포트 추가!
from architectures.cnnlstm_classifier import CNNLSTMClassifierTrainer
from architectures.spectrogram_cnn import SpectrogramCNNTrainer
from architectures.pinn_cnnlstmautoencoder import PINN_CNNLSTMAutoEncoderTrainer
from sensors import Sensor
from database_rdb import SessionLocal

db = AIStore()

def get_sensor_metadata(sensor_id: str):
    """MariaDB에서 센서의 물리 정보를 가져옵니다."""
    if not sensor_id: return None
    db_session = SessionLocal()
    try:
        sensor = db_session.query(Sensor).filter(Sensor.id == sensor_id).first()
        if sensor:
            return {
                "sampling_rate": sensor.sampling_rate,
                # DB 컬럼에 k, c가 없다면 임계값을 기반으로 가중치를 주거나 기본값을 쓸 수 있습니다.
                "k": sensor.threshold_max or 5.0, 
                "c": 0.5
            }
    finally:
        db_session.close()
    return None

def run_unsupervised_training(sensor_type: str, model_type: str, days: int, sensor_id: str = None):
    print(f"--- [TRAIN ENGINE] {model_type} 비지도 모델 학습 시작 ({sensor_type} / ID: {sensor_id or 'ALL'}) ---")
    df = db.fetch_unsupervised_data(sensor_type, days, sensor_id)
    if df.empty:
        raise ValueError("학습할 데이터가 부족합니다.")

    if model_type.lower() == "autoencoder":
        trainer = AutoEncoderTrainer(sensor_type)
        model_path = trainer.train(df)
    elif model_type.lower() == "cnnlstmautoencoder":
        trainer = CNNLSTMAutoEncoderTrainer(sensor_type)
        model_path = trainer.train(df)
    elif model_type.lower() == "pinn_cnnlstmautoencoder":
        sensor_meta = get_sensor_metadata(sensor_id)
        trainer = PINN_CNNLSTMAutoEncoderTrainer(sensor_type)
        # 물리 파라미터를 함께 넘김
        model_path = trainer.train(df, sensor_metadata=sensor_meta)

    else:
        raise ValueError(f"지원하지 않는 비지도 모델 타입입니다: {model_type}")

    return model_path

def run_supervised_training(sensor_type: str, model_type: str, days: int, sensor_id: str = None):
    print(f"--- [TRAIN ENGINE] {model_type} 지도(분류) 모델 학습 시작 ({sensor_type} / ID: {sensor_id or 'ALL'}) ---")
    df = db.fetch_supervised_data(sensor_type, days, sensor_id)
    
    # 지도 학습은 각 라벨별로 데이터가 있어야 하므로 체크
    if df.empty or 'label' not in df.columns:
        raise ValueError("지도 학습용 라벨 데이터가 존재하지 않습니다.")

    #  모델 타입에 따른 분기 (지도 학습용)
    if model_type.lower() == "cnnlstm_classifier":
        trainer = CNNLSTMClassifierTrainer(sensor_type)
        model_path = trainer.train(df)
    
    # 나중에 추가할 모델들 예약 자리!
    elif model_type.lower() == "spectrogram_cnn":
        trainer = SpectrogramCNNTrainer(sensor_type)
        model_path = trainer.train(df)
    # elif model_type.lower() == "timeseries_transformer":
    #     trainer = TransformerClassifierTrainer(sensor_type)
    #     model_path = trainer.train(df)
    
    else:
        raise ValueError(f"지원하지 않는 지도 모델 타입입니다: {model_type}")

    return model_path