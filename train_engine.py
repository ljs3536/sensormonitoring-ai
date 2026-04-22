# sensor-ai/train_engine.py
import torch
import numpy as np
from database import AIStore
from database_rdb import SessionLocal
# 비지도 학습용 임포트
from architectures.autoencoder import AutoEncoderTrainer 
from architectures.cnnlstmautoencoder import CNNLSTMAutoEncoderTrainer

# 지도 학습용 임포트 추가!
from architectures.cnnlstm_classifier import CNNLSTMClassifierTrainer
from architectures.spectrogram_cnn import SpectrogramCNNTrainer
from architectures.pinn_cnnlstmautoencoder import PINN_CNNLSTMAutoEncoderTrainer
from services.model_service import SensorService
influx_store = AIStore()

def run_unsupervised_training(sensor_type: str, model_type: str, days: int, sensor_id: str = None):
    print(f"--- [TRAIN ENGINE] {model_type} 비지도 모델 학습 시작 ({sensor_type} / ID: {sensor_id or 'ALL'}) ---")
    df = influx_store.fetch_unsupervised_data(sensor_type, days, sensor_id)
    if df.empty:
        raise ValueError("학습할 데이터가 부족합니다.")

    if model_type.lower() == "autoencoder":
        trainer = AutoEncoderTrainer(sensor_type)
        model_path = trainer.train(df)
    elif model_type.lower() == "cnnlstmautoencoder":
        trainer = CNNLSTMAutoEncoderTrainer(sensor_type)
        model_path = trainer.train(df)
    elif model_type.lower() == "pinn_cnnlstmautoencoder":
        # DB에서 메타데이터 맵 가져오기 (Service 활용)
        db = SessionLocal()
        try:
            sensor_meta_map = SensorService.get_all_sensor_metadata(db, sensor_type)
        finally:
            db.close()
        trainer = PINN_CNNLSTMAutoEncoderTrainer(sensor_type)
        # 물리 파라미터를 함께 넘김
        model_path = trainer.train(df, sensor_meta_map)

    else:
        raise ValueError(f"지원하지 않는 비지도 모델 타입입니다: {model_type}")

    return model_path

def run_supervised_training(sensor_type: str, model_type: str, days: int, sensor_id: str = None):
    print(f"--- [TRAIN ENGINE] {model_type} 지도(분류) 모델 학습 시작 ({sensor_type} / ID: {sensor_id or 'ALL'}) ---")
    df = influx_store.fetch_supervised_data(sensor_type, days, sensor_id)
    
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