# sensor-ai/train_engine.py
import torch
import numpy as np
from database import AIStore
from architectures.autoencoder import AutoEncoderTrainer 

db = AIStore()

def run_training(sensor_type: str, model_type: str, days: int):
    """
    파라미터에 따라 적절한 모델을 선택하여 학습을 수행합니다.
    """
    print(f"--- [TRAIN ENGINE] {model_type} 모델 학습 시작 ({sensor_type}) ---")
    
    # 1. 데이터 로드
    df = db.fetch_training_data(sensor_type, days)
    if df.empty:
        raise ValueError("학습할 데이터가 부족합니다.")

    # 2. 모델 타입에 따른 분기 (Strategy)
    if model_type.lower() == "autoencoder":
        trainer = AutoEncoderTrainer(sensor_type)
        model_path = trainer.train(df) # 학습 후 TFLite 경로 반환
    elif model_type.lower() == "lstm":
        # TODO: LSTM 학습 로직 추가
        pass
    else:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

    return model_path