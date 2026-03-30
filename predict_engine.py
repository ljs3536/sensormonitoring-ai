# sensor-ai/predict_engine.py
import torch
import torch.nn as nn
import numpy as np
import os
from architectures.autoencoder import SensorAutoEncoder # 우리가 만든 모델 구조 임포트
from architectures.cnnlstmautoencoder import CNNLSTMAutoEncoder

def run_inference(file_path: str, model_type:str, input_data: list):
    """
    저장된 PyTorch(.pt) 모델을 로드하여 추론을 실행하고 상세 리포트를 반환합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {file_path}")
    
    input_size = len(input_data)
    
    # 1. 모델 구조 세팅 및 가중치(.pt) 로드
    if model_type.lower() == "autoencoder":
        model = SensorAutoEncoder(input_size=input_size)
    elif model_type.lower() == "cnnlstmautoencoder":
        model = CNNLSTMAutoEncoder(seq_len=input_size)

    
    model.load_state_dict(torch.load(file_path, weights_only=True))
    model.eval() # 평가 모드 설정 (학습 모드 아님)

    # 2. 데이터 전처리 (학습할 때와 동일하게 Min-Max 스케일링)
    input_arr = np.array(input_data, dtype=np.float32)
    max_val = np.max(np.abs(input_arr))
    if max_val == 0: max_val = 1
    input_normalized = input_arr / max_val

    # PyTorch 텐서 변환 및 배치 차원(1) 추가: 모양을 (1, 128)로 만듦
    if model_type.lower() == "autoencoder":
        tensor_x = torch.tensor(input_normalized).unsqueeze(0)
    elif model_type.lower() == "cnnlstmautoencoder":
        tensor_x = torch.tensor(input_normalized).unsqueeze(0).unsqueeze(0)
    

    # 3. 추론 (Reconstruction) 실행
    with torch.no_grad(): # 예측할 때는 기울기 계산을 안 하므로 메모리 절약
        output = model(tensor_x)

    # 4. 복원 오차(MSE) 계산
    mse = nn.MSELoss()(output, tensor_x).item()

    # 5. 결과 지표 가공 (임계치 기반)
    # 실제 실무에서는 이 THRESHOLD를 학습 시 데이터의 분포를 보고 동적으로 설정합니다.
    THRESHOLD = 0.5 
    
    is_anomaly = mse > THRESHOLD

    # 프론트엔드 게이지(0~100%)를 위한 점수 정규화
    # 임계치(0.05)일 때 50%의 점수를 가지도록 스케일링 (최대 1.0)
    anomaly_score = min(mse / (THRESHOLD * 2), 1.0)

    # 위험도 레벨 분류
    if anomaly_score < 0.3:
        severity = "SAFE"
    elif anomaly_score < 0.7:
        severity = "WARNING"
    else:
        severity = "CRITICAL"

    # 6. 풍부한 결과 딕셔너리 반환
    return {
        "raw_mse": round(mse, 5),                 # 순수 복원 오차
        "threshold": THRESHOLD,                   # 판정 기준값
        "anomaly_score": round(anomaly_score, 4), # UI 게이지 표출용 점수 (0~1)
        "prediction": "abnormal" if is_anomaly else "normal", # 최종 판정
        "severity": severity,                     # 위험도 텍스트
        "message": "데이터 패턴이 정상 범주를 벗어났습니다." if is_anomaly else "정상적인 센서 패턴입니다."
    }