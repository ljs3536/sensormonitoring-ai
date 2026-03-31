# sensor-ai/predict_engine.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json

# 모델들 임포트
from architectures.autoencoder import SensorAutoEncoder 
from architectures.cnnlstmautoencoder import CNNLSTMAutoEncoder
from architectures.cnnlstm_classifier import CNNLSTMClassifier # 🌟 추가

# =====================================================================
# 1. 비지도 학습용 추론 엔진 (AutoEncoder - 기존 로직)
# =====================================================================
def run_unsupervised_inference(file_path: str, model_type: str, input_data: list):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {file_path}")
    
    input_size = len(input_data)
    
    if model_type.lower() == "autoencoder":
        model = SensorAutoEncoder(input_size=input_size)
    elif model_type.lower() == "cnnlstmautoencoder":
        model = CNNLSTMAutoEncoder(seq_len=input_size)
    else:
        raise ValueError("지원하지 않는 비지도 모델입니다.")

    model.load_state_dict(torch.load(file_path, weights_only=True))
    model.eval() 

    input_arr = np.array(input_data, dtype=np.float32)
    max_val = np.max(np.abs(input_arr))
    if max_val == 0: max_val = 1
    input_normalized = input_arr / max_val

    if model_type.lower() == "autoencoder":
        tensor_x = torch.tensor(input_normalized).unsqueeze(0)
    elif model_type.lower() == "cnnlstmautoencoder":
        tensor_x = torch.tensor(input_normalized).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor_x)

    mse = nn.MSELoss()(output, tensor_x).item()

    original_signal = (tensor_x.squeeze().numpy() * max_val).tolist()
    reconstructed_signal = (output.squeeze().numpy() * max_val).tolist()
    pointwise_error = np.abs(np.array(original_signal) - np.array(reconstructed_signal)).tolist()

    THRESHOLD = 0.5 
    is_anomaly = mse > THRESHOLD
    anomaly_score = min(mse / (THRESHOLD * 2), 1.0)

    if anomaly_score < 0.3: severity = "SAFE"
    elif anomaly_score < 0.7: severity = "WARNING"
    else: severity = "CRITICAL"

    return {
        "learning_type": "unsupervised", # 프론트엔드 구분용
        "raw_mse": round(mse, 5),
        "threshold": THRESHOLD,
        "anomaly_score": round(anomaly_score, 4),
        "prediction": "abnormal" if is_anomaly else "normal",
        "severity": severity,
        "message": "데이터 패턴이 정상 범주를 벗어났습니다." if is_anomaly else "정상적인 센서 패턴입니다.",
        "chart_data": {
            "original": original_signal,
            "reconstructed": reconstructed_signal,
            "errors": pointwise_error
        }
    }


# =====================================================================
# 🌟 2. 지도 학습용 추론 엔진 (Classifier - 신규 로직)
# =====================================================================
def run_supervised_inference(file_path: str, model_type: str, input_data: list):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {file_path}")

    # 매핑 파일(.json) 경로 찾기 (모델 파일명 끝의 .pt를 _mapping.json으로 변경)
    mapping_path = file_path.replace(".pt", "_mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"라벨 매핑 파일을 찾을 수 없습니다: {mapping_path}")

    # 🌟 매핑 정보 & 학습 당시의 max_val 로드
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    # 데이터 꺼내기
    index_to_label = mapping_data["index_to_label"]
    train_max_val = mapping_data["max_val"]
    
    num_classes = len(index_to_label)
    input_size = len(input_data)

    # 모델 세팅
    if model_type.lower() == "cnnlstm_classifier":
        model = CNNLSTMClassifier(seq_len=input_size, num_classes=num_classes)
    else:
        raise ValueError("지원하지 않는 지도 모델입니다.")

    model.load_state_dict(torch.load(file_path, weights_only=True))
    model.eval()

    # [핵심] 예측할 때 자기 자신이 아닌, 학습 당시의 train_max_val로 나눠줌!!
    input_arr = np.array(input_data, dtype=np.float32)
    if train_max_val == 0: train_max_val = 1
    input_normalized = input_arr / train_max_val

    # CNN-LSTM 용 차원 (batch=1, channels=1, seq_len=128)
    tensor_x = torch.tensor(input_normalized).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor_x) # 결과값은 확률이 아닌 순수 수치(Logits)
        
        # Softmax를 적용해서 총합이 100%(1.0)가 되는 확률값으로 변환!
        probabilities = F.softmax(logits, dim=1).squeeze().numpy()

    # 가장 확률이 높은 인덱스 찾기
    best_index = int(np.argmax(probabilities))
    best_confidence = float(probabilities[best_index])
    
    # 인덱스를 다시 문자열 라벨로 변환 ("0" -> "normal")
    predicted_label = index_to_label[str(best_index)]

    # 모든 클래스에 대한 확률을 Dictionary로 포장 (UI 차트용)
    prob_dict = {index_to_label[str(i)]: float(prob) for i, prob in enumerate(probabilities)}

    # 위험도 레벨 분류 (normal이 아니면 무조건 Warning/Critical)
    severity = "SAFE"
    if predicted_label.lower() != "normal":
        severity = "CRITICAL" if best_confidence > 0.8 else "WARNING"

    # 프론트엔드 반환 데이터
    return {
        "learning_type": "supervised", # 프론트엔드 구분용
        "prediction": predicted_label, # 최종 예측 라벨
        "confidence": round(best_confidence, 4), # 확신도 (예: 0.985)
        "probabilities": prob_dict, # {"normal": 0.015, "anomaly": 0.985}
        "severity": severity,
        "message": f"이 데이터는 {best_confidence*100:.1f}% 확률로 [{predicted_label}] 상태입니다.",
        "chart_data": {
            "original": input_data # 원본 데이터는 UI에서 참고용으로 그릴 수 있게 전달
        }
    }