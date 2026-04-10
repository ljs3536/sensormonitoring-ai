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
from architectures.cnnlstm_classifier import CNNLSTMClassifier 
from architectures.spectrogram_cnn import SpectrogramCNN
from preprocess import SpectrogramTransformer

# =====================================================================
# 1. 비지도 학습용 추론 엔진 (AutoEncoder)
# =====================================================================
def run_unsupervised_inference(sensor_type: str, file_path: str, model_type: str, input_data: list):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {file_path}")
    
    # 🌟 1. 매핑 파일 로드 (학습 당시의 max_val 가져오기)
    mapping_path = file_path.replace(".pt", "_mapping.json")
    train_max_val = 1.0
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
            train_max_val = mapping_data.get("max_val", 1.0)

    # 🌟 2. 센서 타입에 따른 설정
    is_adxl = sensor_type.lower() == "adxl"
    features = 3 if is_adxl else 1
    
    # MLP(SensorAutoEncoder)는 전체 데이터 개수가 입력 사이즈가 됨
    actual_input_count = len(input_data) 
    
    # 🌟 3. 모델 초기화 (features와 seq_len을 정확히 전달)
    if model_type.lower() == "autoencoder":
        model = SensorAutoEncoder(input_size=actual_input_count)
    elif model_type.lower() == "cnnlstmautoencoder":
        # 하이브리드 모델은 (채널, 길이) 구조이므로 128로 고정
        model = CNNLSTMAutoEncoder(seq_len=128, features=features)
    else:
        raise ValueError("지원하지 않는 비지도 모델입니다.")

    model.load_state_dict(torch.load(file_path, weights_only=True))
    model.eval() 

    # 🌟 4. 데이터 전처리
    input_arr = np.array(input_data, dtype=np.float32)
    input_normalized = input_arr / (train_max_val if train_max_val != 0 else 1.0)

    # 🌟 5. 모델별 입력 차원 맞추기 (Reshape)
    if model_type.lower() == "autoencoder":
        # MLP: [1, 128] 또는 [1, 384]
        tensor_x = torch.tensor(input_normalized).unsqueeze(0)
    elif model_type.lower() == "cnnlstmautoencoder":
        if is_adxl:
            # ADXL 384개를 (3, 128)로 눕힌 뒤 배치 차원 추가 -> [1, 3, 128]
            reshaped = input_normalized.reshape(128, 3).T
            tensor_x = torch.tensor(reshaped, dtype=torch.float32).unsqueeze(0)
        else:
            # Piezo -> [1, 1, 128]
            tensor_x = torch.tensor(input_normalized).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor_x)

    # 🌟 6. 오차 계산 및 시각화 데이터 준비
    mse = nn.MSELoss()(output, tensor_x).item()

    # 차트 출력을 위해 데이터를 1차원 리스트로 다시 평탄화
    # 학습 때 쓴 max_val을 다시 곱해서 원본 스케일로 복원
    original_signal = (tensor_x.cpu().numpy().flatten() * train_max_val).tolist()
    reconstructed_signal = (output.cpu().numpy().flatten() * train_max_val).tolist()
    pointwise_error = np.abs(np.array(original_signal) - np.array(reconstructed_signal)).tolist()

    # 임계치 설정 (이 부분은 데이터에 따라 조정이 필요할 수 있습니다)
    THRESHOLD = 0.17
    # 기존: mse 기반 판단
    #is_anomaly = mse > THRESHOLD
    # 개선: 128개 지점 중 오차가 가장 큰 값을 확인
    max_point_error = np.max(pointwise_error)
    THRESHOLD_MAX = 0.8 # 이 수치는 테스트를 통해 조정

    is_anomaly = (mse > THRESHOLD) or (max_point_error > THRESHOLD_MAX)
    anomaly_score = min(mse / (THRESHOLD * 2), 1.0)
    mse_score = mse / (THRESHOLD * 2)
    max_score = max_point_error / THRESHOLD_MAX
    anomaly_score = min(max(mse_score, max_score), 1.0)

    severity = "SAFE"
    if anomaly_score > 0.8: severity = "CRITICAL" # 80% 이상이면 위험
    elif anomaly_score > 0.4: severity = "WARNING" # 40% 이상이면 주의

    return {
        "learning_type": "unsupervised",
        "raw_mse": round(mse, 5),
        "threshold": THRESHOLD,
        "model_type": model_type,
        "anomaly_score": round(anomaly_score, 4),
        "prediction": "abnormal" if is_anomaly else "normal",
        "severity": severity,
        "message": "비정상 패턴 감지" if is_anomaly else "정상 패턴입니다.",
        "chart_data": {
            "original": original_signal,
            "reconstructed": reconstructed_signal,
            "errors": pointwise_error
        }
    }

# =====================================================================
# 🌟 2. 지도 학습용 추론 엔진 (Classifier)
# =====================================================================
def run_supervised_inference(sensor_type:str, file_path: str, model_type: str, input_data: list):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {file_path}")

    mapping_path = file_path.replace(".pt", "_mapping.json")
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    index_to_label = mapping_data["index_to_label"]
    train_max_val = mapping_data["max_val"]
    
    is_adxl = sensor_type.lower() == "adxl"
    features = 3 if is_adxl else 1
    num_classes = len(index_to_label)

    # 1. 모델 초기화
    if model_type.lower() == "cnnlstm_classifier":
        model = CNNLSTMClassifier(seq_len=128, features=features, num_classes=num_classes)
        
        input_arr = np.array(input_data, dtype=np.float32) / (train_max_val if train_max_val != 0 else 1.0)
        
        if is_adxl:
            # (384,) -> (3, 128)
            reshaped = input_arr.reshape(128, 3).T
            tensor_x = torch.tensor(reshaped).unsqueeze(0) # [1, 3, 128]
        else:
            tensor_x = torch.tensor(input_arr).unsqueeze(0).unsqueeze(0) # [1, 1, 128]

    elif model_type.lower() in ["spectrogram_cnn"]:
        model = SpectrogramCNN(in_channels=features, num_classes=num_classes)
        
        # 🌟 [중요] n_mels를 학습 때와 동일하게 32로 설정해야 합니다!
        transformer = SpectrogramTransformer(sample_rate=1000, n_mels=32) 
        
        input_arr = np.array(input_data, dtype=np.float32) / (train_max_val if train_max_val != 0 else 1.0)
        
        if is_adxl:
            reshaped = input_arr.reshape(128, 3).T
            x_2d = transformer(reshaped) 
        else:
            x_2d = transformer(input_arr)
        tensor_x = x_2d.unsqueeze(0)

    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model_type}")

    model.load_state_dict(torch.load(file_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(tensor_x)
        probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    best_index = int(np.argmax(probabilities))
    best_confidence = float(probabilities[best_index])
    predicted_label = index_to_label[str(best_index)]
    prob_dict = {index_to_label[str(i)]: float(prob) for i, prob in enumerate(probabilities)}

    severity = "SAFE"
    if predicted_label.lower() != "normal":
        severity = "CRITICAL" if best_confidence > 0.8 else "WARNING"

    # 🌟 [수정] 프론트엔드와 키 이름을 정확히 맞춥니다.
    return {
        "learning_type": "supervised",
        "prediction": predicted_label,
        "model_type": model_type, # 👈 언더바 추가
        "confidence": round(best_confidence, 4),
        "probabilities": prob_dict,
        "severity": severity,
        "message": f"AI 분석 결과 [{predicted_label}] 상태일 확률이 {best_confidence*100:.1f}%입니다.", # 👈 메시지 추가
        "chart_data": {"original": input_data}
    }