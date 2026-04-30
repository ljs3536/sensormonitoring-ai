# sensor-ai/architectures/cnnlstm_classifier.py
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import shutil

# 1. 지도 학습용 CNN-LSTM 분류기 모델
class CNNLSTMClassifier(nn.Module):
    def __init__(self, seq_len=128, features=1, num_classes=2):
        super(CNNLSTMClassifier, self).__init__()
        
        # [특징 추출] CNN
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=features, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # [시간 흐름 파악] LSTM
        self.encoder_lstm = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
        
        # [분류기] Dense Layer (결과를 클래스 개수만큼 출력)
        # 128 길이를 2번 stride=2로 줄였으므로 128 -> 64 -> 32가 됨.
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # 과적합 방지
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, features, 128)
        x = self.encoder_cnn(x)              # -> (batch, 32, 32)
        x = x.transpose(1, 2)                # -> (batch, 32, 32) (LSTM은 배치를 제외하고 [길이, 특징] 순서를 기대함)
        x, _ = self.encoder_lstm(x)          # -> (batch, 32, 16)
        x = self.flatten(x)                  # -> (batch, 512)
        x = self.classifier(x)               # -> (batch, num_classes)
        return x

# 2. 전용 학습기 (Trainer)
class CNNLSTMClassifierTrainer:
    def __init__(self, sensor_type: str):
        self.sensor_type = sensor_type
        self.window_size = 128 

    def train(self, df: pd.DataFrame) -> str:
        print(f"--- [TRAIN] {self.sensor_type} CNN-LSTM Classifier 지도 학습 시작 ---")
        
        # 센서 타입에 따른 특징(Features) 개수 설정
        features = 3 if self.sensor_type.lower() == "adxl" else 1

        # 1. 라벨(정답지) 전처리
        if 'label' not in df.columns:
            raise ValueError("지도 학습을 위해서는 'label' 컬럼이 반드시 필요합니다.")
        
        # 문자로 된 라벨("normal", "anomaly")을 숫자(0, 1)로 변환 (Label Encoding)
        unique_labels = df['label'].unique().tolist()
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"💡 학습 감지된 클래스: {label_to_index}")
        
        # 숫자형 데이터만 추출
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("학습할 수 있는 숫자형 데이터가 없습니다.")
        if features == 3:
            values = df[num_cols[:3]].values
            num_windows = len(values) // self.window_size
            data_chopped = values[:num_windows * self.window_size]
            data_matrix = data_chopped.reshape(num_windows, self.window_size, 3).transpose(0, 2, 1)
        else:
            values = df[num_cols[0]].values
            num_windows = len(values) // self.window_size
            data_chopped = values[:num_windows * self.window_size]
            data_matrix = data_chopped.reshape(-1, 1, self.window_size)

            
        labels = df['label'].map(label_to_index).values # 문자를 숫자로 바꾼 배열
        
        if len(values) < self.window_size:
            raise ValueError(f"데이터가 너무 적습니다. (현재: {len(values)})")

        # 여기서 중요한 점: 각 128개 묶음마다 정답(label)이 1개 필요함
        labels_chopped = labels[:num_windows * self.window_size]
        labels_matrix = labels_chopped.reshape(-1, self.window_size)
        # stats.mode는 매트릭스의 각 행(axis=1)에서 가장 흔한 값을 찾습니다.
        mode_result = stats.mode(labels_matrix, axis=1, keepdims=False)
        final_labels = mode_result.mode.flatten()
        

        # Min-Max 스케일링
        max_val = np.max(np.abs(data_matrix))
        if max_val == 0: max_val = 1
        data_normalized = data_matrix / max_val

        # PyTorch 텐서 변환
        tensor_x = torch.tensor(data_normalized, dtype=torch.float32)
        tensor_y = torch.tensor(final_labels, dtype=torch.long) # 분류용 정답은 Long타입이어야 함

        dataset = TensorDataset(tensor_x, tensor_y) # 입력과 정답이 다름 (X, Y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 🌟 3. 모델 생성 시 features 값 전달
        model = CNNLSTMClassifier(
            seq_len=self.window_size, 
            features=features, # 👈 여기서 1 또는 3이 전달됨!
            num_classes=len(unique_labels)
        )

        # 지도 학습(분류)의 핵심: 오차 계산을 CrossEntropyLoss로 변경!
        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 20
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")

        # # 🌟 저장 로직 부분 수정
        # model_dir = "models"
        # os.makedirs(model_dir, exist_ok=True)
        # timestamp = int(time.time())
        # file_name = f"{self.sensor_type}_cnnlstm_classifier_{timestamp}.pt"
        # file_path = os.path.join(model_dir, file_name)
        
        # torch.save(model.state_dict(), file_path)
        
        # # 🌟 매핑 정보 저장 시 model_type을 명시합니다.
        # mapping_path = file_path.replace(".pt", "_mapping.json")
        # with open(mapping_path, 'w') as f:
        #     index_to_label = {idx: label for label, idx in label_to_index.items()}
        #     save_data = {
        #         "index_to_label": index_to_label,
        #         "max_val": float(max_val),
        #         "model_type": "cnnlstm_classifier" # 👈 추론 엔진의 자동 분기를 위해 추가
        #     }
        #     json.dump(save_data, f)

        # print(f"--- [TRAIN] 모델 생성 완료! 저장 위치: {file_path} ---")
        # return file_path
        # ========================================================
        # 🌟 로컬 vs K8s 환경을 스스로 판단하는 저장 로직
        # ========================================================
        
        # 1. 내가 지금 K8s 안에 있는지 확인 (K8s는 이 환경변수를 무조건 가지고 있음)
        IS_K8S_ENV = "KUBERNETES_SERVICE_HOST" in os.environ
        
        # 2. 경로 설정 (환경 변수가 없으면 로컬 개발용 상대 경로 './models' 사용)
        final_dir = os.getenv("MODEL_DIR", "./models")
        temp_dir = os.getenv("TEMP_MODEL_DIR", "./temp_models")
        
        os.makedirs(final_dir, exist_ok=True)
        
        timestamp = int(time.time())
        base_filename = f"{self.sensor_type}_cnnlstm_classifier_{timestamp}"
        
        final_model_path = os.path.join(final_dir, f"{base_filename}.pt")
        final_mapping_path = os.path.join(final_dir, f"{base_filename}_mapping.json")

        index_to_label = {idx: label for label, idx in label_to_index.items()}
                 
        # 메타데이터 준비
        meta_data = {
            "index_to_label": index_to_label, 
            "max_val": max_val.tolist(),
            "model_type": "cnnlstm_classifier"
        }

        if IS_K8S_ENV:
            # ---------------------------------------------------
            # [K8s 환경] 하이브리드 저장 (RAM -> PVC로 복사)
            # ---------------------------------------------------
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_model_path = os.path.join(temp_dir, f"{base_filename}.pt")
            temp_mapping_path = os.path.join(temp_dir, f"{base_filename}_mapping.json")
            
            # 1. 빠른 RAM에 먼저 쓰기
            torch.save(self.model.state_dict(), temp_model_path)
            with open(temp_mapping_path, 'w') as f:
                json.dump(meta_data, f)
                
            # 2. 느린 영구 디스크로 한 번에 복사
            shutil.copy(temp_model_path, final_model_path)
            shutil.copy(temp_mapping_path, final_mapping_path)
            print(f"--- [K8s] RAM -> PVC 하이브리드 저장 완료 ---")
            
        else:
            # ---------------------------------------------------
            # [Local 환경] 다이렉트 저장 (바로 SSD에 쓰기)
            # ---------------------------------------------------
            torch.save(self.model.state_dict(), final_model_path)
            with open(final_mapping_path, 'w') as f:
                json.dump(meta_data, f)
            print(f"--- [Local] 다이렉트 로컬 저장 완료 ---")

        return final_model_path