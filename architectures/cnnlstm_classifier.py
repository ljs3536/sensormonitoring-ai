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
        x = self.encoder_cnn(x)              # -> (batch, 32, 32)
        x = x.transpose(1, 2)                # -> (batch, 32, 32)
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
        
        values = df[num_cols[0]].values
        labels = df['label'].map(label_to_index).values # 문자를 숫자로 바꾼 배열
        
        if len(values) < self.window_size:
            raise ValueError(f"데이터가 너무 적습니다. (현재: {len(values)})")

        num_windows = len(values) // self.window_size
        data_chopped = values[:num_windows * self.window_size]
        
        # 여기서 중요한 점: 각 128개 묶음마다 정답(label)이 1개 필요함
        
        labels_chopped = labels[:num_windows * self.window_size]
        labels_matrix = labels_chopped.reshape(-1, self.window_size)
        # stats.mode는 매트릭스의 각 행(axis=1)에서 가장 흔한 값을 찾습니다.
        mode_result = stats.mode(labels_matrix, axis=1, keepdims=False)
        final_labels = mode_result.mode
        # 가장 간단한 방법: 128개 묶음의 가장 마지막 데이터의 라벨을 그 묶음의 정답으로 사용
        #final_labels = labels_matrix[:, -1] # (데이터 수,) 형태의 정답지
        
        data_matrix = data_chopped.reshape(-1, 1, self.window_size)

        # Min-Max 스케일링
        max_val = np.max(np.abs(data_matrix))
        if max_val == 0: max_val = 1
        data_normalized = data_matrix / max_val

        # PyTorch 텐서 변환
        tensor_x = torch.tensor(data_normalized, dtype=torch.float32)
        tensor_y = torch.tensor(final_labels, dtype=torch.long) # 분류용 정답은 Long타입이어야 함

        dataset = TensorDataset(tensor_x, tensor_y) # 입력과 정답이 다름 (X, Y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 모델 초기화 (클래스 개수에 맞춰서)
        model = CNNLSTMClassifier(seq_len=self.window_size, num_classes=len(unique_labels))
        
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

        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        timestamp = int(time.time())
        file_name = f"{self.sensor_type}_cnnlstm_classifier_{timestamp}.pt"
        file_path = os.path.join(model_dir, file_name)
        
        torch.save(model.state_dict(), file_path)
        
        # 나중에 Predict 할 때 숫자를 다시 문자로 바꾸기 위해 매핑 정보도 같이 저장해 줍니다.
        # 나중에 Predict 할 때 똑같은 비율로 스케일링하기 위해 max_val도 같이 저장!
        mapping_path = os.path.join(model_dir, f"{self.sensor_type}_cnnlstm_classifier_{timestamp}_mapping.json")
        with open(mapping_path, 'w') as f:
            index_to_label = {idx: label for label, idx in label_to_index.items()}
            # 라벨 딕셔너리와 max_val을 하나로 묶어서 저장합니다.
            save_data = {
                "index_to_label": index_to_label,
                "max_val": float(max_val)  # numpy float은 json 직렬화가 안 되므로 float() 씌움
            }
            json.dump(save_data, f)

        print(f"--- [TRAIN] 분류 모델 생성 완료! 저장 위치: {file_path} ---")
        return file_path