# sensor-ai/architectures/cnnlstmautoencoder.py
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. CNN-LSTM AutoEncoder 구조
class CNNLSTMAutoEncoder(nn.Module):
    def __init__(self, seq_len=128, features=1):
        super(CNNLSTMAutoEncoder, self).__init__()
        
        # [인코더] CNN: 특징 추출 및 길이 압축 (128 -> 64 -> 32)
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=features, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # [인코더] LSTM: 시간의 흐름 기억
        self.encoder_lstm = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)

        # [디코더] LSTM: 압축된 시간 흐름 복원
        self.decoder_lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        
        # [디코더] CNN: 원래 길이로 복원
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=features, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        # 입력 차원: (batch, channels=1, seq_len=128)
        x = self.encoder_cnn(x)              # -> (batch, 32, 32)
        x = x.transpose(1, 2)                # -> (batch, 32, 32) (LSTM용 축 변환)
        x, _ = self.encoder_lstm(x)   # -> (batch, 32, 16)
        x, _ = self.decoder_lstm(x)          # -> (batch, 32, 32)
        x = x.transpose(1, 2)                # -> (batch, 32, 32) (CNN용 축 변환)
        x = self.decoder_cnn(x)              # -> (batch, 1, 128)
        return x

# 2. 전용 학습기(Trainer)
class CNNLSTMAutoEncoderTrainer:
    def __init__(self, sensor_type: str):
        self.sensor_type = sensor_type
        self.window_size = 128 

    def train(self, df: pd.DataFrame) -> str:
        print(f"--- [TRAIN] {self.sensor_type} CNN-LSTM AutoEncoder 학습 시작 ---")
        # 센서 타입에 따른 특징(Features) 개수 설정
        features = 3 if self.sensor_type.lower() == "adxl" else 1

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) < features:
            raise ValueError("학습할 수 있는 숫자형 데이터가 없습니다.")
        
       # 2. 데이터 가공 및 Reshape (Conv1d용 차원 맞추기)
        if features == 3:
            vals = df[num_cols[:3]].values
            num_windows = len(vals) // self.window_size
            data_chopped = vals[:num_windows * self.window_size]
            # (Batch, 128, 3) -> (Batch, 3, 128)
            data_matrix = data_chopped.reshape(num_windows, self.window_size, 3).transpose(0, 2, 1)
        else:
            vals = df[num_cols[0]].values
            num_windows = len(vals) // self.window_size
            data_chopped = vals[:num_windows * self.window_size]
            data_matrix = data_chopped.reshape(num_windows, 1, self.window_size)

        # 🌟 3. 정규화 및 텐서 변환
        max_val = np.max(np.abs(data_matrix))
        if max_val == 0: max_val = 1
        data_normalized = data_matrix / max_val

        tensor_x = torch.tensor(data_normalized, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_x)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 🌟 4. 모델 초기화 (features 전달)
        model = CNNLSTMAutoEncoder(seq_len=self.window_size, features=features)
        criterion = nn.MSELoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 5. 학습 루프
        epochs = 20
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")

        # 6. 저장 (모델 + 매핑 정보)
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        timestamp = int(time.time())
        file_name = f"{self.sensor_type}_cnnlstm_ae_{timestamp}.pt"
        file_path = os.path.join(model_dir, file_name)
        
        torch.save(model.state_dict(), file_path)
        
        # 예측 시 일관성을 위한 매핑 파일
        mapping_path = os.path.join(model_dir, f"{self.sensor_type}_cnnlstm_ae_{timestamp}_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump({"max_val": float(max_val)}, f)

        print(f"--- [SUCCESS] 하이브리드 모델 저장 완료: {file_path} ---")
        return file_path