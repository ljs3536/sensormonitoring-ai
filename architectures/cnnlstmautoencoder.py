# sensor-ai/architectures/cnnlstmautoencoder.py
import os
import time
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
        
        # [인코더] CNN: 국소적 특징 추출 및 길이 압축
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
        x, (hn, cn) = self.encoder_lstm(x)   # -> (batch, 32, 16)
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
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("학습할 수 있는 숫자형 데이터가 없습니다.")
        
        values = df[num_cols[0]].values 
        
        if len(values) < self.window_size:
            raise ValueError(f"데이터가 너무 적습니다. (현재: {len(values)} / 필요: {self.window_size})")

        num_windows = len(values) // self.window_size
        data_chopped = values[:num_windows * self.window_size]
        
        # 1D CNN 입력을 위해 (데이터 수, 1채널, 128길이) 형태로 Reshape
        data_matrix = data_chopped.reshape(-1, 1, self.window_size)

        # Min-Max 스케일링
        max_val = np.max(np.abs(data_matrix))
        if max_val == 0: max_val = 1
        data_normalized = data_matrix / max_val

        # PyTorch 텐서 변환
        tensor_x = torch.tensor(data_normalized, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_x)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = CNNLSTMAutoEncoder(seq_len=self.window_size)
        criterion = nn.MSELoss() 
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
        file_name = f"{self.sensor_type}_cnnlstm_ae_{int(time.time())}.pt"
        file_path = os.path.join(model_dir, file_name)
        
        torch.save(model.state_dict(), file_path)
        print(f"--- [TRAIN] 하이브리드 모델 생성 완료! 저장 위치: {file_path} ---")

        return file_path