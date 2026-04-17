# sensor-ai/architectures/pinn_cnnlstmautoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
from torch.utils.data import DataLoader
from preprocess import TimeSeriesDataset

class PINN_CNNLSTMAutoEncoder(nn.Module):
    def __init__(self, seq_len=128, features=1):
        super(PINN_CNNLSTMAutoEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.lstm_enc = nn.LSTM(input_size=16, hidden_size=8, batch_first=True)
        self.lstm_dec = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
        
        self.deconv1 = nn.ConvTranspose1d(in_channels=16, out_channels=features, kernel_size=2, stride=2)
        
        # 🌟 수정 1: Sigmoid 대신 Tanh 사용 (진동은 -, + 를 모두 표현해야 함)
        self.tanh = nn.Tanh() 

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = x.transpose(1, 2)
        x, (hn, cn) = self.lstm_enc(x)
        
        x, _ = self.lstm_dec(x)
        x = x.transpose(1, 2)
        
        x = self.deconv1(x)
        x = self.tanh(x) # 🌟 Tanh 적용
        
        x = x.transpose(1, 2)
        return x

class PINN_CNNLSTMAutoEncoderTrainer:
    def __init__(self, sensor_type: str):
        self.sensor_type = sensor_type
        self.seq_len = 128
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PINN_CNNLSTMAutoEncoder(seq_len=self.seq_len, features=1 if sensor_type=="piezo" else 3).to(self.device)

    def calculate_physics_loss(self, x_pred, dt, k, c, m=1.0):
        # 🌟 수정 2: dt 스케일 안정화
        # 실제 dt(예: 0.001)를 쓰면 가속도가 10^6으로 폭발하므로, 신경망 내부 스케일에 맞춰 정규화된 시간(예: dt=1)을 사용합니다.
        # 물리 파라미터(k, c)의 상대적 비율만 유지하면 물리적 파동 형태를 학습할 수 있습니다.
        norm_dt = 1.0 
        
        v = (x_pred[:, 2:, :] - x_pred[:, :-2, :]) / (2 * norm_dt)
        a = (x_pred[:, 2:, :] - 2 * x_pred[:, 1:-1, :] + x_pred[:, :-2, :]) / (norm_dt ** 2)
        x_inner = x_pred[:, 1:-1, :]
        
        residual = (m * a) + (c * v) + (k * x_inner)
        physics_loss = torch.mean(residual ** 2)
        return physics_loss

    def train(self, df, epochs=50, batch_size=32, sensor_metadata=None):
        sampling_rate = sensor_metadata.get("sampling_rate", 1000) if sensor_metadata else 1000
        dt = 1.0 / sampling_rate
        # 스케일이 조정된 가상의 강성과 감쇠 계수
        k = 0.5 
        c = 0.1 
        
        features = ["voltage"] if self.sensor_type == "piezo" else ["x", "y", "z"]
        raw_data = df[features].values
        
        max_val = np.max(np.abs(raw_data))
        if max_val == 0: max_val = 1.0
        
        # 🌟 수정 3: Sigmoid용 0~1 정규화가 아닌, 진동에 맞는 -1~1 정규화
        normalized_data = raw_data / max_val 

        dataset = TimeSeriesDataset(normalized_data, self.seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        mse_loss_fn = nn.MSELoss()
        
        # 🌟 수정 4: 물리 손실 가중치 하향 조정
        # 초반에는 데이터 형태(Recon)부터 외우도록 물리 가중치를 아주 작게 줍니다.
        physics_weight = 0.5 

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                
                batch_pred = self.model(batch_x)
                
                recon_loss = mse_loss_fn(batch_pred, batch_x)
                physics_loss = self.calculate_physics_loss(batch_pred, dt, k, c)
                
                loss = recon_loss + (physics_weight * physics_loss)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Recon Loss: {recon_loss.item():.6f} | Physics Loss: {physics_loss.item():.6f}")

        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        timestamp = int(time.time())
        file_name = f"{self.sensor_type}_pinn_cnnlstmae__{timestamp}.pt"
        file_path = os.path.join(model_dir, file_name)
        torch.save(self.model.state_dict(), file_path)
        
        mapping_path = os.path.join(model_dir, f"{self.sensor_type}_pinn_cnnlstmae_{timestamp}_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump({"max_val": float(max_val), "k": float(k), "c": float(c)}, f)

        return file_path