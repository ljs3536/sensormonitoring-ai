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
from sensors import Sensor
from database_rdb import SessionLocal

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

    def calculate_physics_loss(self, x_pred, dt, k_tensor, c_tensor, m=1.0):
        # k_tensor, c_tensor는 (batch_size, 1, 1) 형태의 텐서
        norm_dt = 1.0
        
        v = (x_pred[:, 2:, :] - x_pred[:, :-2, :]) / (2 * norm_dt)
        a = (x_pred[:, 2:, :] - 2 * x_pred[:, 1:-1, :] + x_pred[:, :-2, :]) / (norm_dt ** 2)
        x_inner = x_pred[:, 1:-1, :]
        
        # 여기서 k, c가 상수가 아니라 각 샘플에 맞는 텐서값이 곱해짐
        residual = (m * a) + (c_tensor * v) + (k_tensor * x_inner)
        return torch.mean(residual ** 2)
    
    def get_all_sensor_metadata(self, sensor_type: str):
        """MariaDB에서 해당 센서 타입의 전체 물리 정보(k, c)를 가져와 딕셔너리로 변환합니다."""
        db_session = SessionLocal()
        try:
            # 해당 타입의 모든 센서 조회
            sensors = db_session.query(Sensor).filter(Sensor.type == sensor_type).all()
            # sensor_id를 Key로 하는 딕셔너리 생성
            meta_map = {}
            for s in sensors:
                meta_map[s.id] = {
                    "k": float(s.physics_k) if hasattr(s, 'k') and s.k else 0.5,
                    "c": float(s.physics_c) if hasattr(s, 'c') and s.c else 0.01,
                    "sampling_rate": s.sampling_rate or 1000
                }
            return meta_map
        finally:
            db_session.close()

    def train(self, df, epochs=50, batch_size=128):
        # 1. 모든 센서 메타데이터 로드
        meta_map = self.get_all_sensor_metadata(self.sensor_type)
        
        # sensor_id 컬럼을 기반으로 k, c 매핑
        if 'sensor_id' not in df.columns:
            raise ValueError("DataFrame에 'sensor_id' 컬럼이 없습니다. InfluxDB 태그 설정을 확인하세요.")
            
        df['k'] = df['sensor_id'].map(lambda x: meta_map.get(x, {}).get('k', 0.5))
        df['c'] = df['sensor_id'].map(lambda x: meta_map.get(x, {}).get('c', 0.01))
        print(df)
        data_features = ["voltage"] if self.sensor_type == "piezo" else ["x", "y", "z"]
        physics_features = ["k", "c"]
        
        # 2. 데이터 준비 및 정규화
        raw_data = df[data_features].values
        # 범용 모델이므로 전체 센서 데이터의 통계치로 정규화합니다.
        mean_val = np.mean(raw_data, axis=0)
        max_val = np.max(np.abs(raw_data - mean_val), axis=0)
        max_val = np.where(max_val == 0, 1.0, max_val)
        
        normalized_vibration = (raw_data - mean_val) / max_val
        
        # k, c는 정규화된 진동 데이터와 합쳐서 Dataset에 넣습니다.
        # Shape: (N, Features + 2)
        combined_data = np.hstack([normalized_vibration, df[physics_features].values])

        dataset = TimeSeriesDataset(combined_data, self.seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        mse_loss_fn = nn.MSELoss()
        
        # 공통 dt (센서마다 크게 다르다면 이 또한 배치별로 넘겨야 함)
        # 여기서는 메타데이터 중 첫 번째 값 혹은 기본값을 사용
        base_sr = list(meta_map.values())[0]['sampling_rate'] if meta_map else 1000
        dt = 1.0 / base_sr

        self.model.train()
        for epoch in range(epochs):
            physics_weight = 0.01 if epoch >= int(epochs * 0.3) else 0.0
            
            for batch_data, in dataloader:
                batch_data = batch_data.to(self.device).float()
                
                # 데이터 분리: 앞쪽은 진동, 뒤쪽 2개는 k, c
                batch_x = batch_data[:, :, :len(data_features)] 
                batch_k = batch_data[:, 0, -2].view(-1, 1, 1) # 시퀀스 중 첫 값만 사용 (배치 내 동일)
                batch_c = batch_data[:, 0, -1].view(-1, 1, 1)
                
                optimizer.zero_grad()
                batch_pred = self.model(batch_x)
                
                recon_loss = mse_loss_fn(batch_pred, batch_x)
                # 배치의 k, c 텐서를 그대로 물리 손실 함수에 전달
                physics_loss = self.calculate_physics_loss(batch_pred, dt, batch_k, batch_c)
                
                loss = recon_loss + (physics_weight * physics_loss)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Recon: {recon_loss.item():.6f} | Physics: {physics_loss.item():.6f}")

        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        timestamp = int(time.time())
        file_path = os.path.join(model_dir, f"{self.sensor_type}_pinn_universal_{timestamp}.pt")
        torch.save(self.model.state_dict(), file_path)
        
        mapping_path = file_path.replace(".pt", "_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump({
                "mean_val": mean_val.tolist(), 
                "max_val": max_val.tolist(),
                "sensor_type": self.sensor_type,
                "is_universal": True
            }, f)

        return file_path