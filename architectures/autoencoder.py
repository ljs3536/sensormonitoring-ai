# sensor-ai/architectures/autoencoder.py
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. PyTorch 신경망 구조 정의 (128개의 센서 데이터를 입력받음)
class SensorAutoEncoder(nn.Module):
    def __init__(self, input_size=128):
        super(SensorAutoEncoder, self).__init__()
        # 인코더: 데이터를 압축하여 핵심 패턴만 남김
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # 디코더: 압축된 패턴을 다시 원래 데이터로 복원 시도
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoderTrainer:
    def __init__(self, sensor_type: str):
        self.sensor_type = sensor_type
        self.window_size = 128 # 한 번에 분석할 데이터 묶음 크기

    def train(self, df: pd.DataFrame) -> str:
        print(f"--- [TRAIN] {self.sensor_type} 실제 PyTorch 딥러닝 학습 시작 ---")
        
        # 센서 타입에 따른 특징(Features) 및 입력 크기 결정
        features = 3 if self.sensor_type.lower() == "adxl" else 1
        input_size = self.window_size * features

        # 1. 데이터 전처리 (DataFrame에서 숫자 데이터만 추출)
        # 보통 InfluxDB에서 가져오면 '_value' 컬럼이나 'value', 'x', 'y' 등이 있습니다.

        print("📊 현재 DataFrame의 컬럼들:", df.columns.tolist())
        print(df.head())
        # 🌟 2. 데이터 가공 (ADXL은 X,Y,Z를 한 줄로 이어 붙임)
        if features == 3:
            # ADXL: x, y, z 컬럼을 찾습니다.
            # pivot 이후에는 _value가 아니라 x, y, z가 컬럼명입니다.
            adxl_cols = ['x', 'y', 'z']
            if all(col in df.columns for col in adxl_cols):
                values = df[adxl_cols].values # (N, 3)
            else:
                # 컬럼명이 다를 경우 숫자형 중 뒤에서 3개를 가져옵니다.
                num_cols = df.select_dtypes(include=[np.number]).columns
                values = df[num_cols[-3:]].values
                
            num_windows = len(values) // self.window_size
            data_chopped = values[:num_windows * self.window_size]
            data_matrix = data_chopped.reshape(num_windows, -1)
            
        else:
            # Piezo: 보통 'value'라는 이름으로 필드가 생성됩니다.
            if 'value' in df.columns:
                values = df['value'].values
            elif '_value' in df.columns:
                values = df['_value'].values
            else:
                # pivot 결과에서 result, table 등을 피하기 위해 가장 마지막 숫자 컬럼을 선택
                num_cols = df.select_dtypes(include=[np.number]).columns
                values = df[num_cols[-1]].values

            num_windows = len(values) // self.window_size
            data_chopped = values[:num_windows * self.window_size]
            data_matrix = data_chopped.reshape(-1, self.window_size)

        if len(num_cols) == 0:
            raise ValueError("학습할 수 있는 숫자형 데이터가 없습니다.")
        
        # 💡 디버깅용 로그: 진짜 데이터가 들어왔는지 확인
        print(f"✅ 추출된 데이터 샘플: {values[:5]}")
        # 🌟 3. 정규화 및 텐서 변환
        max_val = np.max(np.abs(data_matrix))
        print(f"🔥 최종 확인된 max_val: {max_val}")
        if max_val == 0: max_val = 1
        data_normalized = data_matrix / max_val

        tensor_x = torch.tensor(data_normalized, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_x) 
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 🌟 4. 모델 초기화 (결정된 input_size 전달)
        model = SensorAutoEncoder(input_size=input_size)
        criterion = nn.MSELoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 5. 학습 루프
        epochs = 500
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
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

        # 6. 모델 및 매핑 정보 저장 (Predict 시 사용)
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        timestamp = int(time.time())
        file_name = f"{self.sensor_type}_autoencoder_{timestamp}.pt"
        file_path = os.path.join(model_dir, file_name)
        
        torch.save(model.state_dict(), file_path)

        # 🌟 일관성을 위해 매핑 파일도 함께 생성
        mapping_path = os.path.join(model_dir, f"{self.sensor_type}_autoencoder_{timestamp}_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump({"max_val": float(max_val)}, f)
        print("max_val : ", max_val)
        print(f"--- [TRAIN] 모델 생성 완료! 저장 위치: {file_path} ---")
        return file_path