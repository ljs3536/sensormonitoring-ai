# sensor-ai/architectures/autoencoder.py
import os
import time
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
            nn.Linear(64, 16),
            nn.ReLU()
        )
        # 디코더: 압축된 패턴을 다시 원래 데이터로 복원 시도
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
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
        
        # 1. 데이터 전처리 (DataFrame에서 숫자 데이터만 추출)
        # 보통 InfluxDB에서 가져오면 '_value' 컬럼이나 'value', 'x', 'y' 등이 있습니다.
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("학습할 수 있는 숫자형 데이터가 없습니다.")
        
        values = df[num_cols[0]].values # 첫 번째 숫자 컬럼 사용
        
        # 데이터 개수 검증
        if len(values) < self.window_size:
            raise ValueError(f"데이터가 너무 적습니다. (현재: {len(values)} / 필요: {self.window_size})")

        # 128개씩 묶어서 매트릭스 형태로 만들기
        num_windows = len(values) // self.window_size
        data_chopped = values[:num_windows * self.window_size]
        data_matrix = data_chopped.reshape(-1, self.window_size)

        # 정규화 (Min-Max 스케일링으로 범위를 줄여줌)
        max_val = np.max(np.abs(data_matrix))
        if max_val == 0: max_val = 1
        data_normalized = data_matrix / max_val

        # PyTorch 학습용 텐서(Tensor)로 변환
        tensor_x = torch.tensor(data_normalized, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_x) # AutoEncoder는 입력과 정답이 동일함
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 2. PyTorch 모델 초기화 및 세팅
        model = SensorAutoEncoder(input_size=self.window_size)
        criterion = nn.MSELoss() # 오차 계산 방식: 평균제곱오차
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 3. 학습 루프 진행 (실제로 CPU/GPU를 갈궈서 똑똑해지는 과정)
        epochs = 20  # 전체 데이터를 20번 반복 학습
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
            
            # 5번 에포크마다 진행 상황 출력
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

        # 4. 학습 완료된 모델 파일 저장 (.pt 형식)
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # 파일명 예시: models/piezo_autoencoder_167999999.pt
        file_name = f"{self.sensor_type}_autoencoder_{int(time.time())}.pt"
        file_path = os.path.join(model_dir, file_name)
        
        # 모델의 가중치(지능) 저장
        torch.save(model.state_dict(), file_path)
        print(f"--- [TRAIN] 모델 생성 완료! 저장 위치: {file_path} ---")

        return file_path