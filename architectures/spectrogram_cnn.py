import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy import stats

# 🌟 상위 폴더의 preprocess.py에서 변환기 가져오기
from preprocess import SpectrogramTransformer

# [A] 2D CNN 모델 정의
class SpectrogramCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(SpectrogramCNN, self).__init__()
        # 입력: [batch, 1, 64, 9] (흑백 이미지 1장)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # -> [16, 32, 4]
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # -> [32, 16, 2]
        )
        
        # 🌟 계산: 32(채널) * 8(H) * 2(W) = 512
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 2, 64), # 👈 1024에서 512로 수정 (32 * 16 * 2 -> 32 * 8 * 2)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# [B] 2D CNN 전용 데이터셋 관리자
class SpectrogramDataset(Dataset):
    def __init__(self, data_matrix, labels, transformer):
        self.data = data_matrix
        self.labels = labels
        self.transformer = transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1D 데이터를 꺼내서 즉시 2D 스펙트로그램으로 변환
        x_1d = self.data[idx]
        x_2d = self.transformer(x_1d)
        y = self.labels[idx]
        return x_2d, y

# [C] 2D CNN 전용 학습기 (Trainer)
class SpectrogramCNNTrainer:
    def __init__(self, sensor_type: str):
        self.sensor_type = sensor_type
        self.window_size = 128
        self.transformer = SpectrogramTransformer(sample_rate=1000, n_mels=32)

    def train(self, df: pd.DataFrame) -> str:
        print(f"--- [TRAIN] {self.sensor_type} CNN 학습 시작 ---")
        is_adxl = self.sensor_type.lower() == "adxl"
        in_channels = 3 if is_adxl else 1

        # 1. 라벨 처리
        unique_labels = df['label'].unique().tolist()
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        
        # 2. 데이터 윈도우 분할 (1D)
        num_cols = df.select_dtypes(include=[np.number]).columns
        labels = df['label'].map(label_to_index).values
        
        # 🌟 2. 데이터 추출 및 차원 변형 (ADXL 대응)
        if is_adxl:
            # X, Y, Z 세 컬럼을 가져옴
            vals = df[num_cols[:3]].values 
            num_windows = len(vals) // self.window_size
            data_chopped = vals[:num_windows * self.window_size]
            # (Batch, 128, 3) -> (Batch, 3, 128) 로 변경하여 Transformer에 전달
            data_matrix = data_chopped.reshape(num_windows, self.window_size, 3).transpose(0, 2, 1)
        else:
            # Piezo: 1개 컬럼
            vals = df[num_cols[0]].values
            num_windows = len(vals) // self.window_size
            data_chopped = vals[:num_windows * self.window_size]
            data_matrix = data_chopped.reshape(num_windows, 1, self.window_size)

        # 라벨 처리 (최빈값)
        labels_matrix = labels[:num_windows * self.window_size].reshape(-1, self.window_size)
        final_labels = stats.mode(labels_matrix, axis=1, keepdims=False).mode.flatten()

        # 정규화
        max_val = np.max(np.abs(data_matrix))
        if max_val == 0: max_val = 1
        data_normalized = data_matrix / max_val

        # 🌟 3. Dataset 및 Model 생성 시 채널 정보 전달
        dataset = SpectrogramDataset(data_normalized, final_labels, self.transformer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 모델 생성 시 in_channels 전달
        model = SpectrogramCNN(in_channels=in_channels, num_classes=len(unique_labels))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 6. 학습 루프
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
            if (epoch+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")

        # 7. 저장
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        ts = int(time.time())
        file_path = os.path.join(model_dir, f"{self.sensor_type}_spectrogram_cnn_{ts}.pt")
        torch.save(model.state_dict(), file_path)

        # 매핑 정보 저장 (중요: max_val 포함)
        mapping_path = file_path.replace(".pt", "_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump({
                "index_to_label": {idx: lbl for lbl, idx in label_to_index.items()},
                "max_val": float(max_val),
                "model_type": "spectrogram_cnn"
            }, f)

        return file_path