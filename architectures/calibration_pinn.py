# sensor-ai/architectures/calibration_pinn.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from preprocess import TimeSeriesDataset
from database_rdb import get_db, SessionLocal
from models import AiModel
import os 

class InversePINN_Calibrator(nn.Module):
    def __init__(self, seq_len=128, features=1, init_k=1.0, init_c=0.1):
        super(InversePINN_Calibrator, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.lstm_enc = nn.LSTM(input_size=16, hidden_size=8, batch_first=True)
        self.lstm_dec = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
        
        self.deconv1 = nn.ConvTranspose1d(in_channels=16, out_channels=features, kernel_size=2, stride=2)
        self.tanh = nn.Tanh() 

        # 🌟 핵심 1: 파라미터 초기화 (외부에서 대략적인 값을 던져줄 수 있음)
        self.raw_k = nn.Parameter(torch.tensor([init_k], dtype=torch.float32))
        self.raw_c = nn.Parameter(torch.tensor([init_c], dtype=torch.float32))

    def get_physics_params(self):
        # 🌟 핵심 2: Softplus 적용 (k, c가 절대로 음수가 되지 않도록 방어)
        # 물리 법칙에서 강성이나 감쇠가 음수가 되면 발산(폭발)해버립니다.
        k = F.softplus(self.raw_k)
        c = F.softplus(self.raw_c)
        return k, c

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = x.transpose(1, 2)
        x, (hn, cn) = self.lstm_enc(x)
        
        x, _ = self.lstm_dec(x)
        x = x.transpose(1, 2)
        
        x = self.deconv1(x)
        x = self.tanh(x)
        
        x = x.transpose(1, 2)
        return x


class InversePINN_CalibratorTrainer:
    def __init__(self, sensor_type: str):
        self.sensor_type = sensor_type
        self.seq_len = 128
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = 3 if sensor_type.lower() == "adxl" else 1
        
        # 보편적인 초기값으로 세팅
        self.model = InversePINN_Calibrator(
            seq_len=self.seq_len, 
            features=self.features,
            init_k=0.5, 
            init_c=0.01
        ).to(self.device)

    def calculate_physics_loss(self, x_pred):
        norm_dt = 1.0 # 범용 모델과 동일하게 정규화된 스케일 사용
        
        v = (x_pred[:, 2:, :] - x_pred[:, :-2, :]) / (2 * norm_dt)
        a = (x_pred[:, 2:, :] - 2 * x_pred[:, 1:-1, :] + x_pred[:, :-2, :]) / (norm_dt ** 2)
        x_inner = x_pred[:, 1:-1, :]
        
        # 모델 내부에서 현재 학습 중인 k, c 값을 꺼내옵니다.
        k, c = self.model.get_physics_params()
        
        residual = (1.0 * a) + (c * v) + (k * x_inner)
        return torch.mean(residual ** 2)

    def calibrate(self, df, pretrained_model_path: str = None, epochs=50, batch_size=32):
        print("[Auto-Calibration] 물리 파라미터 최적화 시작...")
        
        # 1. 데이터 전처리 (기존 범용 모델 학습과 완벽히 동일하게 영점 조절)
        features_cols = ["voltage"] if self.sensor_type == "piezo" else ["x", "y", "z"]
        raw_data = df[features_cols].values
        
        mean_val = np.mean(raw_data, axis=0)
        centered_data = raw_data - mean_val
        max_val = np.max(np.abs(centered_data), axis=0)
        max_val = np.where(max_val == 0, 1.0, max_val)
        normalized_data = centered_data / max_val 

        dataset = TimeSeriesDataset(normalized_data, self.seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        mse_loss_fn = nn.MSELoss()
        
        # =========================================================
        #  3. 사전 학습 모델 유무에 따른 동적 학습 전략 (Two-Track)
        # =========================================================
        has_pretrained = pretrained_model_path and os.path.exists(pretrained_model_path)

        if has_pretrained:
            print(f"[INFO] 최신 범용 모델 로드 완료: {pretrained_model_path}")
            print("[MODE] 네트워크 동결 & 물리 파라미터 집중 최적화 모드")
            
            self.model.load_state_dict(torch.load(pretrained_model_path, weights_only=True), strict=False)
            
            # 네트워크 가중치 동결 (k, c만 제외)
            for name, param in self.model.named_parameters():
                if "raw_k" not in name and "raw_c" not in name:
                    param.requires_grad = False
            
            optimizer = optim.Adam([
                {'params': [self.model.raw_k], 'lr': 0.05},
                {'params': [self.model.raw_c], 'lr': 0.2}
            ])
            use_warmup = False
            
        else:
            print("[WARNING] 사용 가능한 사전 학습 모델이 없습니다.")
            print("[MODE] 빈 도화지(Random Weights)부터 네트워크와 물리 파라미터 동시 학습")
            
            optimizer = optim.Adam([
                {'params': self.model.conv1.parameters()},
                {'params': self.model.lstm_enc.parameters()},
                {'params': self.model.lstm_dec.parameters()},
                {'params': self.model.deconv1.parameters()},
                {'params': [self.model.raw_k], 'lr': 0.05},
                {'params': [self.model.raw_c], 'lr': 0.2}
            ], lr=0.001)
            use_warmup = True
            
        self.model.train()
        for epoch in range(epochs):
            warmup_epochs = int(epochs * 0.3) if use_warmup else 0
            total_loss = 0

            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device).float()
                optimizer.zero_grad()
                
                batch_pred = self.model(batch_x)
                recon_loss = mse_loss_fn(batch_pred, batch_x)
                
                # Warm-up 적용 여부에 따른 Loss 계산
                if use_warmup and epoch < warmup_epochs:
                    loss = recon_loss 
                else:
                    physics_loss = self.calculate_physics_loss(batch_pred)
                    if has_pretrained:
                        # 동결 모드에서는 물리 손실에 올인 (혹은 recon을 약간만 섞음)
                        loss = physics_loss + (0.1 * recon_loss)
                    else:
                        # 동시 학습 모드
                        loss = recon_loss + (1.0 * physics_loss)
                
                loss.backward()
                
                # Warm-up 기간 중 k, c 업데이트 방지
                if use_warmup and epoch < warmup_epochs:
                    self.model.raw_k.grad = None
                    self.model.raw_c.grad = None
                    
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                current_k, current_c = self.model.get_physics_params()
                print(f"Epoch {epoch+1}/{epochs} | K: {current_k.item():.4f} | C: {current_c.item():.4f} | Loss: {total_loss/len(dataloader):.6f}")

        final_k, final_c = self.model.get_physics_params()
        print(f"[Auto-Calibration 완료] 추천 물리 값 -> k: {final_k.item():.4f}, c: {final_c.item():.4f}")
        
        # =========================================================
        # 🌟 5. 정상 데이터 기반 Auto-Threshold (임계치) 자동 추출
        # =========================================================
        print("[Auto-Threshold] 정상 데이터의 기준 잔차(Baseline) 측정 중...")
        self.model.eval() # 평가 모드 전환
        max_physics_losses = []
        
        with torch.no_grad():
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device).float()
                batch_pred = self.model(batch_x)
                
                # 예측 모드(predict_engine.py)와 완벽히 동일한 방식으로 잔차 계산
                norm_dt = 1.0
                v = (batch_pred[:, 2:, :] - batch_pred[:, :-2, :]) / (2 * norm_dt)
                a = (batch_pred[:, 2:, :] - 2 * batch_pred[:, 1:-1, :] + batch_pred[:, :-2, :]) / (norm_dt ** 2)
                x_inner = batch_pred[:, 1:-1, :]
                
                # 최종 k, c를 적용한 물리 잔차 (batch_size, seq_len-2, features)
                residual = (1.0 * a) + (final_c.item() * v) + (final_k.item() * x_inner)
                physics_error_tensor = residual ** 2
                
                # 배치 내에서 가장 컸던 Max Loss 추출
                batch_max_loss = torch.max(physics_error_tensor).item()
                max_physics_losses.append(batch_max_loss)

        # 전체 정상 데이터 중에서 가장 높게 튀었던 잔차값
        baseline_max_loss = max(max_physics_losses)
        
        #  여유율(Margin) 부여: 정상 최고치의 2.0배 ~ 3.0배를 임계치로 설정
        # 너무 타이트하면 정상 노이즈에도 알람이 울리므로 (False Alarm 방지)
        recommended_threshold_max = baseline_max_loss * 2.5 

        print(f" [Auto-Threshold 완료] 기준 Max: {baseline_max_loss:.5f} -> 추천 임계치: {recommended_threshold_max:.5f}")
        
        # 기존 리턴 값에 threshold 추가
        return round(final_k.item(), 4), round(final_c.item(), 4), round(recommended_threshold_max, 5)