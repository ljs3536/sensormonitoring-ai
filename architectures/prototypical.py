import torch
import torch.nn as nn
import numpy as np
import os

# 🌟 심장 교체: Linear -> 1D-CNN (시계열 패턴 스캔)
class CNNEncoder(nn.Module):
    def __init__(self, input_dim=None, embed_dim=64):
        super(CNNEncoder, self).__init__()
        
        # Conv1d를 사용하여 시퀀스의 패턴(Peak)을 스캔합니다.
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            # 🌟 기존 코드의 핵심: 데이터 길이가 128이든 320이든 무조건 1칸으로 평균을 냄!
            # 이것 덕분에 Input Dimension이 달라도 모델이 터지지 않습니다.
            nn.AdaptiveAvgPool1d(1) 
        )
        
        # 압축된 특징을 최종 임베딩 차원(64)으로 정제
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        # 파이토치 Conv1d는 (배치, 채널, 길이) 형태를 요구하므로 채널(1)을 중간에 삽입
        # 예: (Batch, 320) -> (Batch, 1, 320)
        x = x.unsqueeze(1) 
        
        x = self.features(x)      # 결과: (Batch, 64, 1)
        x = x.view(x.size(0), -1) # 결과: (Batch, 64) - 1차원으로 쫙 폄
        x = self.fc(x)            # 결과: (Batch, embed_dim)
        return x

# 🌟 기존과 동일한 MLOps 관리자 뼈대
class PrototypicalLeakDetector:
    def __init__(self, input_dim=None, embed_dim=64, model_dir="models/leak"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.input_dim = input_dim
        self.model = None
        self.prototypes = {}
        self.thresholds = {}

    # 현재 중심점 가져오기 (단수형 -> 복수형으로 수정)
    def get_center(self):
        # self.prototypes 가 비어있는지 확인
        if not hasattr(self, 'prototypes') or self.prototypes is None:
            raise ValueError("모델 중심점(prototypes)이 초기화되지 않았습니다.")
        
        # 텐서 타입인 self.prototypes를 그대로 반환합니다.
        # (만약 딕셔너리나 리스트 형태로 감싸져 있다면, 인덱싱 [0] 이 필요할 수도 있습니다)
        return self.prototypes 

    # 새로운 중심점 덮어씌우기
    def set_center(self, new_center_tensor):
        self.prototypes = new_center_tensor
        
    def fit(self, X, y, epochs=100, lr=0.001):
        if self.model is None:
            self.input_dim = X.shape[1]
            self.model = CNNEncoder(self.input_dim, self.embed_dim).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            initial_embeds = self.model(X_tensor)
            fixed_center = torch.mean(initial_embeds, dim=0).detach()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = self.model(X_tensor)

            # 정상 데이터들을 중심점으로 모으기
            dists = torch.sum((embeddings - fixed_center) ** 2, dim=1)
            loss = torch.mean(dists)

            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            final_embeds = self.model(X_tensor)
            final_center = torch.mean(final_embeds, dim=0)
            self.prototypes[0] = final_center.cpu().numpy()
            
            dists_np = torch.norm(final_embeds - final_center, dim=1).cpu().numpy()
        
            calc_mean = float(np.mean(dists_np))
            calc_std = float(np.std(dists_np))
                        
            # 평균의 일정 비율(예: 5%)을 최소 std로 보장 (권장)
            min_std_ratio = calc_mean * 0.05 
            final_std = max(calc_std, min_std_ratio, 0.0001)

            self.thresholds[0] = {
                'mean': calc_mean,
                'std': final_std
            }

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(X_tensor).cpu().numpy()

        results = []
        proto_0 = self.prototypes[0]
        thresh_mean = self.thresholds[0]['mean']
        thresh_std = self.thresholds[0]['std']
        
        # 3-Sigma Rule 통계적 울타리
        anomaly_limit = thresh_mean + (3.0 * thresh_std)

        for i, emb in enumerate(embeddings):
            dist = np.linalg.norm(emb - proto_0)
            
            reason = ""
            if dist <= anomaly_limit:
                prob = 0.49 * (dist / (anomaly_limit + 1e-6))
                is_leak = "N"
            else:
                excess = dist / (anomaly_limit + 1e-6)
                prob = min(0.99, 0.50 + 0.1 * excess)
                is_leak = "Y"
                raw_data = X[i]
                peak_idx = int(np.argmax(raw_data)) # 가장 값이 큰 인덱스
                peak_val = float(raw_data[peak_idx])
                reason = f"정상 범위를 초과했습니다. (특징점: {peak_idx}번째 주파수 대역에서 {peak_val:.4f}의 비정상적 진동폭 감지)"
                
            results.append({"prob": float(prob), "is_leak": is_leak, "reason": reason})

        return results

    def save(self, file_name):
        if not file_name.endswith('.pt'):
            file_name += '.pt'

        path = os.path.join(self.model_dir, file_name)
        torch.save({
            'input_dim': self.input_dim, 
            'state_dict': self.model.state_dict(),
            'prototypes': self.prototypes,
            'thresholds': self.thresholds
        }, path)
        return path

    def load(self, file_path): # 🌟 수정됨: 정확한 파일 경로를 직접 받음
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {file_path}")
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
        
        self.input_dim = checkpoint.get('input_dim', 320) 
        # 불러올 때도 CNNEncoder 로 생성
        self.model = CNNEncoder(self.input_dim, self.embed_dim).to(self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.prototypes = checkpoint['prototypes']
        self.thresholds = checkpoint['thresholds']