import torch
import torch.nn as nn
import numpy as np
import os
import random

# 앞서 만든 CNNEncoder와 100% 동일한 인코더 사용 (성능 비교를 위해 통일)
class CNNEncoder(nn.Module):
    def __init__(self, input_dim=None, embed_dim=64):
        super(CNNEncoder, self).__init__()
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
            nn.AdaptiveAvgPool1d(1) 
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

class FewShotPrototypicalDetector:
    # 🌟 k_shots=5 추가: 5개만 뽑아서 중심점을 만들겠다는 뜻!
    def __init__(self, input_dim=None, embed_dim=64, k_shots=5, model_dir="models/fewshot"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.k_shots = k_shots
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

    #  새로운 중심점 덮어씌우기
    def set_center(self, new_center_tensor):
        self.prototypes = new_center_tensor

    def fit(self, X, y, epochs=100, lr=0.001, seed=42):
        # 재현성을 위해 시드 고정
        random.seed(seed)
        
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
            dists = torch.sum((embeddings - fixed_center) ** 2, dim=1)
            loss = torch.mean(dists)
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            final_embeds = self.model(X_tensor)
            
            # 랜덤 샘플링 제거! 
            # 서비스 로직에서 이미 정예 멤버(예: 100개)만 추려서 X에 넣어줬으므로 그냥 다 씁니다.
            actual_shots = final_embeds.shape[0] 
            
            # 들어온 모든 데이터의 평균을 중심점(Prototype)으로 사용
            final_center = torch.mean(final_embeds, dim=0)
            self.prototypes[0] = final_center.cpu().numpy()
            
            # 임계값 계산
            dists_np = torch.norm(final_embeds - final_center, dim=1).cpu().numpy()
            calc_mean = float(np.mean(dists_np))
            calc_std = float(np.std(dists_np))
            
            # 비율 기반 하한선(5%) 적용
            min_std_ratio = calc_mean * 0.05 
            final_std = max(calc_std, min_std_ratio, 0.0001)

            self.thresholds[0] = {
                'mean': calc_mean,
                'std': final_std if actual_shots > 1 else 1.0
            }
        #  5개 랜덤 추출
        # self.model.eval()
        # with torch.no_grad():
        #     final_embeds = self.model(X_tensor)
            
        #     # 전체 데이터 수가 k_shots보다 작으면 전체를 씀, 아니면 k_shots만큼 랜덤 샘플링
        #     total_samples = final_embeds.shape[0]
        #     actual_shots = min(self.k_shots, total_samples)
            
        #     # 인덱스를 랜덤하게 뽑음
        #     sampled_indices = random.sample(range(total_samples), actual_shots)
        #     sampled_embeds = final_embeds[sampled_indices]
            
        #     #  5개 데이터의 평균만 내서 중심점(Prototype)으로 사용!
        #     final_center = torch.mean(sampled_embeds, dim=0)
        #     self.prototypes[0] = final_center.cpu().numpy()
            
        #     # 임계값(Threshold)도 5개 데이터에 대해서만 계산
        #     dists_np = torch.norm(final_embeds - final_center, dim=1).cpu().numpy()
        
        #     calc_mean = float(np.mean(dists_np))
        #     calc_std = float(np.std(dists_np))
        #     min_std_ratio = calc_mean * 0.05 
        #     final_std = max(calc_std, min_std_ratio, 0.0001)

        #     self.thresholds[0] = {
        #         'mean': calc_mean,
        #         'std': final_std if actual_shots > 1 else 1.0
        #     }

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(X_tensor).cpu().numpy()

        results = []
        proto_0 = self.prototypes[0]
        thresh_mean = self.thresholds[0]['mean']
        thresh_std = self.thresholds[0]['std']
        
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
                
                # 텍스트 생성
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
            'thresholds': self.thresholds,
            'k_shots': self.k_shots
        }, path)
        return path

    def load(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {file_path}")
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
        
        self.input_dim = checkpoint.get('input_dim', 320) 
        self.model = CNNEncoder(self.input_dim, self.embed_dim).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.prototypes = checkpoint['prototypes']
        self.thresholds = checkpoint['thresholds']
        self.k_shots = checkpoint.get('k_shots', 5)