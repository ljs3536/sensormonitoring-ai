import torch
import torch.nn as nn
import numpy as np
import os

class FFTEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64):
        super(FFTEncoder, self).__init__()
        # 기존 CNN과 달리 FFT는 이미 주파수 특성이 추출된 상태이므로 Linear가 훨씬 빠르고 강력합니다.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(), # 기존 코드에서 쓰시던 GELU 반영
            nn.Dropout(0.2),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

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

    def fit(self, X, y, epochs=100, lr=0.001):
        if self.model is None:
            self.input_dim = X.shape[1]
            self.model = FFTEncoder(self.input_dim, self.embed_dim).to(self.device)

        # 🌟 핵심 방어 1: weight_decay(L2 정규화)를 줘서 가중치가 0이 되는 모드 붕괴 방지
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # 🌟 핵심 방어 2: 학습 시작 전에 중심점(Center)을 고정 (Deep SVDD 기법)
        self.model.eval()
        with torch.no_grad():
            initial_embeds = self.model(X_tensor)
            fixed_center = torch.mean(initial_embeds, dim=0).detach()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = self.model(X_tensor)

            # 정상 데이터들이 고정된 중심점(fixed_center)으로 뭉치도록 유도
            dists = torch.sum((embeddings - fixed_center) ** 2, dim=1)
            loss = torch.mean(dists)

            loss.backward()
            optimizer.step()

        # 🌟 기존 make_prototype_v1.py 의 로직과 동일 (학습 후 임베딩 평균을 Prototype으로 저장)
        self.model.eval()
        with torch.no_grad():
            final_embeds = self.model(X_tensor)
            final_center = torch.mean(final_embeds, dim=0)
            self.prototypes[0] = final_center.cpu().numpy()
            
            # 🌟 통계적 임계값(Threshold) 계산 : 거리의 평균과 표준편차 저장
            dists_np = torch.norm(final_embeds - final_center, dim=1).cpu().numpy()
            self.thresholds[0] = {
                'mean': float(np.mean(dists_np)),
                'std': float(np.std(dists_np))
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
        
        # 🌟 통계학의 3-Sigma Rule: 평균에서 표준편차의 3배를 넘어가면 99.7% 확률로 이상(누출) 데이터
        anomaly_limit = thresh_mean + (3.0 * thresh_std)

        for emb in embeddings:
            dist = np.linalg.norm(emb - proto_0)
            
            if dist <= anomaly_limit:
                # 한계치 이내면 정상(N) -> 확률 1% ~ 49%로 스케일링
                prob = 0.49 * (dist / (anomaly_limit + 1e-6))
                is_leak = "N"
            else:
                # 한계치를 넘으면 누출(Y) -> 확률 50% ~ 99%로 스케일링
                excess = dist / (anomaly_limit + 1e-6)
                prob = min(0.99, 0.50 + 0.1 * excess)
                is_leak = "Y"
                
            results.append({"prob": float(prob), "is_leak": is_leak})

        return results

    def save(self, sensor_id):
        path = os.path.join(self.model_dir, f"proto_model_{sensor_id}.pt")
        # 기존의 ckpt 파일과 prototype.npy를 이 딕셔너리 하나로 합쳐서 저장!
        torch.save({
            'input_dim': self.input_dim, 
            'state_dict': self.model.state_dict(),
            'prototypes': self.prototypes,
            'thresholds': self.thresholds
        }, path)
        return path

    def load(self, sensor_id):
        path = os.path.join(self.model_dir, f"proto_model_{sensor_id}.pt")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.input_dim = checkpoint.get('input_dim', 320) 
        self.model = FFTEncoder(self.input_dim, self.embed_dim).to(self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.prototypes = checkpoint['prototypes']
        self.thresholds = checkpoint['thresholds']