# 🧠 Sensor AI
> PyTorch 기반의 시계열 센서 데이터 분석 및 이상 탐지 엔진

이 모듈은 InfluxDB에 축적된 센서 데이터를 학습하여 장비의 상태를 분류(Classification)하거나 이상 징후(Anomaly Detection)를 탐지합니다. 시계열 데이터의 시간적, 공간적 특징을 동시에 추출하기 위해 CNN-LSTM 하이브리드 구조를 채택했습니다.

## 🛠 Tech Stack
- **Framework:** FastAPI, PyTorch
- **Architecture:** CNN-LSTM, AutoEncoder
- **Data Handling:** NumPy, Pandas, Scipy
- **Database :** InfluxDB 2.7 (Time-series), MariaDB (Relational)

## 🚀 Getting Started
### 1. Infrastructure (Database Setup)
#### A. InfluxDB (Docker)
시계열 데이터 저장을 위해 InfluxDB 2.7 버전을 사용합니다. 데이터 영속성을 위해 볼륨 매핑을 적용합니다.
```
docker run -d \
  --name influxdb \
  -p 8086:8086 \
  -v influxdb2_data:/var/lib/influxdb2 \
  influxdb:2.7
```
- Web UI: http://localhost:8086 (admin / admin1234)

- Organization: sensor_hq

- Bucket: sensor_data
#### B. MariaDB (Docker)
AI 모델의 메타데이터 및 Soft Delete 상태 관리를 위해 사용합니다.
```
docker-compose up -d mariadb
```

### 2. Environment Setup
```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
### 3. Running Inference Server
AI 모듈은 FastAPI를 통해 백엔드와 통신하며 추론 서비스를 제공합니다.
```
uvicorn main:app --reload --port 8002
```

## 🏗 Model Architectures
### 1. CNN-LSTM Classifier (지도 학습)
목적: 정상(Normal)과 다양한 결함(Anomaly) 상태를 분류.
특징: CNN 레이어를 통해 주파수 특징을 추출하고, LSTM 레이어로 시간적 문맥을 파악합니다.
Output: 클래스별 확률 분포(Probabilities) 및 확신도(Confidence).

### 2. AutoEncoder (비지도 학습)
목적: 라벨이 없는 데이터에서 미지의 이상 징후 탐지.
특징: 데이터를 압축 후 복원하는 과정에서 발생하는 **Reconstruction Error(MSE)**를 기반으로 이상 점수(Anomaly Score)를 산출합니다.

## 📊 Data Pipeline & Normalization
이 프로젝트의 핵심은 학습과 추론 단계의 데이터 정규화 일관성입니다.
- Preprocessing: 모든 데이터는 $x_{norm} = \frac{x}{max\_val}$ 방식을 통해 정규화됩니다.
- Consistency: 학습 시 산출된 max_val을 모델 가중치와 함께 _mapping.json 파일에 저장합니다.
- Inference: 추론 시 저장된 max_val을 불러와 동일한 스케일로 데이터를 변환함으로써 오판 가능성을 최소화했습니다.

## 📂 Project Structure
- architectures/: 모델 정의 (AutoEncoder, CNN-LSTM 등)
- models/: 학습 완료된 모델 가중치(.pt) 및 매핑 정보(.json) 저장소
- train_engine.py: InfluxDB 데이터 로드 및 모델 학습 로직
- predict_engine.py: 저장된 모델을 로드하여 추론 수행
  
