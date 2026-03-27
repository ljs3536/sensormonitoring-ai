# sensor-ai/main.py
# 서비스 엔트리 포인트 (FastAPI 라우팅)
from fastapi import FastAPI, BackgroundTasks
from database import AIStore  # 아까 만드신 database.py 임포트
import numpy as np
import time
import os

app = FastAPI()
db = AIStore() # DB 클라이언트 인스턴스 생성

# 모델 저장 경로 설정
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

model_status = {"status": "ready", "last_trained": None}

@app.post("/train")
async def train_model(sensor_type: str, background_tasks: BackgroundTasks, days: int = 7):
    """
    비동기로 데이터를 가져와 학습을 진행합니다.
    """
    def heavy_training_task(s_type, lookback_days):
        global model_status
        try:
            model_status["status"] = "training"
            print(f"--- [START] {s_type} 데이터 로드 (최근 {lookback_days}일) ---")
            
            # 1. InfluxDB에서 Pandas DataFrame으로 데이터 가져오기
            df = db.fetch_training_data(s_type, days=lookback_days)
            
            if df.empty:
                print(f"--- [ERROR] 학습할 데이터가 없습니다! ---")
                model_status["status"] = "ready"
                return

            print(f"--- [DATA LOADED] 총 {len(df)}개의 행을 불러왔습니다. ---")

            # 2. 전처리 및 PyTorch 학습 로직 (여기에 PyTorch 코드가 들어갑니다)
            # 예: values = df['value'].values.astype(np.float32)
            # TODO: PyTorch 모델 정의 -> Train -> .pt 저장
            time.sleep(5) # 학습 시뮬레이션

            # 3. TFLite 변환 및 저장
            # TODO: TFLiteConverter를 사용해 .tflite 파일로 저장
            model_path = os.path.join(MODEL_DIR, f"{s_type}_model.tflite")
            with open(model_path, "w") as f: f.write("Dummy TFLite Content") # 파일 생성 시뮬레이션

            model_status["status"] = "ready"
            model_status["last_trained"] = time.ctime()
            print(f"--- [FINISH] {s_type} 모델 학습 및 TFLite 저장 완료: {model_path} ---")

        except Exception as e:
            print(f"--- [CRITICAL ERROR] 학습 실패: {e} ---")
            model_status["status"] = "error"

    background_tasks.add_task(heavy_training_task, sensor_type, days)
    return {"message": f"{sensor_type} 학습이 시작되었습니다.", "status": "processing"}


@app.post("/predict")
async def predict_anomaly(sensor_type: str, data: list):
    """
    전달받은 데이터를 TFLite 모델로 분석합니다.
    """
    if model_status["status"] == "training":
        return {"error": "모델이 현재 학습 중입니다."}
        
    # TODO: 실제 TFLite Interpreter 로드 및 invoke 로직
    # 임시: 더미 스코어 계산
    score = float(np.std(data) / 10.0)
    return {
        "anomaly_score": round(min(score, 1.0), 4),
        "prediction": "abnormal" if score > 0.7 else "normal"
    }

@app.get("/status")
async def get_status():
    return model_status