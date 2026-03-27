# sensor-ai/predict_engine.py
import numpy as np
import tflite_runtime.interpreter as tflite
import os

MODEL_DIR = "models"

def run_inference(sensor_type: str, model_type: str, input_data: list):
    """
    저장된 TFLite 모델을 로드하여 추론을 실행합니다.
    """
    model_path = os.path.join(MODEL_DIR, f"{sensor_type}_{model_type}.tflite")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    # TFLite 인터프리터 설정
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 데이터 전처리 및 입력 (Numpy 변환)
    input_arr = np.array(input_data, dtype=np.float32).reshape(input_details[0]['shape'])
    
    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()
    
    # 결과 추출 (AutoEncoder의 경우 복원 오차 계산)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # 예: MSE(Mean Squared Error)를 이상 점수로 활용
    mse = np.mean(np.power(input_arr - output_data, 2))
    return float(mse)