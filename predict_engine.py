# sensor-ai/predict_engine.py
import numpy as np
import os

# TFLite 라이브러리 유연하게 불러오기 (에러 방지용)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        # tflite-runtime이 없으면 무거운 tensorflow에서라도 가져오기
        from tensorflow import lite as tflite
    except ImportError:
        # 둘 다 없으면 None 처리하여 서버가 죽는 것 방지
        tflite = None

def run_inference(file_path: str, input_data: list):
    """
    저장된 TFLite 모델을 로드하여 추론을 실행합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {file_path}")

    # --- [라이브러리가 없을 때의 더미 처리 로직] ---
    if tflite is None:
        print("--- [WARNING] TFLite 라이브러리가 설치되지 않아 더미 결과를 반환합니다 ---")
        # 임시 점수 계산 (표준편차 활용)
        score = float(np.std(input_data) / 10.0)
        return score

    # --- [실제 TFLite 추론 로직] ---
    interpreter = tflite.Interpreter(model_path=file_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 데이터 전처리 및 입력
    input_arr = np.array(input_data, dtype=np.float32).reshape(input_details[0]['shape'])
    
    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()
    
    # 결과 추출 (복원 오차 계산)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    mse = np.mean(np.power(input_arr - output_data, 2))
    
    return float(mse)