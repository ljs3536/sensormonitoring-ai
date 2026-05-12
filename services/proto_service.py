# services/proto_service.py
import torch
import numpy as np
import datetime
from sqlalchemy import func
from sklearn.decomposition import PCA
from database_rdb import SessionLocal, SensorData, ModelRegistry, PredictionLog
from architectures.prototypical import PrototypicalLeakDetector 
from architectures.fewshot_prototypical import FewShotPrototypicalDetector
import os

# --- 데이터 로드 (DB -> Numpy) ---
def _get_training_data(db, sensor_id, days):
    """DB에서 최근 N일치 정상 데이터를 가져와 Numpy로 반환"""
    time_threshold = datetime.datetime.now() - datetime.timedelta(days=days)
    records = db.query(SensorData).filter(
        SensorData.MAC_ADDR == sensor_id, 
        SensorData.REG_DT >= time_threshold, 
        SensorData.LEAK_YN == "N"
    ).all()

    X = [[float(val) for val in r.SENSOR_DATA.split('|')] for r in records if r.SENSOR_DATA]
    return np.array(X)

def _do_ema_blending(detector, active_model_path, X_train, alpha=0.1):
    """기존 모델 로드 후 신규 데이터와 10% 블렌딩 연산"""
    # 1. 기존 모델 불러오기
    detector.load(active_model_path)
    old_proto_dict = detector.get_center() # 기존 중심점 {0: array}
    old_t = detector.thresholds[0]         # 기존 울타리 {'mean': f, 'std': f}

    # 2. 신규 데이터의 '임시' 기준점 및 울타리 계산 (Tensor 사용)
    X_tensor = torch.as_tensor(X_train, dtype=torch.float32).to(detector.device)
    detector.model.eval()
    with torch.no_grad():
        new_embeds = detector.model(X_tensor)
        # 현재 데이터들의 평균 위치
        new_center_tensor = torch.mean(new_embeds, dim=0) # Tensor
        
        # 현재 데이터들이 이 위치에서 얼마나 떨어져 있는지(거리) 계산
        dists = torch.norm(new_embeds - new_center_tensor, dim=1)
        new_t_mean = torch.mean(dists).item()   # 현재 데이터의 평균 거리
        new_t_std = torch.std(dists).item()     # 현재 데이터의 거리 편차 (노이즈 수준)

    # 3. 미세 조정 (EMA 연산)
    # (1) 위치 조정 (Center)
    old_center_np = old_proto_dict[0]
    new_center_np = new_center_tensor.cpu().numpy()
    
    blended_center = (1 - alpha) * old_center_np + (alpha * new_center_np)
    # (2) 범위 조정 (Threshold - Mean : 울타리의 크기)
    blended_mean = (1 - alpha) * old_t['mean'] + (alpha * new_t_mean)
    # (3) 텐션 조정 (Threshold - Std : 울타리의 유연함)
    min_std = 0.0001 # 최소한의 유연성 확보
    blended_std = max(min_std, (1 - alpha) * old_t['mean'] + (alpha * new_t_std))

    # 4. 모델에 주입
    detector.set_center({0: blended_center})
    detector.thresholds[0] = {'mean': blended_mean, 'std': blended_std}
    print(f"   ㄴ [EMA] 기존 중심점 대비 {int(alpha*100)}% 이동 완료")
    return detector

def _calculate_metrics_and_pca(detector, X_train):
    """최종 결정된 모델로 시각화 및 평가 지표 생성"""
    proto_0_np = detector.prototypes[0]
    t_mean = detector.thresholds[0]['mean']
    t_std = detector.thresholds[0]['std']
    
    X_tensor = torch.as_tensor(X_train, dtype=torch.float32).to(detector.device)
    detector.model.eval()
    with torch.no_grad():
        embeds_np = detector.model(X_tensor).cpu().numpy()

    # 히스토그램용 거리 계산
    dists = np.linalg.norm(embeds_np - proto_0_np, axis=1)
    counts, bin_edges = np.histogram(dists, bins=20)

    # PCA 2D 변환
    pca = PCA(n_components=2)
    if len(embeds_np) >= 2:
        points_2d = pca.fit_transform(embeds_np)
        center_2d = pca.transform(proto_0_np.reshape(1, -1))[0]
    else:
        points_2d = [[0.0, 0.0]] * len(embeds_np)
        center_2d = [0.0, 0.0]

    return {
        "train_samples": len(X_train),
        "threshold_mean": float(t_mean),
        "threshold_std": float(t_std),
        "anomaly_limit_3sigma": float(t_mean + (3.0 * t_std)),
        "tightness_score": float(1.0 / (t_std + 1e-6)),
        "train_dist_hist": {"counts": counts.tolist(), "bins": bin_edges.tolist()},
        "pca_2d_points": points_2d.tolist(),
        "pca_2d_center": center_2d.tolist()
    }

# --- 백테스팅 (모의고사 실행) ---
def _run_backtest(db, detector, sensor_id, model_id):
    """최근 3일 데이터로 모의고사"""
    test_threshold = datetime.datetime.now() - datetime.timedelta(days=3)
    records = db.query(SensorData).filter(SensorData.MAC_ADDR == sensor_id, SensorData.REG_DT >= test_threshold).all()
    X_test = np.array([[float(val) for val in r.SENSOR_DATA.split('|')] for r in records if r.SENSOR_DATA])
    
    if len(X_test) > 0:
        results = detector.predict(X_test)
        logs = [PredictionLog(MODEL_ID=model_id, MAC_ADDR=sensor_id, PROBABILITY=r['prob']*100, RESULT=r['is_leak']) for r in results]
        db.bulk_save_objects(logs)
        db.commit()
        print(f"백테스팅 완료 ({len(logs)}건)")

# --- 전체 프로세스 조율 ---
def train_proto_model_internal(sensor_id: str, model_type: str, update_mode: str, days: int, auto_activate: bool, memo: str = None):
    db = SessionLocal() 
    try:
        # [Step 1] 데이터 준비
        X_train = _get_training_data(db, sensor_id, days)
        if len(X_train) == 0: return print(f"{sensor_id}: 학습 데이터가 부족합니다.")

        # [Step 2] 아키텍처 준비
        detector = PrototypicalLeakDetector() if model_type == "all" else FewShotPrototypicalDetector()

        # [Step 3] 학습/업데이트 전략 실행
        if update_mode == "ema":
            active_model = db.query(ModelRegistry).filter(
                ModelRegistry.MAC_ADDR == sensor_id, ModelRegistry.MODEL_TYPE == model_type, ModelRegistry.STATUS == "ACTIVE"
            ).first()
            
            if active_model and os.path.exists(active_model.FILE_PATH):
                print(f"[Mode: EMA] v{active_model.VERSION} 모델 기반으로 미세 조정합니다.")
                detector = _do_ema_blending(detector, active_model.FILE_PATH, X_train, alpha=0.1)
            else:
                print("[Mode: EMA] 활성화된 모델이 없어 '신규 학습'으로 자동 전환합니다.")
                detector.fit(X_train, np.zeros(len(X_train)), epochs=50)
        else:
            print(f"[Mode: REPLACE] 데이터를 사용하여 모델을 새로 학습합니다.")
            detector.fit(X_train, np.zeros(len(X_train)), epochs=50)

        # [Step 4] 모델 파일 저장
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sensor_id}_{model_type}_{ts}"
        saved_path = detector.save(filename)

        # [Step 5] 평가 지표 계산
        metrics = _calculate_metrics_and_pca(detector, X_train)

        # [Step 6] DB 등록
        max_v = db.query(func.max(ModelRegistry.VERSION)).filter(ModelRegistry.MAC_ADDR == sensor_id).scalar() or 0
        new_status = "ACTIVE" if auto_activate else "CANDIDATE"
        if new_status == "ACTIVE":
            db.query(ModelRegistry).filter(ModelRegistry.MAC_ADDR == sensor_id, ModelRegistry.STATUS == "ACTIVE").update({"STATUS": "INACTIVE"})

        new_model = ModelRegistry(
            MAC_ADDR=sensor_id, MODEL_TYPE=model_type, VERSION=max_v + 1,
            FILE_PATH=saved_path, STATUS=new_status, EVAL_METRICS=metrics,
            THRESHOLD_MEAN=metrics['threshold_mean'], THRESHOLD_STD=metrics['threshold_std'],
            TRAIN_SAMPLES=metrics['train_samples'],
            MEMO=memo
        )
        db.add(new_model)
        db.commit()

        # 백테스팅
        _run_backtest(db, detector, sensor_id, new_model.MODEL_ID)
        print(f"[AI] {sensor_id} 모델 v{new_model.VERSION} 생성 및 등록 완료!")

    except Exception as e:
        print(f"[AI] 학습 중 치명적 에러: {e}")
        import traceback
        traceback.print_exc() # 에러 위치를 정확히 찍어줍니다.
    finally:
        db.close()