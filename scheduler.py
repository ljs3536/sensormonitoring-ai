import os
import datetime
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler
from database_rdb import SessionLocal
from models import AiModel

def hard_delete_old_models():
    """
    is_deleted가 True이고, 삭제된 지 7일이 지난 모델을 
    물리적 파일과 함께 영구 삭제합니다.
    """
    db: Session = SessionLocal()
    print(f"--- [CLEANUP] 구형 모델 영구 삭제 작업 시작 ({datetime.datetime.now()}) ---")

    try:
        # 1. 기준 시간 설정 (현재로부터 7일 전)
        retention_days = 7
        threshold_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)

        # 2. 조건에 맞는 모델 조회 (is_deleted=True AND deleted_at < 7일 전)
        old_models = db.query(AiModel).filter(
            AiModel.is_deleted == True,
            AiModel.deleted_at <= threshold_date
        ).all()

        if not old_models:
            print("--- [CLEANUP] 삭제할 대상 모델이 없습니다. ---")
            return
        for model in old_models:
            # A. 물리적 파일 삭제 (.pt)
            if model.file_path and os.path.exists(model.file_path):
                os.remove(model.file_path)
                print(f"   > 파일 삭제 완료: {model.file_path}")
            
            # B. 매핑 파일 삭제 (_mapping.json)
            mapping_path = model.file_path.replace(".pt", "_mapping.json") if model.file_path else None
            if mapping_path and os.path.exists(mapping_path):
                os.remove(mapping_path)
                print(f"   > 매핑 파일 삭제 완료: {mapping_path}")

            # C. DB 레코드 영구 삭제
            db.delete(model)
            print(f"   > DB 레코드 영구 삭제 완료 (ID: {model.id})")

        db.commit()
        print(f"--- [CLEANUP] 총 {len(old_models)}개의 모델을 영구 삭제했습니다. ---")    
    except Exception as e:
        db.rollback()
        print(f"--- [ERROR] 청소 작업 중 오류 발생: {e} ---")
    finally:
        db.close()

# 스케줄러 설정
scheduler = BackgroundScheduler()

# 매일 새벽 3시에 실행하도록 예약
scheduler.add_job(hard_delete_old_models, 'cron', hour=3, minute=0)

# 테스트용 (서버 켜질 때마다 바로 실행되게 하려면 아래 줄 주석 해제)
# scheduler.add_job(hard_delete_old_models, 'interval', minutes=1)