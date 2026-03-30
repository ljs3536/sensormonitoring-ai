# sensor-ai/database.py (AI 전용 DB 클라이언트)
# influxDB 연동 (AIStore 클래스)
import influxdb_client
import pandas as pd
from config import settings

class AIStore:
    def __init__(self):
        # 환경 변수 등 설정 (config.py 공유 권장)
        self.client = influxdb_client.InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org
        )
        self.query_api = self.client.query_api()

    def fetch_training_data(self, sensor_type, days=7):
        """학습을 위해 7일치 데이터를 DataFrame으로 가져옴"""
        query = f'''
            from(bucket: "{settings.influxdb_bucket}")
            |> range(start: -{days}d)
            |> filter(fn: (r) => r["_measurement"] == "{sensor_type}_sensor")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        # AI 학습에는 Pandas DataFrame이 가장 편합니다.
        return self.query_api.query_data_frame(query)