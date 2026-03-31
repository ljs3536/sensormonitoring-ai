# sensor-ai/database.py
import influxdb_client
import pandas as pd
from config import settings

class AIStore:
    def __init__(self):
        self.client = influxdb_client.InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org
        )
        self.query_api = self.client.query_api()

    # ---------------------------------------------------------
    # 1. 비지도 학습용 (AutoEncoder) - 정상 데이터만 쏙쏙 뽑기
    # ---------------------------------------------------------
    def fetch_unsupervised_data(self, sensor_type, days=7):
        """
        [AutoEncoder 용도]
        라벨이 'normal'이거나, 예전에 저장해서 라벨 자체가 없는(null) 데이터만 가져옵니다.
        """
        query = f'''
            from(bucket: "{settings.influxdb_bucket}")
            |> range(start: -{days}d)
            |> filter(fn: (r) => r["_measurement"] == "{sensor_type}_sensor")
            // 🔥 핵심: 라벨이 없거나(과거 데이터), 라벨이 'normal'인 것만 필터링!
            |> filter(fn: (r) => not exists r["label"] or r["label"] == "normal")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df = self.query_api.query_data_frame(query)
        
        # Pandas DataFrame이 리스트 형태로 반환될 수 있으므로 전처리
        if isinstance(df, list):
            if len(df) == 0: return pd.DataFrame()
            df = pd.concat(df, ignore_index=True)
            
        return df

    # ---------------------------------------------------------
    # 2. 지도 학습용 (Classification) - 라벨이 있는 모든 데이터 뽑기
    # ---------------------------------------------------------
    def fetch_supervised_data(self, sensor_type, days=7):
        """
        [새로운 CNN-LSTM 분류기 용도]
        정상/비정상 상관없이 '라벨(label)'이 명확하게 찍혀있는 모든 데이터를 가져옵니다.
        """
        query = f'''
            from(bucket: "{settings.influxdb_bucket}")
            |> range(start: -{days}d)
            |> filter(fn: (r) => r["_measurement"] == "{sensor_type}_sensor")
            // 🔥 핵심: 라벨(tag)이 존재하는(exists) 데이터만 싹 다 가져옴!
            |> filter(fn: (r) => exists r["label"])
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df = self.query_api.query_data_frame(query)
        
        if isinstance(df, list):
            if len(df) == 0: return pd.DataFrame()
            df = pd.concat(df, ignore_index=True)
            
        # 지도 학습은 결측치가 있으면 안 되므로, 라벨이 비어있는 행을 혹시 몰라 한 번 더 날려줍니다.
        if not df.empty and 'label' in df.columns:
            df = df.dropna(subset=['label'])
            
        return df

    def close(self):
        self.client.close()