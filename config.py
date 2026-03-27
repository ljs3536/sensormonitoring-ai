from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # .env 파일에 있는 키 이름과 변수명을 (대소문자 무시하고) 매핑해줍니다.
    influxdb_url: str
    influxdb_token: str
    influxdb_org: str
    influxdb_bucket: str
    
    # rdb
    mariadb_url: str

    class Config:
        env_file = ".env" # 이 클래스가 실행될 때 .env 파일을 읽어오라고 지시

# 이 settings 객체 하나만 임포트하면 프로젝트 어디서든 환경변수를 꺼내 쓸 수 있습니다.
settings = Settings()