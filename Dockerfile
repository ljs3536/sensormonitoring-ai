# AI 라이브러리는 무겁기 때문에 적절한 베이스 선택이 중요합니다.
FROM python:3.9-slim

WORKDIR /app

# AI 관련 라이브러리(pandas, numpy, scikit-learn 등) 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 학습된 모델 파일(.pkl 등)과 추론 코드를 복사
COPY . .

# 포트 설정
EXPOSE 8002

CMD ["uvicorn", "main:app"]