# 1. 베이스 이미지 (원하는 Python 버전으로 맞추기)
FROM python:3.8.20

# 2. 작업 디렉토리 생성 및 설정
WORKDIR /app

# # 3. 시스템 의존성 설치 (Ubuntu 패키지 예시)
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

#✅ 4. .env 파일 복사
# #  (코드에서 load_dotenv()를 쓸 경우 필요)
# COPY .env .

# # 5. 소스코드 전체 복사
# COPY . .

# 컨테이너 시작 시 uvicorn 실행
# CMD ["uvicorn", "BeeMall_Chatbot:app", "--host", "0.0.0.0", "--port", "8011"]




# 시스템 의존성 설치 (Ubuntu 패키지 예시 + redis-server 설치)
RUN apt-get update && \
    apt-get install -y redis-server && \
    rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# .env 파일 복사 (load_dotenv() 쓸 경우 필요)
COPY .env .

# 애플리케이션 소스 전체 복사
COPY . .

#컨테이너 시작 시 Redis 백그라운드 실행 + Uvicorn 실행
#    - redis-server를 백그라운드로 띄우고 나서 uvicorn을 기동
CMD ["sh", "-c", "redis-server --daemonize yes && uvicorn BeeMall_Chatbot:app --host 0.0.0.0 --port 8011"]