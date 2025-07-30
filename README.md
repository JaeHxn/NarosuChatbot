# NarosuChatbot

# ngork은  내 로컬 서버(로컬호스트)를 외부에서 접근할 수 있도록 해주는 터널링 서비스
# 사설 네트워크 내부에서도 외부에서 접속 가능.
# Python 은 3.8.20

**[fb-chatbot 서버 빌드] ->  docker build -t fb-chatbot .     
버전명시시 - > docker build -t fb-chatbot:v3.01 .

**[컨테이너 실행] -> docker run -d -p 8011:8011 --name fb_chatbot_container fb-chatbot:v3.01
    [.env 인식용] - > docker run -d --env-file .env -p 8011:8011 --name fb_chatbot_container fb-chatbot:v3.01

+  실시간 마지막 50줄만 보기
docker logs -f fb_chatbot_container
