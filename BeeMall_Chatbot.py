import asyncio
import base64
import json
import logging
import math
import os
import re
import time
import urllib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union
from urllib.parse import quote

import faiss
import numpy as np
import pandas as pd
import redis
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
    RedisChatMessageHistory,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

executor = ThreadPoolExecutor()

# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
REDIS_URL = "redis://localhost:6379/0"
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
MANYCHAT_API_KEY = os.getenv('MANYCHAT_API_KEY')

print(f"🔍 로드된 VERIFY_TOKEN: {VERIFY_TOKEN}")
print(f"🔍 로드된 PAGE_ACCESS_TOKEN: {PAGE_ACCESS_TOKEN}")
print(f"🔍 로드된 API_KEY: {API_KEY}")
print(f"🔍 로드된 MANYCHAT_API_KEY: {MANYCHAT_API_KEY}")

# ✅ FAISS 인덱스 파일 경로 설정
faiss_file_path = f"04_28_faiss_3s.faiss"

EMBEDDING_MODEL = "text-embedding-3-small"

def get_redis():
    return redis.Redis.from_url(REDIS_URL)

# ✅ FastAPI 인스턴스 생성
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5050",
                   "https://satyr-inviting-quetzal.ngrok-free.app", 
                   "https://viable-shark-faithful.ngrok-free.app"],  # 외부 도메인 추가
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("response_time_logger")
print(f"🔐 API KEY: {MANYCHAT_API_KEY}")


# 응답 속도 측정을 위한 미들웨어 추가
@app.middleware("http")
async def measure_response_time(request: Request, call_next):
    start_time = time.time()  # 요청 시작 시간
    response = await call_next(request)  # 요청 처리
    process_time = time.time() - start_time  # 처리 시간 계산

    response.headers["ngrok-skip-browser-warning"] = "1"
    response.headers["X-Frame-Options"] = "ALLOWALL"  # 또는 제거 방식도 가능 #BeeMall 챗봇 Iframe 막히는것 때문에 헤더 추가가
    response.headers["Content-Security-Policy"] = "frame-ancestors *" #BeeMall 챗봇 Iframe 막히는것 때문에 헤더 추가가

    # '/chatbot' 엔드포인트에 대한 응답 속도 로깅
    if request.url.path == "/webhook":
        print(f"📊 [TEST] Endpoint: {request.url.path}, 처리 시간: {process_time:.4f} 초")  # print로 직접 확인
        logger.info(f"📊 [Endpoint: {request.url.path}] 처리 시간: {process_time:.4f} 초")
    
    response.headers["X-Process-Time"] = str(process_time)  # 응답 헤더에 처리 시간 추가
    return response

# ✅ Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

'''# ✅ Redis 기반 메시지 기록 관리 함수
def get_message_history(session_id: str) -> RedisChatMessageHistory:
    """
    Redis를 사용하여 메시지 기록을 관리합니다.
    :param session_id: 사용자의 고유 세션 ID
    :return: RedisChatMessageHistory 객체
    """
    try:
        history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
        return history
    except Exception as e:
        print(f"❌ Redis 연결 오류: {e}")
        raise HTTPException(status_code=500, detail="Redis 연결에 문제가 발생했습니다.")'''

# 요청 모델
class QueryRequest(BaseModel):
    query: str


# ✅ JSON 직렬화를 위한 int 변환 함수
def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
        return obj.item()
    return obj

# ✅ 엑셀 데이터 로드 및 변환 (본문상세설명 컬럼 제외하고 임베딩용 텍스트 생성)
def load_excel_to_texts(file_path):
    try:
        data = pd.read_excel(file_path)
        data.columns = data.columns.str.strip()

        # 임베딩용 데이터프레임에서 '본문상세설명' 제외
        if '본문상세설명' in data.columns:
            embedding_df = data.drop(columns=['본문상세설명'])
        else:
            embedding_df = data

        texts = [" | ".join([f"{col}: {row[col]}" for col in embedding_df.columns]) for _, row in embedding_df.iterrows()]
        return texts, data  # 원본 데이터(data)는 본문상세설명 포함
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"엑셀 파일 로드 오류: {str(e)}")

# ✅ FAISS 인덱스 저장
def save_faiss_index(index, file_path):
    try:
        faiss.write_index(index, file_path)
    except Exception as e:
        print(f"❌ FAISS 인덱스 저장 오류: {e}")

# ✅ FAISS 인덱스 로드
def load_faiss_index(file_path):
    try:
        return faiss.read_index(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS 인덱스 로딩 오류: {str(e)}")

# ✅ 문서 임베딩 함수 (병렬 처리)
def embed_texts_parallel(texts, embedding_model=EMBEDDING_MODEL, max_workers=8):
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            임베딩 = OpenAIEmbeddings(model=embedding_model, openai_api_key=API_KEY)
            embeddings = list(executor.map(임베딩.embed_query, texts))
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 생성 오류: {str(e)}")

# ✅ FAISS 인덱스 생성 및 저장 함수 (병렬 처리 적용)
def create_and_save_faiss_index(file_path):
    try:
        start_time = time.time()
        
        # 엑셀 파일 로드 및 변환
        texts, _ = load_excel_to_texts(file_path)
        print(f"📊 엑셀 파일 로드 및 변환 완료! ({len(texts)}개 텍스트)")

        # 임베딩 생성 (병렬 처리 적용)
        embeddings = embed_texts_parallel(texts, EMBEDDING_MODEL)
        print(f"📊 임베딩 생성 완료!")
        
        # ✅ 예시 텍스트 1줄 출력해서 본문상세설명 포함 여부 확인
        print("🔎 임베딩 대상 텍스트 예시 1줄:")
        print(texts[0])  # 본문상세설명 포함 여부 확인용
        
        # 임베딩 벡터의 개수와 각 벡터의 차원 출력
        print(f"🔍🔍 임베딩 벡터 개수: {len(embeddings)}, 임베딩 차원: {embeddings.shape[1]}")
        print(f"🔍🔍 임베딩 벡터 개수: {embeddings.shape[0]}")

        # FAISS 인덱스 설정
        faiss.normalize_L2(embeddings)
        d = embeddings.shape[1]
        nlist = min(200, len(texts) // 100)  # 클러스터 개수 설정 (데이터 개수에 비례)
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        # 인덱스 학습 및 추가
        index.train(embeddings)
        index.add(embeddings)

        # 인덱스 저장
        save_faiss_index(index, faiss_file_path)

        end_time = time.time()
        print(f"✅ FAISS 인덱스 생성 및 저장 완료! (걸린 시간: {end_time - start_time:.2f} 초)")
    
    except Exception as e:
        print(f"❌ FAISS 인덱스 생성 및 저장 오류: {e}")
    

# ✅ 인덱스 로드 또는 생성하기
def initialize_faiss_index():
    if not os.path.exists(faiss_file_path):
        # 현재 디렉토리의 'db' 폴더 안에서 엑셀 파일을 검색
        file_path = os.path.join(os.getcwd(), "db", "ownerclan_주간인기상품_0428.xlsx")
        
        # 🔍 엑셀 데이터 로드 확인
        texts, data = load_excel_to_texts(file_path)
        print(data.head())  # 데이터의 첫 5개 행 출력 (엑셀 데이터 확인용)
        print(texts[0])  # 텍스트의 첫 번째 항목 출력 
        
        create_and_save_faiss_index(file_path)
    index = load_faiss_index(faiss_file_path)
    return index

# ✅ 인덱스 초기화 실행
index = initialize_faiss_index()

# ✅ LLM을 이용한 키워드 추출 및 대화 이력 반영
def extract_keywords_with_llm(query):
    try:
        
        print(f"🔍 [extract_keywords_with_llm] 입력값: {query}")

        # ✅ Step 1: API 키 확인
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ [ERROR] {API_KEY} 환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")
        API_KEY = os.environ["OPENAI_API_KEY"]
        
        if not API_KEY or not isinstance(API_KEY, str):
            raise ValueError("❌ [ERROR] OpenAI API_KEY가 None이거나 잘못되었습니다!")

        print(f"🔍 [DEBUG] OpenAI API Key 확인 완료")

        # ✅ [Step 1] query 값 검증
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"❌ [ERROR] query 값이 잘못되었습니다: {query} (타입: {type(query)})")

        redis_start = time.time()

        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)

        print(f"🔍 [Step 2] LLM API 호출 시작...")
        print("💬 llm.invoke 직전")
        # 기존 대화 이력과 함께 LLM에 전달
        response = llm.invoke([
            SystemMessage(content="""
                당신은 상품 추천 챗봇의 핵심 키워드 추출기 역할을 합니다.
                사용자의 대화 내역을 반영하여 **상품 검색에 적합한 핵심 키워드 목록**을 추출해주세요.
                만약 단어 간에 띄어쓰기가 있다면 하나의 단어일 수도 있습니다. 띄어쓰기가 있다면 단어끼리 붙여서도 문장을 분석해보세요. 여러 방법, 여러 방면으로 생각해서 추출해주세요.

                ---

                🎯 [목표]
                - 상품 추천에 필요한 핵심 단어만 간결하게 추출합니다.
                - 중복되거나 의미가 겹치는 단어는 제거하고 정리합니다.

                🛑 [주의사항 - 부정 표현 필터링]
                - 사용자가 "싫다", "아니다", "말고", "제외", "안돼", "하지마" 등의 표현을 사용한 경우,
                해당 단어 또는 관련된 상품 종류는 키워드에서 **제외**해주세요.

                예)
                - 입력: "스카프 말고 셔츠 보여줘" → 키워드: 셔츠
                - 입력: "스트라이프 아니어도 돼" → 키워드: 셔츠
                - 입력: "여성용인데 캐주얼 말고 포멀로" → 키워드: 여성, 포멀
                - 입력: "코튼은 빼고 린넨 원단 원해" → 키워드: 린넨
                - 입력: "검정색은 싫고 흰색 계열 보여줘" → 키워드: 흰색
                - 입력: "니트 말고 반팔티 없어요?" → 키워드: 반팔티
                - 입력: "긴팔보다는 반팔로 보여주세요" → 키워드: 반팔
                - 입력: "지난번에 보여준 건 아니고 다른 셔츠 보여줘" → 키워드: 셔츠
                - 입력: "스커트 말고 바지 쪽으로 추천해줘" → 키워드: 바지
                - 입력: "체크무늬는 제외하고 추천해줘" → 키워드: 추천
                - 입력: "슬림핏은 안되고 루즈핏으로" → 키워드: 루즈핏
                - 입력: "스트라이프는 괜찮지만 도트는 안돼요" → 키워드: 스트라이프
                - 입력: "겨울옷 말고 봄에 입을 옷 찾아줘" → 키워드: 봄, 옷

                🌐 [언어 변환]
                - 만약 외국어로 입력되었다면, 먼저 자연스럽게 한국어로 번역한 뒤 핵심 키워드를 추출하세요.

                ---

                📦 [형식]
                - 쉼표로 구분된 핵심 키워드 목록만 출력하세요.
                - 예시 출력: 여자, 셔츠, 여름, 린넨
            """),
            HumanMessage(content=f"{query}")
        ])
        print("✅ llm.invoke 호출 성공")
        

        print(f"✅ [Step 4] LLM 응답 확인: {response}")

        # ✅ [Step 5] 응답 값 검증
        if response is None:
            raise ValueError("❌ [ERROR] LLM 응답이 None입니다.")

        if not hasattr(response, "content"):
            raise AttributeError(f"❌ [ERROR] 응답 객체에 `content` 속성이 없습니다: {response}")

        if not isinstance(response.content, str) or not response.content.strip():
            raise ValueError(f"❌ [ERROR] LLM 응답이 비어 있거나 잘못된 데이터입니다: {response.content}")

        # 키워드 업데이트
        # ✅ 응답에서 '핵심 키워드: ' 부분 제거하여 임베딩에 사용하도록 함
        keywords_text = response.content.replace("추출된 핵심 키워드:" , "").strip()
        
        # ✅ 벡터 검색용으로는 핵심 키워드 부분을 제거한 텍스트 사용
        keywords_for_embedding = [keyword.strip() for keyword in keywords_text.split(",")]
        combined_keywords = ", ".join(keywords_for_embedding)
        
        # ✅ AI 응답에서는 원본 텍스트(response.content)도 함께 사용할 수 있게 저장
        keywords = {
            "original_text": response.content,  # AI 응답용 원본 텍스트
            "processed_keywords": combined_keywords  # 벡터 검색용 키워드 텍스트
        }
        
        redis_time = time.time() - redis_start
        logger.info(f"📊 LLM을 이용한 키워드 추출 시간: {redis_time:.4f} 초")
        
        if not combined_keywords:
            raise ValueError("❌ [ERROR] 키워드 추출 결과가 비어 있음.")

        print(f"✅ [Step 7] 추출된 키워드: {combined_keywords}")

        return combined_keywords
    except Exception as e:
        print(f"❌ [ERROR] extract_keywords_with_llm 실행 중 오류 발생: {e}")
        raise

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    
    return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)

def clear_message_history(session_id: str):
    """
    Redis에 저장된 특정 세션의 대화 기록을 초기화합니다.
    """
    try:
        history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
        history.clear()
        print(f"✅ 세션 {session_id}의 대화 기록이 초기화되었습니다.")
    except Exception as e:
        print(f"❌ Redis 초기화 오류: {e}")
        raise HTTPException(status_code=500, detail="대화 기록 초기화 중 오류가 발생했습니다.")


# 🔥 상품 캐시 (전역 선언)
PRODUCT_CACHE = {}
# 🔗 구매하기 버튼 클릭 시 호출되는 ManyChat용 Hook 주소
MANYCHAT_HOOK_BASE_URL = "https://viable-shark-faithful.ngrok-free.app/product-select"


@app.get("/webhook")
async def verify_webhook(request: Request):
    try:
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        
        print(f"🔍 받은 Verify Token: {token}")
        print(f"🔍 서버 Verify Token: {VERIFY_TOKEN}")
        
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("✅ 웹훅 인증 성공")
            return int(challenge)
        else:
            print("❌ 웹훅 인증 실패")
            return {"status": "error", "message": "Invalid token"}
    except Exception as e:
        print(f"❌ 인증 처리 오류: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    start_time = time.time()

    try:
        # ✅ Step 1: 요청 데이터 파싱
        data = await request.json()
        parse_time = time.time() - start_time
        logger.info(f"📊 [Parse Time]: {parse_time:.4f} 초")

        # ✅ Step 2: 메시지 처리 시작
        process_start = time.time()

        if data.get("field") == "messages":
            value = data.get("value", {})

            sender_id = value.get("sender", {}).get("id")
            user_message = value.get("message", {}).get("text", "").strip()
            postback = value.get("postback", {})

            # ✅ postback 처리
            postback_payload = postback.get("payload")
            if postback_payload and postback_payload.startswith("BUY::"):
                product_code = postback_payload.split("::")[1]
                background_tasks.add_task(handle_product_selection, sender_id, product_code)
                return {
                    "version": "v2",
                    "content": {
                        "messages": [
                            {"type": "text", "text": f"✅ 상품 {product_code} 정보가 전송되었습니다!"}
                        ]
                    }
                }

            # ✅ reset 처리
            if sender_id and user_message:
                if user_message.lower() == "reset":
                    print(f"🔄 [RESET] 세션 {sender_id}의 대화 기록 초기화!")
                    clear_message_history(sender_id)
                    return {
                        "version": "v2",
                        "content": {
                            "messages": [
                                {
                                    "type": "text",
                                    "text": f"🔄 Chat reset complete!\n💬 Enter a keyword and let the AI work its magic 🛍️."
                                }
                            ]
                        },
                        "message": f"세션 {sender_id}의 대화 기록이 초기화되었습니다."
                    }

                # ✅ 일반 메시지 → AI 응답 처리
                background_tasks.add_task(process_ai_response, sender_id, user_message)

            process_time = time.time() - process_start
            logger.info(f"📊 [Processing Time 전체]: {process_time:.4f} 초")

        # 기본 응답
        return {
            "version": "v2",
            "content": {
                "messages": [
                    {
                        "type": "text",
                        "text": "🛍️ Just a moment, smart picks coming soon! ⏳"
                    }
                ]
            }
        }

    except Exception as e:
        print(f"❌ 웹훅 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# 🔁 추천 응답 처리 함수
async def process_ai_response(sender_id: str, user_message: str):
    try:
        print(f"🕒 [AI 처리 시작] 유저 ID: {sender_id}, 메시지: {user_message}")

        # ✅ 외부 응답 생성 (동기 → 비동기 실행)
        loop = asyncio.get_running_loop()
        bot_response = await loop.run_in_executor(executor, external_search_and_generate_response, user_message, sender_id)

        # ✅ 응답 확인 및 메시지 준비
        if isinstance(bot_response, dict):
            combined_message_text = bot_response.get("combined_message_text", "")
            results = bot_response.get("results", [])

            # ✅ 상품 캐시에 저장 (product_code → 상품 딕셔너리 전체 저장)
            for product in results:
                product_code = product.get("상품코드")
                if product_code:
                    PRODUCT_CACHE[product_code] = product

            messages_data = []

            # ✅ AI 응답 메시지 먼저 추가
            if combined_message_text:
                messages_data.append({
                    "type": "text",
                    "text": combined_message_text
                })

            # ✅ 카드형 메시지를 하나로 묶기 위한 elements 리스트
            cards_elements = []

            for product in results:
                product_code = product.get("상품코드", "None")

                # 가격과 배송비 정수 변환 후 포맷팅
                try:
                    price = int(float(product.get("가격", 0)))
                except:
                    price = 0
                try:
                    shipping = int(float(product.get("배송비", 0)))
                except:
                    shipping = 0

                cards_elements.append({
                    "title": f"✨ {product['제목']}",
                    "subtitle": (
                        f"가격: {price:,}원\n"
                        f"배송비: {shipping:,}원\n"
                        f"원산지: {product.get('원산지', '')}"
                    ),
                    "image_url": product.get("이미지", ""),
                    "buttons": [
                        {
                            "type": "url",
                            "caption": "🤩 View Product 🧾",
                            "url": product.get("상품링크", "#")
                        },
                        {
                            "type": "dynamic_block_callback",
                            "caption": "🛍️ Buy Now 💰",
                            "url": "https://viable-shark-faithful.ngrok-free.app/product-select",
                            "method": "post",
                            "payload": {
                                "product_code": product_code,
                                "sender_id": sender_id
                            }
                        }
                    ]
                })

            # ✅ 전체 카드 메시지로 추가
            messages_data.append({
                "type": "cards",
                "image_aspect_ratio": "horizontal",  # 또는 "square"
                "elements": cards_elements
})

            # ✅ 메시지 전송
            send_message(sender_id, messages_data)
            print(f"✅ [Combined 메시지 전송 완료]: {combined_message_text}")
            print(f"버튼 생성용 product_code: {product_code}")
            print("✅ 최종 messages_data:", json.dumps(messages_data, indent=2, ensure_ascii=False))

        else:
            print(f"❌ AI 응답 오류 발생")

    except Exception as e:
        print(f"❌ AI 응답 처리 오류: {e}")

def clean_html_content(html_raw: str) -> str:
    try:
        html_cleaned = html_raw.replace('\n', '').replace('\r', '')
        html_cleaned = html_cleaned.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("'", "'")
        if html_cleaned.count("<center>") > html_cleaned.count("</center>"):
            html_cleaned += "</center>"
        if html_cleaned.count("<p") > html_cleaned.count("</p>"):
            html_cleaned += "</p>"
        return html_cleaned
    except Exception as e:
        print(f"❌ HTML 정제 오류: {e}")
        return html_raw


'''####################################################################################################################
external_search_and_generate_response는 ManyChat 같은 외부 서비스와 연동되는 챗봇용 API이고, 구축된 UI 에는 사용되지 않음.
'''

# ✅ 외부 검색 및 응답 생성 함수
def external_search_and_generate_response(request: Union[QueryRequest, str], session_id: str = None) -> dict:  

    # ✅ [Step 1] 요청 데이터 확인
    query = request
    print(f"🔍 사용자 검색어: {query}")

    if not isinstance(query, str):
        raise TypeError(f"❌ [ERROR] 잘못된 query 타입: {type(query)}")
    

    # ✅ [Step 2] Reset 요청 처리
    if query.lower() == "reset":
        if session_id:
            clear_message_history(session_id)
        return {"message": f"세션 {session_id}의 대화 기록이 초기화되었습니다."}

    try:
        # ✅ Step 3: Redis 기록 불러오기
        redis_start = time.time()
        session_history = get_session_history(session_id)
        redis_time = time.time() - redis_start
        print(f"📊 [Step 3] Redis 메시지 기록 관리 시간: {redis_time:.4f} 초")

        # ✅ [Step 4~5] 최신 메시지 기록 다시 불러오기
        previous_queries = [msg.content for msg in session_history.messages if isinstance(msg, HumanMessage)]
        # ✅ 현재 입력값이 이전 대화에 이미 있다면 제거 (중복 방지)
        if query in previous_queries:
            previous_queries.remove(query)
        print(f"🔁 [Step 5] 최신 Redis 대화 내역: {previous_queries}")
        
        print("🔍 [DEBUG] Redis 메시지 저장 순서 확인:")
        for i, msg in enumerate(session_history.messages):
            print(f"{i+1}번째 ▶️ {type(msg).__name__} | 내용: {msg.content}")

        # ✅ [Step 6] LLM 키워드 추출
        llm_start = time.time()
        combined_query = " ".join(previous_queries + [query])
        print(f"🔍 [Step 6-1] combined_query: {combined_query}")

        if not combined_query or not isinstance(combined_query, str):
            raise ValueError(f"❌ [ERROR] combined_query가 올바른 문자열이 아닙니다: {combined_query} (타입: {type(combined_query)})")

        combined_keywords = extract_keywords_with_llm(combined_query)
        llm_time = time.time() - llm_start

        if not combined_keywords or not isinstance(combined_keywords, str):
            raise ValueError(f"❌ [ERROR] 키워드 추출 실패: {combined_keywords}")

        print(f"🔍 [Step 6-2] combined_keywords: {combined_keywords}")
        print(f"📊 [Step 6-3] LLM 키워드 추출 시간: {llm_time:.4f} 초")

        # ✅ [Step 7] 엑셀 데이터 로드
        excel_start = time.time()
        try:
            _, data = load_excel_to_texts("db/ownerclan_주간인기상품_0428.xlsx")
        except Exception as e:
            raise ValueError(f"❌ [ERROR] 엑셀 데이터 로딩 실패: {e}")

        excel_time = time.time() - excel_start
        print(f"📊 [Step 7] 엑셀 데이터 로드 시간: {excel_time:.4f} 초")

        # ✅ [Step 8] OpenAI 임베딩 생성
        embedding_start = time.time()
        try:
            query_embedding = embed_texts_parallel([combined_keywords], EMBEDDING_MODEL)
            faiss.normalize_L2(query_embedding)
        except Exception as e:
            raise ValueError(f"❌ [ERROR] 임베딩 생성 실패: {e}")

        embedding_time = time.time() - embedding_start
        print(f"📊 [Step 8] OpenAI 임베딩 생성 시간: {embedding_time:.4f} 초")

        # ✅ [Step 9] FAISS 검색 수행
        faiss_start = time.time()
        try:
            D, I = index.search(query_embedding, k=5)
        except Exception as e:
            raise ValueError(f"❌ [ERROR] FAISS 검색 실패: {e}")

        faiss_time = time.time() - faiss_start
        print(f"📊 [Step 9] FAISS 검색 시간: {faiss_time:.4f} 초")


        # ✅ [Step 10] 검색 결과 유효성 검사
        if I is None or not I.any():
            print("❌ [ERROR] FAISS 검색 결과 없음")
            return {
                "query": query,
                "results": [],
                "message": "검색 결과가 없습니다. 다른 키워드를 입력하세요!",
                "message_history": [
                    {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
                    for msg in session_history.messages
                ],
            }

        # ✅ [Step 11] 검색 결과 JSON 변환
        results = []
        for idx_list in I:
            for idx in idx_list:
                if idx >= len(data):
                    print(f"❌ [ERROR] 잘못된 인덱스: {idx}")
                    continue

                try:
                    result_row = data.iloc[idx]

                    # ✅ 상품상세설명 -> base64 인코딩 (디코딩 에러 방지)
                    html_raw = result_row.get("본문상세설명", "") or ""
                    html_cleaned = clean_html_content(html_raw)

                    try:
                        if isinstance(html_raw, bytes):
                            html_raw = html_raw.decode("cp949")  # 혹시 바이너리 형태일 경우 디코딩
                    except Exception as e:
                        print(f"⚠️ [본문 디코딩 경고] cp949 디코딩 실패: {e}")

                    try:
                        encoded_html = base64.b64encode(html_cleaned.encode("utf-8", errors="ignore")).decode("utf-8")
                        safe_html = urllib.parse.quote_plus(encoded_html)
                        preview_url = f"https://viable-shark-faithful.ngrok-free.app/preview?html={safe_html}"
                    except Exception as e:
                        print(f"❌ [본문 인코딩 실패] {e}")
                        preview_url = "https://naver.com"

                    # ✅ 상품링크가 비어있다면 preview_url 사용
                    product_link = result_row.get("상품링크", "")
                    if not product_link or product_link in ["링크 없음", "#", None]:
                        product_link = preview_url

                    # ✅ 옵션 처리: 조합형옵션 → '옵션명 (+가격)' 형식, 재고는 표시 안함
                    option_raw = str(result_row.get("조합형옵션", "")).strip()
                    option_display = "없음"
                    if option_raw and option_raw.lower() != "nan":
                        option_lines = option_raw.splitlines()
                        parsed_options = []
                        for line in option_lines:
                            try:
                                name, extra_price, _ = line.split(",")
                                extra_price = int(float(extra_price))
                                price_str = f"(+{extra_price:,}원)" if extra_price > 0 else ""
                                parsed_options.append(f"{name} {price_str}".strip())
                            except Exception as e:
                                print(f"⚠️ 옵션 파싱 실패: {line} → {e}")
                                parsed_options.append(name)
                        option_display = "\n".join(parsed_options)

                    result_info = {
                        "상품코드": str(result_row.get("상품코드", "없음")),
                        "제목": result_row.get("마켓상품명", "제목 없음"),
                        "가격": convert_to_serializable(result_row.get("마켓실제판매가", 0)),
                        "배송비": convert_to_serializable(result_row.get("배송비", 0)),
                        "이미지": result_row.get("이미지중", "이미지 없음"),
                        "원산지": result_row.get("원산지", "정보 없음"),
                        "상품링크": product_link,
                        "옵션": option_display,
                        "조합형옵션": option_raw,
                        "최대구매수량": convert_to_serializable(result_row.get("최대구매수량", 0))
                    }
                    results.append(result_info)
                    
                    # ✅ 상품 코드 기준으로 캐시에 저장
                    PRODUCT_CACHE[result_info["상품코드"]] = result_info

                except KeyError as e:
                    print(f"❌ [ERROR] KeyError: {e}")
                continue


        if not results:
            return {"query": query, "results": [], "message": "검색 결과가 없습니다."}

        # ✅ results를 텍스트로 변환
        if results:
            results_text = "<br>".join(
                [
                    f"상품코드: {item['상품코드']}, 제목: {item['제목']}, 가격: {item['가격']}원, "
                    f"배송비: {item['배송비']}원, 원산지: {item['원산지']}, 이미지: {item['이미지']}"
                    for item in results
                ]
            )
        else:
            results_text = "검색 결과가 없습니다."
        
        
        # ✅ [Step 12] LLM 기반 대화 응답 생성
        message_history=[]
        start_response = time.time()    
        # ✅ ChatPromptTemplate 및 RunnableWithMessageHistory 생성
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
        당신은 쇼핑몰 챗봇으로, 친절하고 인간적인 대화를 통해 고객의 쇼핑 경험을 돕습니다.
        사용자의 언어에 맞게 번역해서 답변하세요(예시: 한국어->한국어, 영어->영어, 베트남어->베트남어 등)

        🎯 목표:
        - 사용자의 요구를 이해하고 대화의 맥락을 반영하여 적합한 상품을 추천합니다.

        ⚙️ 작동 방식:
        - 대화 이력을 참고해 문맥을 파악하고 사용자의 요청에 맞는 상품을 연결합니다.
        - 필요한 경우 후속 질문으로 사용자의 요구를 구체화합니다.

        📌 주의사항:
        - 아래 검색 결과는 LLM 내부 참고용입니다.
        - 상품을 나열하거나 직접 출력하지 마세요.
        - 키워드 요약이나 후속 질문을 위한 참고용으로만 활용하세요.
        """),

            MessagesPlaceholder(variable_name="message_history"),

            ("system", f"[검색 결과 - 내부 참고용 JSON]\n{json.dumps(results[:5], ensure_ascii=False).replace('{', '{{').replace('}', '}}')}"),


            ("system", f"[이전 대화 내용]\n{message_history}"),

            ("human", query)
        ])
        
        runnable = prompt | llm
        with_message_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",  # 입력 메시지의 키
            history_messages_key="message_history",
        )

        # ✅ LLM 실행 및 메시지 기록 업데이트
        response = with_message_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )

        response_time = time.time() - start_response
        print(f"📊 [Step 12] LLM 응답 생성 시간: {response_time:.4f} 초")

        # ✅ 메시지 기록을 Redis에서 가져오기
        session_history = get_session_history(session_id)
        message_history = [
            {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
            for msg in session_history.messages
        ]


        # ✅ 출력 디버깅
        #print("*** Response:", response)
        #print("*** Message History:", message_history)
        #print("✅✅✅✅*✅✅✅✅ Results:", results)
        #print(f"✅ [Before Send] Results Type: {type(results[:5])}")
        #print(f"✅ [Before Send] Results Content: {results[:5]}")

        # ✅ Combined Message 만들기 (검색 결과 + LLM 응답)
        combined_message_text = f"🤖 AI 답변: {response.content}"
        print(f"🔍 [Step 12-1] Combined Message: {combined_message_text}")
        
        # ✅ JSON 반환
        return {
            "query": query,
            "results": results,
            "combined_message_text": combined_message_text,
            "message_history": message_history
        }
        
    
        # 전체 처리 시간 로깅
        total_time = time.time() - start_time
        logger.info(f"📊 [Total Time] 전체 external_search_and_generate_response 처리 시간: {total_time:.4f} 초")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def send_message(sender_id: str, messages: list):  
    try:  
        url = "https://api.manychat.com/fb/sending/sendContent"
        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json"
        }

        # ✅ 메시지 구조 확인
        if not isinstance(messages, list):
            print(f"❌ [ERROR] messages는 리스트여야 합니다. 전달된 타입: {type(messages)}")
            return

        # ✅ LLM 응답 (첫 번째 메시지) 전송
        if messages:
            llm_text = messages[0]
            data = {
                "subscriber_id": sender_id,
                "data": {
                    "version": "v2",
                    "content": {
                        "messages": [llm_text],
                        "actions": [],
                        "quick_replies": []
                    }
                },
                "message_tag": "ACCOUNT_UPDATE"
            }
            response = requests.post(url, headers=headers, json=data)
            print(f"✅ [LLM 메시지 전송]: {response.json()}")

        # ✅ 카드 묶음 메시지 전송
        if len(messages) > 1:
            card_block = messages[1]
            data = {
                "subscriber_id": sender_id,
                "data": {
                    "version": "v2",
                    "content": {
                        "messages": [card_block],
                        "actions": [],
                        "quick_replies": []
                    }
                },
                "message_tag": "ACCOUNT_UPDATE"
            }

            response = requests.post(url, headers=headers, json=data)
            print(f"✅ [카드 메시지 전송]: {response.json()}")

    except Exception as e:
        print(f"❌ ManyChat 메시지 전송 오류: {e}")

class ManychatFieldUpdater:
    BASE_URL = "https://api.manychat.com/fb/subscriber/setCustomField"
    
    def __init__(self, subscriber_id: str, api_key: str):
        self.subscriber_id = subscriber_id
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def set_field(self, field_id: str, value):
        data = {
            "subscriber_id": self.subscriber_id,
            "field_id": field_id,
            "field_value": value
        }
        response = requests.post(self.BASE_URL, headers=self.headers, json=data)
        if response.status_code == 200:
            print(f"✅ {field_id} 저장 성공: {value}")
        else:
            print(f"❌ {field_id} 저장 실패: {response.status_code}, {response.text}")

    def set_unique_code(self, field_id: str, code: str):
        self.set_field(field_id, code)

    def set_product_name(self, field_id: str, name: str):
        self.set_field(field_id, name)

    def set_option(self, field_id: str, option: str):
        self.set_field(field_id, option)

    def set_price(self, field_id: str, price: int):
        self.set_field(field_id, price)

    def set_shipping(self, field_id: str, shipping: int):
        self.set_field(field_id, shipping)
    
    def set_product_selection_option(self, field_id: str, option: str):
        self.set_field(field_id, option)
    
    def set_extra_price(self, field_id: str, extra_price: int):
        self.set_field(field_id, extra_price)
    
    def set_product_max_quantity(self, field_id: str, max_quantity: int):
        self.set_field(field_id, max_quantity)
        
    def set_quantity(self, field_id: str, quantity: int):
        self.set_field(field_id, quantity)

    def set_total_price(self, field_id: str, total_price: int):
        self.set_field(field_id, total_price)


class Product_Selections(BaseModel):
    sender_id: str
    product_code: str


@app.post("/product-select")
def handle_product_selection(data: Product_Selections):
    try:
        sender_id = data.sender_id
        product_code = data.product_code

        if not sender_id or not product_code:
            return {
                "version": "v2",
                "content": {
                    "messages": [{"type": "text", "text": "❌ sender_id 또는 product_code가 없습니다."}]
                }
            }

        product = PRODUCT_CACHE.get(product_code)
        if not product:
            return {
                "version": "v2",
                "content": {
                    "messages": [{"type": "text", "text": f"❌ 상품코드 {product_code}에 대한 정보를 찾을 수 없습니다."}]
                }
            }
        
        # 가격, 옵션 정리
        price = int(float(product.get("가격", 0) or 0))
        shipping = int(float(product.get("배송비", 0) or 0))
        option_raw = product.get("조합형옵션", "").strip()

        option_display = "없음"
        if option_raw and option_raw.lower() != "nan":
            option_lines = option_raw.splitlines()
            parsed_options = []
            for line in option_lines:
                try:
                    name, extra_price, _ = line.split(",")
                    extra_price = int(float(extra_price))
                    price_str = f"(+{extra_price:,}원)" if extra_price > 0 else ""
                    parsed_options.append(f"{name.strip()} {price_str}".strip())
                except Exception:
                    parsed_options.append(line.strip())
            option_display = "\n".join(parsed_options)
        
        product["sender_id"] = sender_id
        
        # ✅ Manychat Field 업데이트
        updater = ManychatFieldUpdater(sender_id, MANYCHAT_API_KEY)
        updater.set_unique_code("12886380", product.get('상품코드'))
        updater.set_product_name("12886273", product.get('제목'))
        updater.set_option("12886363", option_display)
        updater.set_price("12890668", price)
        updater.set_shipping("12890670", shipping)
        updater.set_product_max_quantity("12922068", product.get('최대구매수량'))

        # ✅ 외부 Flow 트리거 (비동기처럼 요청 보내기)
        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        flow_payload = {
            "subscriber_id": sender_id,
            "flow_ns": "content20250417015933_369132"
        }
        try:
            res = requests.post(
                "https://api.manychat.com/fb/sending/sendFlow",
                headers=headers,
                json=flow_payload,
                timeout=5  # 실패해도 바로 리턴 안 끌려가게
            )
            print("✅ ManyChat Flow 전송 결과:", res.json())
        except Exception as e:
            print(f"❌ Flow 전송 실패: {e}")

        # ✅ 최종 클라이언트 응답 (Manychat Dynamic Block 규격)
        info_message = (
            f"상품코드\n{product.get('상품코드', '없음')}\n"
            f"제목\n{product.get('제목', '없음')}\n"
            f"원산지\n{product.get('원산지', '없음')}\n"
            f"------------------------------------------\n"
            f"가격\n{price:,}원\n"
            f"배송비\n{shipping:,}원\n"
            f"묶음배송수량\n{product.get('최대구매수량','0')}개\n"
            f"------------------------------------------\n"
            f"옵션\n{option_display}\n"
            f"------------------------------------------"
        ).strip()

        return {
            "version": "v2",
            "content": {
                "messages": [
                    {
                        "type": "text",
                        "text": info_message
                    }
                ]
            }
        }

    except Exception as e:
        print(f"❌ 상품 선택 처리 오류: {e}")
        return {
            "version": "v2",
            "content": {
                "messages": [{"type": "text", "text": f"❌ 서버 오류 발생: {str(e)}"}]
            }
        }



class Option_Selections(BaseModel):
    version: str
    field: str
    value: dict
    page: Optional[int] = 1


@app.post("/manychat-option-request")
def handle_option_request(data: Option_Selections):
    sender_id = data.value.get("sender_id") if isinstance(data.value, dict) else None
    product_code = data.value.get("product_code") if isinstance(data.value, dict) else None
    page = data.page or 1

    if not sender_id or not product_code:
        return {
            "version": "v2",
            "content": {
                "messages": [{"type": "text", "text": "❌ sender_id 또는 product_code가 없습니다."}]
            }
        }

    product = PRODUCT_CACHE.get(product_code)
    if not product:
        return {
            "version": "v2",
            "content": {
                "messages": [{"type": "text", "text": "❌ 상품 정보를 찾을 수 없습니다."}]
            }
        }

    options_raw = product.get("조합형옵션", "")
    if not options_raw or options_raw.lower() in ["nan", ""]:
        # ✅ 단일 옵션 상품일 경우 바로 다음 플로우로 이동
        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        flow_payload = {
            "subscriber_id": sender_id,
            "flow_ns": "content20250424050612_308842"
        }
        res = requests.post(
            "https://api.manychat.com/fb/sending/sendFlow",
            headers=headers,
            json=flow_payload
        )
        print("✅ 단일 옵션 상품 - Flow 전송 결과:", res.json())

        return {
            "version": "v2",
            "content": {
                "messages": [{"type": "text", "text": "🧾 This item has a single option — please select the quantity."}]
            }
        }

    options = options_raw.strip().split("\n")
    start_idx = (page - 1) * 27
    end_idx = start_idx + 27
    paged_options = options[start_idx:end_idx]

    message_batches = []
    current_buttons = []

    for opt in paged_options:
        try:
            name, extra_price, stock = opt.split(",")
            caption = f"{name.strip()} (+{int(float(extra_price)):,}원)" if float(extra_price) > 0 else name.strip()

            current_buttons.append({
                "type": "dynamic_block_callback",
                "caption": caption,
                "url": "https://viable-shark-faithful.ngrok-free.app/manychat-option-select",
                "method": "post",
                "headers": {
                    "Content-Type": "application/json"
                    },
                "payload": {
                    "sender_id": sender_id,
                    "selected_option": caption
                }
            })

            if len(current_buttons) == 3:
                message_batches.append({
                    "type": "text",
                    "text": "📌 Pick your preferred option:",
                    "buttons": current_buttons
                })
                current_buttons = []

        except Exception as e:
            print(f"⚠️ 옵션 파싱 실패: {opt} → {e}")
            continue

    if current_buttons:
        message_batches.append({
            "type": "text",
            "text": "📌 Pick your preferred option:",
            "buttons": current_buttons
        })

    # 다음 페이지 버튼 추가
    if end_idx < len(options):
        message_batches.append({
            "type": "text",
            "text": "👀 View Next Option 🧾",
            "buttons": [
                {
                    "type": "dynamic_block_callback",
                    "caption": "👀 View Next Option 🧾",
                    "url": "https://viable-shark-faithful.ngrok-free.app/manychat-option-request",
                    "method": "post",
                    "headers": {
                        "Content-Type": "application/json"
                        },
                    "payload": {
                        "version": "v2",
                        "field": "messages",
                        "value": {
                            "sender_id": sender_id,
                            "product_code": product_code
                        },
                        "page": page + 1
                    }
                }
            ]
        })

    return {
        "version": "v2",
        "content": {
            "messages": message_batches
        }
    }


@app.post("/manychat-option-select")
def handle_option_selection(payload: dict):
    sender_id = payload.get("sender_id")
    selected_option = payload.get("selected_option")

    if not sender_id or not selected_option:
        return {
            "version": "v2",
            "content": {
                "messages": [{"type": "text", "text": "❌ sender_id 또는 selected_option이 없습니다."}]
            }
        }

    # ✅ 추가금액 추출
    extra_price = 0
    match = re.search(r'\(\+([\d,]+)원\)', selected_option)
    if match:
        try:
            extra_price = int(match.group(1).replace(",", ""))
        except:
            extra_price = 0

    updater = ManychatFieldUpdater(sender_id, MANYCHAT_API_KEY)
    updater.set_product_selection_option("12904981", selected_option)
    updater.set_extra_price("12911810", extra_price)

    # ✅ 옵션 저장 후 Flow로 이동시키기
    headers = {
        "Authorization": f"Bearer {MANYCHAT_API_KEY}",
        "Content-Type": "application/json"
    }
    flow_payload = {
        "subscriber_id": sender_id,
        "flow_ns": "content20250424050612_308842"
    }
    res2 = requests.post(
        "https://api.manychat.com/fb/sending/sendFlow",
        headers=headers,
        json=flow_payload
    )
    print("✅ ManyChat Flow 전송 결과:", res2.json())

    return {
        "version": "v2",
        "content": {
            "messages": [
                {
                    "type": "text",
                    "text": f"✅ Option selected: {selected_option} (Extra: {extra_price:,})원)"
                }
            ]
        }
    }

class QuantityInput(BaseModel):
    sender_id: str
    product_quantity: int


def safe_int(val):
    try:
        return int(float(str(val).replace(",", "").replace("원", "").strip()))
    except:
        return 0


@app.post("/calculate_payment")
def calculate_payment(data: QuantityInput):
    try:
        sender_id = data.sender_id
        quantity = data.product_quantity

        if not sender_id or quantity is None:
            raise ValueError("❌ sender_id 또는 product_quantity 누락됨")

        # 🔍 캐시에서 상품 정보 불러오기
        product = None
        for p in PRODUCT_CACHE.values():
            if p.get("sender_id") == sender_id:
                product = p
                break

        if not product:
            raise ValueError("❌ 해당 유저의 상품 정보가 존재하지 않습니다.")

        # 🔢 기본 정보 추출
        price = safe_int(float(product.get("가격", 0)))
        extra_price = safe_int(float(product.get("추가금액", 0))) if "추가금액" in product else 0
        shipping = safe_int(float(product.get("배송비", 0)))
        max_quantity = safe_int(float(product.get("최대구매수량", 0)))

        # ✅ 총 가격 계산
        total_price = (price + extra_price) * quantity
        if max_quantity == 0:
            shipping_cost = shipping
        else:
            shipping_cost = shipping * math.ceil(quantity / max_quantity)

        total_price += shipping_cost

        # ✅ 천 단위 구분을 위한 포맷팅
        formatted_total_price = "{:,}".format(total_price)
        print(f"✅ 계산 완료 → 총금액: {formatted_total_price}원 (수량: {quantity}, 배송비: {shipping_cost:,}원)")

        # ✅ Manychat 필드 업데이트
        updater = ManychatFieldUpdater(sender_id, MANYCHAT_API_KEY)
        updater.set_quantity("12911653", quantity)  # Product_quantity 필드 ID
        updater.set_total_price("13013393", formatted_total_price)  # Total_price 필드 ID - 포맷팅된 값으로 저장

        # ✅ ManyChat 다음 Flow로 이동
        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        flow_payload = {
            "subscriber_id": sender_id,
            "flow_ns": "content20250501040123_213607"
        }
        res = requests.post(
            "https://api.manychat.com/fb/sending/sendFlow",
            headers=headers,
            json=flow_payload
        )
        print("✅ 최종결제금액 전송완료:", res.json())

        return {
            "Product_quantity": quantity,
            "Total_price": total_price
        }

    except Exception as e:
        print(f"❌ 결제 금액 계산 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ✅ 루트 경로 - HTML 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/preview", response_class=HTMLResponse)
async def product_preview(html: str):
    try:
        decoded_html = base64.b64decode(html).decode("utf-8")
        return f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <title>상품 상세 페이지</title>
            <style>
                body {{
                    font-family: '맑은 고딕', sans-serif;
                    padding: 20px;
                    max-width: 800px;
                    margin: auto;
                    line-height: 1.5;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px auto;
                }}
            </style>
        </head>
        <body>
            {decoded_html}
        </body>
        </html>
        """
    except Exception as e:
        return HTMLResponse(content=f"<h1>오류 발생</h1><p>{e}</p>", status_code=400)





'''
#######################################################################################################################

def generate_bot_response(user_message: str) -> str:
    """
    사용자의 메시지를 받아 챗봇 응답을 생성합니다.
    """
    try:
        # ✅ Redis를 이용한 세션 관리
        session_id = f"user_{user_message[:10]}"  # 간단한 세션 ID 생성 (필요 시 사용자 ID 사용)
        session_history = get_session_history(session_id)

        # ✅ Redis에서 기존 대화 이력 확인
        print(f"🔍 Redis 메시지 기록 (초기 상태): {session_history.messages}")

        # ✅ 사용자 입력을 기록에 추가
        session_history.add_message(HumanMessage(content=user_message))

        # ✅ LLM 기반 응답 생성
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "사용자 메시지에 따라 적절하고 친절한 응답을 생성하세요."),
            MessagesPlaceholder(variable_name="message_history"),
            ("human", user_message)
        ])
        runnable = prompt | llm
        response = runnable.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        )

        # ✅ Redis에 챗봇 응답 저장
        session_history.add_message(AIMessage(content=response.content))

        return response.content
    except Exception as e:
        print(f"❌ 응답 생성 오류: {e}")
        return "죄송합니다. 오류가 발생했습니다. 나중에 다시 시도해주세요."


# ✅ POST 요청 처리 - `/chatbot`
# search_and_generate_response는 UI 디자인이 된 웹 UI와 연결된 API 기본적인 API 요청을 통해 JSON 형태의 데이터를 주고 받음.

@app.post("/chatbot")
def search_and_generate_response(request: QueryRequest):
    query = request.query
    session_id = "redis123"  # 고정된 세션 ID


    reset_request = request.query.lower() == "reset"  # 'reset' 명령으로 초기화
    if reset_request:
        clear_message_history(session_id)
        return {
            "message": f"대화 기록이 초기화되었습니다."
        }



    print(f"🔍 사용자 검색어: {query}")

    try:
        # ✅ Redis 메시지 기록 관리
        session_history = get_session_history(session_id)
        # ✅ 기존 대화 내역 확인
        print(f"🔍 Redis 메시지 기록 (초기 상태): {session_history.messages}")

        # ✅ 기존 대화 내역 확인
        previous_queries = [
            msg.content for msg in session_history.messages if isinstance(msg, HumanMessage)
        ]
        print(f"🔍 Redis 메시지 기록: {previous_queries}")

        # ✅ LLM을 통한 키워드 추출 및 임베딩 생성
        combined_query = " ".join(previous_queries + [query])
        combined_keywords = extract_keywords_with_llm(combined_query)
        print(f"✅ 생성된 검색 키워드: {combined_keywords}")

        # ✅ Redis에 사용자 입력 추가
        session_history.add_message(HumanMessage(content=query))
        print(f"�� Redis 메시지 기록 (변경된 상태): {session_history.messages}")

        _, data = load_excel_to_texts("db/ownerclan_주간인기상품_0428.xlsx")

        # ✅ OpenAI 임베딩 생성
        query_embedding = embed_texts_parallel([combined_keywords], EMBEDDING_MODEL)
        faiss.normalize_L2(query_embedding)

        # ✅ FAISS 검색 수행(가장 가까운 상위 5개 벡터의 거리(D)와 인덱스(I)를 반환)
        D, I = index.search(query_embedding, k=5)

        # ✅ FAISS 검색 결과 검사
        if I is None or I.size == 0:
            return {
                "query": query,
                "results": [],
                "message": "검색 결과가 없습니다. 다른 키워드를 입력하세요!",
                "message_history": [
                    {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
                    for msg in session_history.messages
                ],
            }



        # ✅ 검색 결과 JSON 변환  (엑셀 속성을 따로 매칭)
        results = []
        for idx_list in I:  # 2차원 배열 처리
            for idx in idx_list:
                if idx >= len(data):  # 잘못된 인덱스 방지
                    continue
                result_row = data.iloc[idx]

                # 이미지 URL을 Base64로 변환
                image_url = result_row["이미지중"]

                result_info = {
                    "상품코드": str(result_row["상품코드"]),
                    "제목": result_row["원본상품명"],
                    "가격": convert_to_serializable(result_row["오너클랜판매가"]),
                    "배송비": convert_to_serializable(result_row["배송비"]),
                    "이미지": image_url,
                    "원산지": result_row["원산지"]
                }
                results.append(result_info)

        # ✅ results를 텍스트로 변환
        if results:
            results_text = "\n".join(
                [
                    f"상품코드: {item['상품코드']}, 제목: {item['제목']}, 가격: {item['가격']}원, "
                    f"배송비: {item['배송비']}원, 원산지: {item['원산지']}, 이미지: {item['이미지']}"
                    for item in results
                ]
            )
        else:
            results_text = "검색 결과가 없습니다."
                
        message_history=[]
        
        # ✅ ChatPromptTemplate 및 RunnableWithMessageHistory 생성
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""항상 message_history의 대화이력을 보면서 대화의 문맥을 이해합니다. 당신은 쇼핑몰 챗봇으로, 친절하고 인간적인 대화를 통해 고객의 쇼핑 경험을 돕는 역할을 합니다. 아래는 최근 검색된 상품 목록입니다.
            목표: 사용자의 요구를 명확히 이해하고, 이전 대화의 맥락을 기억해 자연스럽게 이어지는 추천을 제공합니다.
            작동 방식:
            이전 대화 내용을 기반으로 적합한 상품을 연결합니다.
            이건 대화 이력 문장을 보고 문맥을 이해하며, 사용자가 무슨 내용을 작성하고 상품을 찾는지 집중적으로 답변을 작성합니다.
            스타일: 따뜻하고 공감하며, 마치 실제 쇼핑 도우미처럼 친절하고 자연스럽게 응답합니다.
            대화 전략:
            사용자가 원하는 상품을 구체화하기 위해 적절한 후속 질문을 합니다.
            대화의 흐름이 끊기지 않도록 부드럽게 이어갑니다.
            목표는 단순한 정보 제공이 아닌, 고객이 필요한 상품을 정확히 찾을 수 있도록 돕는 데 중점을 둡니다. 당신은 이를 통해 고객이 편안하고 만족스러운 쇼핑 경험을 누릴 수 있도록 최선을 다해야 합니다."""),
            MessagesPlaceholder(variable_name="message_history"),
            ("system", f"다음은 대화이력입니다 : \n{message_history}"),
            ("system", f"다음은 상품결과입니다 : \n{results_text}"),
            ("human", query)
        ])
        
        runnable = prompt | llm

        with_message_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",  # 입력 메시지의 키
            history_messages_key="message_history",
        )

        

        # ✅ LLM 실행 및 메시지 기록 업데이트
        response = with_message_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )

        # ✅ Redis에 AI 응답 추가
        session_history.add_message(AIMessage(content=response.content))

        # ✅ 메시지 기록을 Redis에서 가져오기
        session_history = get_session_history(session_id)
        message_history = [
            {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
            for msg in session_history.messages
        ]


        # ✅ 출력 디버깅
        print("*** Response:", response)
        print("*** Message History:", message_history)
        print("✅*✅*✅* Results:", results)

        # ✅ JSON 반환
        return {
            "query": query,
            "results": results,
            "response": response.content,
            "message_history": message_history
        }

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        '''

# ✅ FastAPI 서버 실행 (포트 고정: 5050)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
