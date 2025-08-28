import asyncio
import base64
import json
import logging
import os
import re
import time
import urllib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, List, Dict, Any, Tuple
from urllib.parse import quote
import math
import random

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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from pydantic import BaseModel

from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema,
    DataType, Collection
)
from langdetect import detect
from openai import OpenAI as OpenAIClient      # 공식 OpenAI 클라이언트
import uvicorn

from collections import defaultdict, Counter
from itertools import zip_longest

executor = ThreadPoolExecutor()

# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
REDIS_URL = "redis://localhost:6379/0"
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
MANYCHAT_API_KEY = os.getenv('MANYCHAT_API_KEY')
key = os.getenv("MANYCHAT_API_KEY")
if isinstance(key, str) and "\x3a" in key:
    key = key.replace("\x3a", ":")
LLM_MODEL  = "gpt-4.1-mini"
EMB_MODEL  = "text-embedding-3-small"
max_total=10  #몇개의 상품을 나올지



# 클라이언트 및 래퍼
client    = OpenAIClient(api_key=API_KEY)
llm       = OpenAI(api_key=API_KEY, model=LLM_MODEL, temperature=0)
embedder  = OpenAIEmbeddings(api_key=API_KEY, model=EMB_MODEL)    # ← embedder 정의 추가


# API_URL = os.getenv("API_URL", "").rstrip("/")  # 예: 
API_URL = "https://fb-narosu.duckdns.org"  # 예: 
print(f"🔍 로드된 VERIFY_TOKEN: {VERIFY_TOKEN}")
print(f"🔍 로드된 PAGE_ACCESS_TOKEN: {PAGE_ACCESS_TOKEN}")
print(f"🔍 로드된 API_KEY: {API_KEY}")
print(f"🔍 로드된 API_URL: {API_URL}")

# # ✅ FAISS 인덱스 파일 경로 설정
# faiss_file_path = f"04_28_faiss_3s.faiss"

# ─── Milvus import & 연결 ───────────────────────────────────────────────
# 올바른 공인 IP와 포트
connections.connect(
    alias="default",
    host="114.110.135.96",
    port="19530"
)
print("✅ Milvus에 연결되었습니다.")



# OpenAI Embedding 모델 (쿼리용)
emb_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
# ────────────────────────────────────────────────────────────────────────

collection_cat = Collection("category_0821")
results = collection_cat.query(
    expr="category_full != ''",
    output_fields=["category_full"]
)

# ── 중복 제거하며 순서 보존해서 리스트 만들기 ─────────
seen = set()
categories = []
for row in results:
    cat = row["category_full"]
    if cat and cat not in seen:
        seen.add(cat)
        categories.append(cat)

print(f"✅ Milvus에서 불러온 카테고리 개수: {len(categories)}")

# 컬렉션 이름
collection_name = "ownerclan_weekly_0428"

# 컬렉션 객체 생성 (조회 용도)
collection = Collection(name=collection_name)
# 💡 저장된 벡터 수 확인
print(f"\n📊 저장된 엔트리 수: {collection.num_entities}")

def get_redis():
    return redis.Redis.from_url(REDIS_URL)

# ✅ FastAPI 인스턴스 생성
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[API_URL,  # 실제 배포 URL
                  "http://localhost:5050"],

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

# 요청 모델
class QueryRequest(BaseModel):
    query: str

# ✅ JSON 직렬화를 위한 int 변환 함수
def convert_to_serializable(obj):
    # None 값 처리
    if obj is None:
        return 999
    
    # 숫자 타입 처리
    if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
        val = obj.item()
        return max(0, val)  # 음수값은 0으로 처리
    
    # 문자열 처리 (숫자로 변환 가능한 경우)
    if isinstance(obj, str):
        try:
            val = float(obj.replace(",", ""))
            return max(0, int(val))  # 음수값은 0으로 처리
        except:
            return 999
    
    # 그 외의 경우
    try:
        val = float(obj)
        return max(0, int(val))  # 음수값은 0으로 처리
    except:
        return 999

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Redis에서 세션 기록을 가져옵니다."""
    try:
        history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
        # 세션이 존재하는지 확인
        messages = history.messages
        return history
    except Exception as e:
        print(f"❌ 대화 기록 가져오기 오류: {e}")
        # 에러 발생 시 새로운 히스토리 객체 생성
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

def quota_to_text(quota: Dict[str, int]) -> str:
    return "\n".join([f'- {cat}: {q}개' for cat, q in quota.items()])

def compute_category_proportions(
    candidates: List[Dict[str, Any]]
) -> Dict[str, float]:
    total = len(candidates)
    if total == 0:
        return {}
    counts = Counter(item["카테고리"] for item in candidates)
    return {cat: cnt / total for cat, cnt in counts.items()}


# 🔥 상품 캐시 (전역 선언)
PRODUCT_CACHE = {}
# 🔗 구매하기 버튼 클릭 시 호출되는 ManyChat용 Hook 주소
MANYCHAT_HOOK_BASE_URL = f"{API_URL}/product-select"


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
                                    "text": f"🔄 All cleaned up and ready to start~ \n💬 Enter a keyword and let the AI work its magic 🛍️."
                                }
                            ]
                        },
                        "message": f"{sender_id}님의 대화 기록이 초기화되었습니다."
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
                            "url": f"{API_URL}/product-select",
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
            # print("✅ 최종 messages_data:", json.dumps(messages_data, indent=2, ensure_ascii=False))

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

##=========================================================================
# 디버깅용 요청 모델
class DebugRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
 
# 디버깅 엔드포인트 추가
@app.post("/debug-search")
async def debug_search(data: DebugRequest):
    """
    external_search_and_generate_response를 바로 호출해서
    결과 payload를 JSON으로 반환합니다.
    """
    try:
        # sync 함수라도 바로 호출 가능
        result = external_search_and_generate_response(data.query, data.session_id)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
##=========================================================================
        

'''####################################################################################################################
external_search_and_generate_response는 ManyChat 같은 외부 서비스와 연동되는 챗봇용 API이고, 구축된 UI 에는 사용되지 않음.
'''

# ✅ 외부 검색 및 응답 생성 함수
def external_search_and_generate_response(request: Union[QueryRequest, str], session_id: str = None) -> dict:
    try:
        start_time = time.time()  # 시작 시간 기록
        
        # ✅ 입력 쿼리 추출 및 타입 확인
        query = request if isinstance(request, str) else request.query
        print(f"🔍 사용자 검색어: {query}")
        
        if not isinstance(query, str):
            raise TypeError(f"❌ [ERROR] 잘못된 query 타입: {type(query)}")
    
        # ✅ 세션 초기화 명령 처리
        if query.lower() == "reset":
            if session_id:
                clear_message_history(session_id)
            return {"message": f"세션 {session_id}의 대화 기록이 초기화되었습니다."}
    





        # ✅ Redis 세션 기록 불러오기 및 최신 입력 저장
        session_history = get_session_history(session_id)
        session_history.add_user_message(query)
    
        previous_queries = [msg.content for msg in session_history.messages if isinstance(msg, HumanMessage)]
        if query in previous_queries:
            previous_queries.remove(query)
        
        # ✅ 전체 중복 제거 (최신 입력을 제외한 나머지에서)
        previous_queries = list(dict.fromkeys(previous_queries))
        

        raw = detect(query)
        lang_code = raw.lower().split("-")[0]   # "EN-us" → "en"

        #가격을 이해하는 매핑
        pattern = re.compile(r'(\d+)[^\d]*원\s*(이하|미만|이상|초과)')
        m = pattern.search(query)
        if m:
            amount = int(m.group(1))
            comp  = m.group(2)
            # 부등호 매핑
            op_map = {"이하":"<=", "미만":"<", "이상":">=", "초과":">"}
            price_op = op_map[comp]
            price_cond = f"market_price {price_op} {amount}"
        else:
            # 디폴트: 제한 없음
            price_cond = None
        





        # 2) 언어 코드 → 사람말 매핑
        lang_map = {
            "ko": "한국어",
            "en": "English",
            "zh-cn": "中文",
            "ja": "日本語",
            "vi": "Tiếng Việt",  # 베트남어
            "th": "ไทย",        # 태국어
        }
        

        target_lang = lang_map.get(lang_code, "한국어")
        print("[Debug] Detected language →", target_lang)

        # 시즌 판단
        # 시즌 관련 상수 정의
        SPRING_KEYWORDS = [
            # 한국어
            "봄","봄맞이","봄소풍","봄나들이","벚꽃","꽃샘추위","춘분","간절기","환절기",
            "트렌치코트","스프링코트","바람막이","가디건","플로럴","꽃무늬","파스텔",
            # English
            "spring","vernal","cherry blossom","blossom","cold snap","windbreaker",
            "trench coat","spring coat","cardigan","floral","floral print","pastel","layering"
        ]

        SUMMER_KEYWORDS = [
            # 한국어
            "여름","썸머","자외선차단","uv차단","썬캡","선바이저","비치","바캉스",
            "래시가드","수영복","비키니","아쿠아슈즈","워터레깅스","비치샌들",
            "라피아","스트로","스트로우","햇빛가리개","쿨링","냉감","쿨터치",
            "흡한속건","속건","통풍","메쉬","린넨","리넨","시어서커","UPF",
            # English
            "summer","uv","uv protection","uv-cut","upf","sun protection","sun cap","sun visor",
            "beach","beachwear","rashguard","rash guard","swimsuit","bikini",
            "aqua shoes","water shoes","water leggings","flip-flops","sandals",
            "raffia","straw","straw hat","sun shade","neck shade",
            "cooling","cool-touch","quick-dry","moisture wicking","breathable","ventilated","mesh","linen","seersucker"
        ]

        AUTUMN_KEYWORDS = [
            # 한국어
            "가을","오텀","단풍","추석","한가위","가을맞이","수확제","간절기","환절기",
            "플란넬","코듀로이","트위드","체크","체커드","셔켓","오버셔츠","울혼방","니트베스트",
            # English
            "autumn","fall","fall foliage","foliage","chuseok","harvest festival",
            "flannel","corduroy","tweed","plaid","checkered","shacket","overshirt","wool blend","layering"
        ]

        WINTER_KEYWORDS = [
            # 한국어
            "겨울","윈터","방한","보온","발열","기모","기모안감","플리스","보아","보아털","쉐르파",
            "다운","패딩","롱패딩","구스다운","덕다운","웰론","충전재",
            "스노우부츠","핫팩","핫팩손난로","바라클라바","넥워머","귀마개",
            "목도리","스카프","머플러","장갑","내복","롱존","이너웨어",
            "방풍","방수","발수",
            "비니","니트","울","캐시미어","퍼","두꺼운옷","두툼한","보온성",
            # English
            "winter","thermal","heat-retaining","fleece","boa","sherpa",
            "down","goose down","duck down","synthetic down","insulation","puffer","parka",
            "snow boots","hot pack","hand warmer","balaclava","neck warmer","earmuffs",
            "scarf","muffler","gloves","mittens","thermal underwear","long johns","base layer",
            "windproof","waterproof","water-repellent",
            "beanie","wool","cashmere","fur","thick","lined"
        ]


        def detect_season(query: str) -> str:
            """쿼리에서 시즌 정보를 감지하는 함수"""
            query_lower = query.lower()
            
            for season, keywords in [ 
                ("봄", SPRING_KEYWORDS),
                ("여름", SUMMER_KEYWORDS),
                ("가을", AUTUMN_KEYWORDS),
                ("겨울", WINTER_KEYWORDS)
            ]:
                if any(keyword in query_lower for keyword in keywords):
                    return season
            return "미정"

        # 시즌 설정
        season = detect_season(query)
        print(f"🌞 감지된 시즌: {season}")

        # LLM 전처리
        llm = OpenAI(
            api_key=API_KEY,
            model=LLM_MODEL,
            temperature=0
        )
        # 대화 이력 가져오기
        history_messages = [msg.content for msg in session_history.messages]
        conversation_context = "\n".join([f"이전 대화: {msg}" for msg in history_messages[-3:]]) if history_messages else "이전 대화 없음"

        system_prompt = (
            f"""System:
                당신은 (1) 검색 엔진의 전처리를 담당하는 AI이자, (2) 쇼핑몰 검색 및 분류 전문가입니다.
                입력 언어가 무엇이든 먼저 한국어로 의미 보존 번역을 수행합니다.
                
                [대화 컨텍스트]
                {conversation_context}

                [전처리 목표]
                - 오타 교정, 불용어/군더더기 제거, 중복 표현 제거
                - 이전 대화 맥락을 고려하여 현재 검색어의 의도 파악
                - 핵심 품목명(한 개)과 의미 있는 속성(계절/성별/색상/사이즈/용량/재질 등)만 남김
                - 동의어/구어 표준화(남자→남성, 여자→여성, 남녀공용/유니섹스→공용, 여름철→여름 등)
                - 숫자+단위 결합(128 GB→128GB, 5000 mAh→5000mAh)

                [특별 규칙: ‘용’ 접미사 정규화]
                - “봄/여름/가을/겨울/간절기/사계절/남성/여성/공용/유아/아동/키즈/성인 + 용” → 접미사 ‘용’ 제거
                예) 여름용 모자→여름 모자, 남성용 등산 바지→남성 등산 바지
                - 단, 의미어는 보존: 전용/공용/용량/내용은 그대로 유지 (예: “닌텐도 전용 케이스”에서 ‘전용’ 삭제 금지)
                - “~에 쓸 수 있는/쓰기 좋은/사용하기 좋은” 등 군더더기는 삭제
                - ‘용도’라는 일반어는 제거

                [금지사항]
                - 고객 검색어를 임의로 확장/추정하지 말 것(예: “스마트폰” → “스마트폰용 이어폰” 금지)
                - 브랜드/모델/카테고리를 새로 만들지 말 것
                - 불필요한 형용사·수식 남발 금지

                [출력 규칙(반드시 정확히 준수)]
                오직 두 줄만 출력, 따옴표 포함. 추가 설명/불릿/번호/코드블록 절대 금지.
                Raw Query: "<query>"
                Preprocessed Query: "<전처리된_쿼리(핵심 품목 + 유의미 속성만, ‘용’ 제거 후 표준형)>"


            """    
        )

        if price_cond:
            system_prompt += f"\n⚠️ 사용자 요청 조건: 가격은 **{amount}원 {comp}** ({price_cond})인 상품만 고려하세요.\n"



        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query}
            ]
        )
        llm_response = resp.choices[0].message.content.strip()
        print("[Debug] LLM full response:\n", llm_response)  # ← 여기에!


        #LLM 응답 파싱
        lines = [l.strip() for l in llm_response.splitlines() if l.strip()]

        def extract_preprocessed(llm_text: str, fallback: str) -> str:
            # 0) 혹시 JSON으로 온 경우 대비
            try:
                obj = json.loads(llm_text)
                v = (obj.get("Preprocessed Query") or
                    obj.get("preprocessed_query") or
                    obj.get("final_query") or
                    obj.get("query_preprocessed"))
                if v and isinstance(v, str) and v.strip():
                    return v.strip()
            except Exception:
                pass

            # 1) 영문 라벨 (권장 형식)
            m = re.search(r'(?i)^\s*preprocessed\s*query\s*:\s*["“]?(.+?)["”]?\s*$', llm_text, re.M)
            if m:
                return m.group(1).strip()

            # 2) 한글 라벨(모델이 한국어로 라벨을 바꿔도 커버)
            for pat in [
                r'^\s*최종\s*검색어\s*[:=]\s*["“]?(.+?)["”]?\s*$',
                r'^\s*전처리된_?쿼리\s*[:=]\s*["“]?(.+?)["”]?\s*$',
            ]:
                m = re.search(pat, llm_text, re.M)
                if m:
                    return m.group(1).strip()

            # 3) 마지막 폴백: 따옴표 속 가장 그럴듯한 문자열
            quotes = re.findall(r'["“]([^"\n”]{2,64})["”]', llm_text)
            if quotes:
                # 공백 포함/길이 기준 간단 휴리스틱
                quotes.sort(key=lambda s: (-(' ' in s), -len(s)))
                return quotes[0].strip()

            # 4) 최종 폴백: 원문
            return fallback

        preprocessed_query = extract_preprocessed(llm_response, query)
        print("[Debug] Preprocessed Query ->", preprocessed_query)




        # --- 쿼리 임베딩 (L2 정규화) ---
        q_vec = np.array(embedder.embed_query(preprocessed_query), dtype=np.float32)
        n = np.linalg.norm(q_vec)
        if np.isfinite(n) and n != 0.0:
            q_vec = q_vec / n
        print(f"[Debug] q_vec dim: {q_vec.shape}, norm: {np.linalg.norm(q_vec):.4f}")

        # --- category_0821에서 L2로 Top5 카테고리 검색 ---
        CAT_COLLECTION = "category_0821"

        def get_top5_categories_from_embeddings(qv: np.ndarray):
            cat_col = Collection(CAT_COLLECTION)
            res = cat_col.search(
                data=[qv],
                anns_field="embedding",
                param={"metric_type":"L2","params":{"nprobe":64}},
                limit=5,
                output_fields=["category_full"]
            )
            top5 = [(hit.entity.get("category_full"), float(hit.distance)) for hit in res[0]]
            print("🔎 카테고리 Top5 (L2, 작을수록 유사):")
            for i,(name,dist) in enumerate(top5,1):
                print(f"  {i}. {name} | L2={dist:.6f}")
            return top5

        top5_cats = get_top5_categories_from_embeddings(q_vec)

        print("\n🔎 (원본) 카테고리 Top5:")
        for i,(name,dist) in enumerate(top5_cats,1):
            print(f"  {i}. {name} | L2={dist:.6f}")

        # --- (옵션) 시즌 보정만 유지: 여름/겨울이면 살짝 정렬 보정 ---
        # 시즌 보정을 위한 긍정/부정 세트 정의
        _SEASON_POS = {
            "봄": tuple(SPRING_KEYWORDS),
            "여름": tuple(SUMMER_KEYWORDS),
            "가을": tuple(AUTUMN_KEYWORDS),
            "겨울": tuple(WINTER_KEYWORDS),
        }

        # 봄/가을엔 극여름·극겨울 단어 감점, 여름/겨울엔 서로 반대 시즌만 감점
        _SEASON_NEG = {
            "봄": tuple(SUMMER_KEYWORDS + WINTER_KEYWORDS),
            "가을": tuple(SUMMER_KEYWORDS + WINTER_KEYWORDS),
            "여름": tuple(WINTER_KEYWORDS),
            "겨울": tuple(SUMMER_KEYWORDS),
        }

        def _season_adjust(name: str, season_hint: str) -> float:
            """
            카테고리명(name)에 시즌 강 키워드가 있으면 L2 거리를 미세 조정.
            - 보너스(거리↓): 해당 시즌의 강 키워드 포함
            - 페널티(거리↑): 반대(혹은 극단) 시즌 강 키워드 포함
            """
            if season_hint not in _SEASON_POS:
                return 0.0

            n = name.lower()
            pos_hit = any(w.lower() in n for w in _SEASON_POS[season_hint])
            neg_hit = any(w.lower() in n for w in _SEASON_NEG[season_hint])

            POS_BONUS   = 0.04   # 거리 감소 → 순위 소폭 상승
            NEG_PENALTY = 0.12   # 거리 증가 → 순위 소폭 하향

            adj = 0.0
            if pos_hit: adj -= POS_BONUS
            if neg_hit: adj += NEG_PENALTY
            return adj

        def season_filter_items(items: list, season_hint: str):
            """시즌에 맞는 상품만 필터링하여 반환"""
            if not season_hint or season_hint == "미정" or not items:
                return items
                
            def matches_season(item_name: str, season: str) -> bool:
                name_lower = item_name.lower()
                
                # 해당 시즌의 키워드가 있으면 True
                if season == "봄" and any(kw.lower() in name_lower for kw in SPRING_KEYWORDS):
                    return True
                elif season == "여름" and any(kw.lower() in name_lower for kw in SUMMER_KEYWORDS):
                    return True
                elif season == "가을" and any(kw.lower() in name_lower for kw in AUTUMN_KEYWORDS):
                    return True
                elif season == "겨울" and any(kw.lower() in name_lower for kw in WINTER_KEYWORDS):
                    return True
                    
                # 반대 시즌 키워드가 있으면 False
                if season == "여름" and any(kw.lower() in name_lower for kw in WINTER_KEYWORDS):
                    return False
                elif season == "겨울" and any(kw.lower() in name_lower for kw in SUMMER_KEYWORDS):
                    return False
                    
                # 아무 키워드도 없으면 True (중립)
                return True
            
            # 필터링 적용
            filtered = [
                item for item in items 
                if matches_season(item.get("제목", ""), season_hint) and 
                matches_season(item.get("카테고리", ""), season_hint)
            ]
            
            return filtered if filtered else items  # 필터링 결과가 없으면 원본 반환

        # season 기본값 보장 (이미 위에서 세팅되어 있으면 이 부분은 그대로 둬도 됨)
        try:
            season
        except NameError:
            season = "미정"

        # 적용: 4계절 모두
        if season in ("봄","여름","가을","겨울"):
            top5_cats = sorted(
                [(n, d + _season_adjust(n, season)) for (n, d) in top5_cats],
                key=lambda x: x[1]
            )
            print("🛠 시즌 보정 후 Top5:")
            for i,(name,dist) in enumerate(top5_cats,1):
                print(f"  {i}. {name} | adj_L2={dist:.6f}")


        # 상위 3개 카테고리에 대해 각각 상품 검색 (시즌 보정 적용)
        def get_adjusted_score(category_name: str, score: float, season: str) -> float:
            """시즌에 따른 점수 보정"""
            season_adj = _season_adjust(category_name, season)
            return score + season_adj

        top3_products = []
        sorted_cats = sorted(
            top5_cats[:3],
            key=lambda x: get_adjusted_score(x[0], x[1], season)
        )

        # 카테고리별 검색 수량 설정
        CATEGORY_QUOTAS = {
            "Top1": {"direct": 25, "vector": 25},  # 총 50개
            "Top2": {"direct": 15, "vector": 15},  # 총 30개
            "Top3": {"direct": 10, "vector": 10}   # 총 20개
        }

        def search_products_by_category(cat_name: str, cat_rank: str, query_tokens: List[str], query_vec: np.ndarray, vector_limit: Optional[int] = None):
            """
            카테고리별 직접 검색과 벡터 검색을 수행하는 함수
            Args:
                cat_name: 카테고리 이름
                cat_rank: "Top1", "Top2", "Top3" 등 카테고리 순위
                query_tokens: 검색어 토큰
                query_vec: 쿼리 벡터
                vector_limit: 벡터 검색 제한 수 (선택적, None이면 기본 할당량 사용)
            Returns:
                direct_hits, vector_hits
            """
            quota = CATEGORY_QUOTAS.get(cat_rank, {"direct": 10, "vector": 10})
            
            # 1. 직접 검색
            direct_expr = f"category_name like '%{cat_name}%' && " + " && ".join(
                f'market_product_name like "%{tok}%"' for tok in query_tokens
            )
            
            output_fields = [
                "product_code", "category_code", "category_name", 
                "market_product_name", "market_price", "shipping_fee",
                "shipping_type", "max_quantity", "composite_options",
                "image_url", "manufacturer", "model_name", "origin",
                "keywords", "description", "return_shipping_fee"
            ]
            
            direct_hits = collection.query(
                expr=direct_expr,
                limit=quota["direct"],
                output_fields=output_fields
            )
            
            # 2. 벡터 검색 (동적 limit 적용)
            actual_limit = vector_limit if vector_limit is not None else quota["vector"]
            vector_hits = collection.search(
                data=[query_vec],
                anns_field="emb",
                param={"metric_type": "L2", "params": {"nprobe": 64}},
                limit=actual_limit,
                expr=f"category_name like '%{cat_name}%'",
                output_fields=output_fields
            )
            
            return direct_hits, vector_hits[0]

        # 카테고리별 검색 실행
        tokens = [t for t in re.sub(r"[용\s]+", " ", preprocessed_query).split() if t]

        for idx, (cat_name, base_score) in enumerate(sorted_cats, 1):
            cat_rank = f"Top{idx}"
            adj_score = get_adjusted_score(cat_name, base_score, season)
            print(f"\n🔍 카테고리 '{cat_name}' 검색 시작... (시즌 보정 점수: {adj_score:.6f})")
            
            # 카테고리별 검색 실행 - 처음에는 직접 검색
            direct_hits, vector_hits = search_products_by_category(cat_name, cat_rank, tokens, q_vec)
            direct_target = CATEGORY_QUOTAS[cat_rank]['direct']
            vector_target = CATEGORY_QUOTAS[cat_rank]['vector']
            
            # 직접 검색 결과가 목표에 미달인 경우 벡터 검색 쿼터 증가
            if len(direct_hits) < direct_target:
                shortage = direct_target - len(direct_hits)
                vector_target += shortage
                print(f"  ⚠️ 직접 검색 부족분 {shortage}개를 벡터 검색으로 보충 시도")
                
                # 벡터 검색 재시도 (증가된 쿼터로)
                _, additional_hits = search_products_by_category(cat_name, cat_rank, tokens, q_vec, vector_limit=vector_target)
                vector_hits = additional_hits
            
            print(f"  ┣ 직접 검색 결과: {len(direct_hits)}개 (목표: {direct_target}개)")
            print(f"  ┗ 벡터 검색 결과: {len(vector_hits)}개 (목표: {vector_target}개)")
            
            # 결과 통합 및 가공
            cat_products = []

            def process_hit(hit, is_vector=False):
                try:
                    if is_vector:
                        hit = hit.entity

                    html_raw = hit.get("description", "") or ""
                    html_cleaned = clean_html_content(html_raw)
                    if isinstance(html_raw, bytes):
                        html_raw = html_raw.decode("cp949")
                    encoded_html = base64.b64encode(html_cleaned.encode("utf-8", errors="ignore")).decode("utf-8")
                    preview_url = f"{API_URL}/preview?html={urllib.parse.quote_plus(encoded_html)}"
                    
                    # 최대구매수량 처리 (기본값 999)
                    max_quantity = hit.get("max_quantity", 999)
                    try:
                        max_quantity = convert_to_serializable(max_quantity)
                        if max_quantity is None or not isinstance(max_quantity, (int, float)) or max_quantity < 0:
                            max_quantity = 999
                    except:
                        max_quantity = 999

                    return {
                        "상품코드": str(hit.get("product_code", "없음")),
                        "제목": hit.get("market_product_name", "제목 없음"),
                        "가격": convert_to_serializable(hit.get("market_price", 0)),
                        "배송비": convert_to_serializable(hit.get("shipping_fee", 0)),
                        "이미지": hit.get("image_url", "이미지 없음"),
                        "원산지": hit.get("origin", "정보 없음"),
                        "상품링크": preview_url,
                        "카테고리": hit.get("category_name", "카테고리 없음"),
                        "검색방식": "벡터검색" if is_vector else "직접검색",
                        "순위점수": idx,  # 카테고리 순위 정보 추가
                        "최대구매수량": max_quantity
                    }
                except Exception as e:
                    print(f"  ⚠️ 상품 정보 처리 오류: {e}")
                    return None
            
            # 직접 검색 결과 처리
            for hit in direct_hits:
                product = process_hit(hit)
                if product:
                    cat_products.append(product)
            
            # 벡터 검색 결과 처리
            for hit in vector_hits:
                product = process_hit(hit, is_vector=True)
                if product:
                    cat_products.append(product)
            
            # 중복 제거 (상품코드 기준) 및 카테고리 순위 유지
            seen_codes = set()
            unique_products = []
            for p in cat_products:
                if p["상품코드"] not in seen_codes:
                    seen_codes.add(p["상품코드"])
                    unique_products.append(p)
            
            # 시즌 필터링 적용
            unique_products = season_filter_items(unique_products, season)
            
            quota = CATEGORY_QUOTAS[f"Top{idx}"]
            target_count = quota["direct"] + quota["vector"]
            print(f"  ✅ 최종 상품 수: {len(unique_products)}개 / 목표: {target_count}개")
            top3_products.extend(unique_products)

        # 무한 스크롤을 위한 함수 - 방법 2: 카테고리 검색 5:3:2
        def get_next_products(products: List[Dict], top5_cats: List[Tuple[str, float]], offset: int = 0) -> Tuple[List[Dict], bool]:
            """
            다음 상품들을 반환하는 함수 (5:3:2 비율 적용)
            Args:
                products: 전체 상품 리스트
                top5_cats: 카테고리 순위 [(category_name, distance), ...]
                offset: 이전까지 로드된 상품 수
            Returns:
                (다음 상품 리스트, 더 불러올 상품이 있는지 여부)
            """
            BATCH_SIZE = 10  # 한 번에 로드할 상품 수
            
            # 카테고리별 쿼터 (5:3:2 비율)
            QUOTA = {"Top1": 5, "Top2": 3, "Top3": 2}
            
            # 전체 상품이 없으면 빈 결과 반환
            if not products:
                return [], False
                
            # 카테고리별로 상품 분류
            products_by_category = defaultdict(list)
            for product in products:
                for cat_idx, (cat_name, _) in enumerate(top5_cats[:3]):  # Top3만 처리
                    cat_key = f"Top{cat_idx+1}"
                    if product["카테고리"].startswith(cat_name):
                        products_by_category[cat_key].append(product)
                        break
            
            # 결과 수집
            current_batch = []
            all_remaining_products = []
            
            # 1) 각 카테고리별로 상품 수집
            for cat in ["Top1", "Top2", "Top3"]:
                pool = products_by_category[cat]
                start = offset
                quota = QUOTA[cat]
                
                # 현재 카테고리의 남은 상품 추가
                available = pool[start:start + quota]
                current_batch.extend(available)
                
                # 남은 상품들 저장
                all_remaining_products.extend(pool[start + quota:])
            
            # 2) 부족분을 남은 상품들로 채우기
            if len(current_batch) < BATCH_SIZE and all_remaining_products:
                needed = BATCH_SIZE - len(current_batch)
                current_batch.extend(all_remaining_products[:needed])
            
            # 3) 중복 제거
            seen = set()
            unique_batch = []
            for item in current_batch:
                key = item["상품코드"]
                if key not in seen:
                    seen.add(key)
                    unique_batch.append(item)
            
            # 4) 다음 페이지 여부 확인
            # 현재 오프셋 이후에 남은 상품이 있는지 확인
            total_remaining = len([p for p in products if p["상품코드"] not in seen])
            has_more = total_remaining > 0
            
            return unique_batch[:BATCH_SIZE], has_more

        # 전체 결과 요약
        print("\n📊 검색 결과 요약:")
        for cat_name, _ in top5_cats[:3]:
            cat_products = [p for p in top3_products if p["카테고리"].startswith(cat_name)]
            direct_count = sum(1 for p in cat_products if p["검색방식"] == "직접검색")
            vector_count = sum(1 for p in cat_products if p["검색방식"] == "벡터검색")
            print(f"  {cat_name}:")
            print(f"    - 직접검색: {direct_count}개")
            print(f"    - 벡터검색: {vector_count}개")
            print(f"    - 총 {len(cat_products)}개")

        # 방법 1의 무한 스크롤 테스트 (5:3:2 비율의 카테고리별 상품리스트)
        print("\n📖 방법 1의 상품리스트 테스트 (5:3:2 카테고리 비율):")
        offset = 0
        method1_shown_products = []  # 이름 변경
        seen_products = set()  # 중복 체크를 위한 세트

        while True:
            next_products, has_more = get_next_products(top3_products, top5_cats, offset)
            if not next_products:
                break
            
            # 중복 제거하며 새로운 상품들 추가
            new_products = []
            for product in next_products:
                product_key = f"{product['상품코드']}_{product['카테고리']}"
                if product_key not in seen_products:
                    seen_products.add(product_key)
                    new_products.append(product)
                    method1_shown_products.append(product)  # 이름 변경
            
            if new_products:  # 새로운 상품이 있을 때만 출력
                print(f"\n=== 방법 1의 상품리스트 추가 로드 (전체 {len(method1_shown_products)}개) ===")  # 텍스트 변경
                for idx, product in enumerate(new_products, len(method1_shown_products) - len(new_products) + 1):
                    print(f"[{idx}] {product['제목']}")
                    print(f"    카테고리: {product['카테고리']}")
                    print(f"    가격: {product['가격']:,}원")
            
            if not has_more:
                print("\n✅ 방법 1의 모든 상품을 불러왔습니다.")  # 텍스트 변경
                print(f"   총 {len(method1_shown_products)}개의 상품이 로드되었습니다.")  # 이름 변경
                break
                
            offset += len(new_products)  # 실제로 추가된 새 상품 수만큼만 증가






        # ================================================================
        # [Top2-Stage] 전 카테고리 대상 직접50 + 벡터50 → 100 후보(raw_candidates)
        #              → 시즌 필터 → 카테고리 Top2 추출 → 10개 비율 분배
        #              → 카테고리별 LLM 재랭킹 → 버튼 페이지 로더
        # ================================================================

        def season_filter_items(items: list, season_hint: str):
            """시즌에 맞는 상품만 필터링하여 반환"""
            if not season_hint or season_hint == "미정" or not items:
                return items
                
            def matches_season(item_name: str, season: str) -> bool:
                name_lower = item_name.lower()
                
                # 해당 시즌의 키워드가 있으면 True
                if season == "봄" and any(kw.lower() in name_lower for kw in SPRING_KEYWORDS):
                    return True
                elif season == "여름" and any(kw.lower() in name_lower for kw in SUMMER_KEYWORDS):
                    return True
                elif season == "가을" and any(kw.lower() in name_lower for kw in AUTUMN_KEYWORDS):
                    return True
                elif season == "겨울" and any(kw.lower() in name_lower for kw in WINTER_KEYWORDS):
                    return True
                    
                # 반대 시즌 키워드가 있으면 False
                if season == "여름" and any(kw.lower() in name_lower for kw in WINTER_KEYWORDS):
                    return False
                elif season == "겨울" and any(kw.lower() in name_lower for kw in SUMMER_KEYWORDS):
                    return False
                    
                # 아무 키워드도 없으면 True (중립)
                return True
            
            # 필터링 적용
            filtered = [
                item for item in items 
                if matches_season(item.get("제목", ""), season_hint) and 
                matches_season(item.get("카테고리", ""), season_hint)
            ]
            
            return filtered if filtered else items  # 필터링 결과가 없으면 원본 반환

        def _has_any(text: str, words) -> bool:
            t = (text or "").lower()
            return any(w.lower() in t for w in words)

        def _build_info_from_row(row):
            # 미리 기본값 설정
            preview_url = "https://naver.com"
            option_raw = ""
            option_display = "없음"
            
            # preview_url 생성
            try:
                html_raw = row.get("description", "") or ""
                html_cleaned = clean_html_content(html_raw)
                if isinstance(html_raw, bytes):
                    html_raw = html_raw.decode("cp949")
                encoded_html = base64.b64encode(html_cleaned.encode("utf-8", errors="ignore")).decode("utf-8")
                preview_url = f"{API_URL}/preview?html={urllib.parse.quote_plus(encoded_html)}"
            except Exception:
                preview_url = "https://naver.com"

            # 옵션 파싱
            try:
                option_raw = str(row.get("composite_options", "")).strip()
                if option_raw.lower() not in ["", "nan", "none", "없음"]:
                    parsed = []
                    for line in option_raw.splitlines():
                        try:
                            parts = line.split(",")
                            if len(parts) >= 2:
                                name = parts[0].strip()
                                extra = int(float(parts[1]))
                                parsed.append(f"{name}{f' (＋{extra:,}원)' if extra>0 else ''}")
                        except Exception:
                            parsed.append(line.strip())
                    option_display = "\n".join(parsed) if parsed else "없음"
            except Exception as err:
                print(f"⚠️ 옵션 파싱 오류: {err}")

            return {
                "상품코드":     str(row.get("product_code", "없음")),
                "제목":        row.get("market_product_name", "제목 없음"),
                "가격":        convert_to_serializable(row.get("market_price", 0)),
                "배송비":      convert_to_serializable(row.get("shipping_fee", 0)),
                "이미지":      row.get("image_url", "이미지 없음"),
                "원산지":      row.get("origin", "정보 없음"),
                "상품링크":    preview_url,
                "옵션":        option_display,
                "조합형옵션":  option_raw,
                "최대구매수량": convert_to_serializable(row.get("max_quantity", 0)),
                "카테고리":    row.get("category_name", "카테고리 없음"),
                "검색방식":    "직접검색",
            }

        def _build_info_from_hit(hit):
            e = hit.entity
            
            # 미리 기본값 설정
            preview_url = "https://naver.com"
            option_raw = ""
            option_display = "없음"
            
            # preview_url 생성
            try:
                html_raw = e.get("description", "") or ""
                html_cleaned = clean_html_content(html_raw)
                if isinstance(html_raw, bytes):
                    html_raw = html_raw.decode("cp949")
                encoded_html = base64.b64encode(html_cleaned.encode("utf-8", errors="ignore")).decode("utf-8")
                preview_url = f"{API_URL}/preview?html={urllib.parse.quote_plus(encoded_html)}"
            except Exception:
                preview_url = "https://naver.com"

            # 옵션 파싱
            try:
                option_raw = str(e.get("composite_options", "")).strip()
                if option_raw.lower() not in ["", "nan", "none", "없음"]:
                    parsed = []
                    for line in option_raw.splitlines():
                        try:
                            parts = line.split(",")
                            if len(parts) >= 2:
                                name = parts[0].strip()
                                extra = int(float(parts[1]))
                                parsed.append(f"{name}{f' (＋{extra:,}원)' if extra>0 else ''}")
                        except Exception:
                            parsed.append(line.strip())
                    option_display = "\n".join(parsed) if parsed else "없음"
            except Exception as err:
                print(f"⚠️ 옵션 파싱 오류: {err}")

            return {
                "상품코드":     str(e.get("product_code", "없음")),
                "제목":        e.get("market_product_name", "제목 없음"),
                "가격":        convert_to_serializable(e.get("market_price", 0)),
                "배송비":      convert_to_serializable(e.get("shipping_fee", 0)),
                "이미지":      e.get("image_url", "이미지 없음"),
                "원산지":      e.get("origin", "정보 없음"),
                "상품링크":    preview_url,
                "옵션":        option_display,
                "조합형옵션":  option_raw,
                "최대구매수량": convert_to_serializable(e.get("max_quantity", 0)),
                "카테고리":    e.get("category_name", "카테고리 없음"),
                "검색방식":    "벡터검색",
            }

        def _interleave_unique(a, b, top_n=None):
            out, seen = [], set()
            ai = bi = 0
            N = top_n or (len(a) + len(b))
            while (ai < len(a) or bi < len(b)) and len(out) < N:
                if ai < len(a):
                    if a[ai]["상품코드"] not in seen:
                        out.append(a[ai]); seen.add(a[ai]["상품코드"])
                    ai += 1
                if bi < len(b) and len(out) < N:
                    if b[bi]["상품코드"] not in seen:
                        out.append(b[bi]); seen.add(b[bi]["상품코드"])
                    bi += 1
            return out

        def season_filter_items(items: list, season_hint: str):
            """시즌에 맞지 않는 후보 제거(강 키워드 기준) — 기존 시즌 키워드 그대로 활용"""
            if season_hint not in ("봄","여름","가을","겨울"):
                return items

            if season_hint == "여름":
                neg = list(WINTER_KEYWORDS) + ["방울","털","비니","니트","퍼","기모","방한"]
            elif season_hint == "겨울":
                neg = list(SUMMER_KEYWORDS) + ["라피아","스트로","스트로우","썬캡","쿨링","냉감","메쉬"]
            elif season_hint in ("봄","가을"):
                neg = list(SUMMER_KEYWORDS) + list(WINTER_KEYWORDS)
            else:
                neg = tuple()

            kept, removed = [], 0
            for it in items:
                name = f"{it.get('제목','')} {it.get('카테고리','')}"
                if _has_any(name, neg):
                    removed += 1
                    continue
                kept.append(it)
            print(f"[Top2-Stage] 시즌('{season_hint}') 필터: 제거 {removed}, 유지 {len(kept)}")
            return kept

        def compute_top2_quota(cat_counts: Counter, total: int = 10, min_each: int = 1):
            """카테고리 상위 2개를 뽑아 10개 배분(반올림, 최소1 보장)"""
            top2 = [c for c, _ in cat_counts.most_common(2)]
            if len(top2) < 2:
                only = top2[0] if top2 else "기타"
                return {only: total}, top2

            c1, c2 = cat_counts[top2[0]], cat_counts[top2[1]]
            s = max(c1 + c2, 1)
            q1 = int(round(total * (c1 / s)))
            q2 = total - q1
            # 최소 보장
            if q1 < min_each: q1, q2 = min_each, total - min_each
            if q2 < min_each: q2, q1 = min_each, total - min_each
            # 합 보정
            if q1 + q2 != total:
                if q1 + q2 > total:
                    if q1 >= q2: q1 -= (q1 + q2 - total)
                    else:        q2 -= (q1 + q2 - total)
                else:
                    if q1 >= q2: q1 += (total - (q1 + q2))
                    else:        q2 += (total - (q1 + q2))
            return {top2[0]: q1, top2[1]: q2}, top2

        def _format_for_llm(cat_name: str, items: list, max_items=140):
            lines = []
            for it in items[:max_items]:
                lines.append(f'- 코드:{it["상품코드"]} | 제목:{it["제목"]} | 가격:{it["가격"]} | 카테고리:{it["카테고리"]}')
            return "\n".join(lines)

        def llm_rank_codes_for_category(cat_name: str, items: list, user_query: str, season_hint: str, top_k: int) -> list:
            """카테고리 내부 재랭킹 → 상위 top_k개 '상품코드'만 JSON 배열로 반환 (설명 금지)"""
            if not items or top_k <= 0:
                return []
            forbid = []; prefer = []
            if season_hint == "여름":
                forbid = list(WINTER_KEYWORDS) + ["방울","털","비니","니트","퍼","기모","방한"]
                prefer = list(SUMMER_KEYWORDS)
            elif season_hint == "겨울":
                forbid = list(SUMMER_KEYWORDS) + ["라피아","스트로","스트로우","썬캡","쿨링","냉감","메쉬"]
                prefer = list(WINTER_KEYWORDS)
            elif season_hint in ("봄","가을"):
                forbid = list(SUMMER_KEYWORDS) + list(WINTER_KEYWORDS)

            prompt = f"""
        당신은 쇼핑 검색 랭킹 전문가입니다. 아래 후보 중에서 카테고리 내부 상위 {top_k}개 '상품코드'만 고르세요.

        하드 규칙:
        - 금지어: {", ".join(forbid) if forbid else "없음"} 가 제목/카테고리에 있으면 제외
        - 선호어: {", ".join(prefer) if prefer else "없음"} 가 있으면 가산점
        - 중복 제목은 낮게, 가격/옵션/일반성도 고려. (금지어 제외가 최우선)
        - **설명 금지**. JSON 배열로만.

        사용자 질의: "{user_query}"
        카테고리: "{cat_name}"
        후보:
        {_format_for_llm(cat_name, items, max_items=140)}

        출력 예시: ["P1","P2","P3"]
        """
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role":"system","content":prompt}],
                temperature=0
            )
            txt = resp.choices[0].message.content.strip()
            m = re.search(r'\[\s*".+?"\s*(?:,\s*".+?"\s*)*\]', txt, re.S)
            codes = []
            if m:
                try:
                    codes = json.loads(m.group(0))
                except Exception:
                    codes = []
            # 부족분 보충 (풀 순서대로)
            if len(codes) < top_k:
                seen = set(codes)
                for it in items:
                    c = it["상품코드"]
                    if c not in seen:
                        codes.append(c); seen.add(c)
                        if len(codes) >= top_k: break
            return codes[:top_k]







        # 1) 전 카테고리 200 후보 생성 (직접100 + 벡터100)
        print("\n[Top2-Stage] 200 후보 생성 (직접100 + 벡터100)")
        tokens = [t for t in re.sub(r"[용\s]+", " ", preprocessed_query).split() if t]
        name_expr = " && ".join(f'market_product_name like "%{tok}%"' for tok in tokens) or 'market_product_name like "%%"'
        expr_parts = [name_expr]
        if price_cond: expr_parts.append(price_cond)
        direct_expr_any = " && ".join(expr_parts)

        direct_hits_100 = collection.query(
            expr=direct_expr_any,
            limit=100,
            output_fields=[
                "product_code","category_code","category_name","market_product_name",
                "market_price","shipping_fee","shipping_type","max_quantity",
                "composite_options","image_url","manufacturer","model_name",
                "origin","keywords","description","return_shipping_fee",
            ]
        )
        direct_items_50 = [_build_info_from_row(r) for r in direct_hits_100]
        print(f"[Top2-Stage] 직접50개: {len(direct_items_50)}")

        vector_hits_100 = collection.search(
            data=[q_vec],
            anns_field="emb",
            param={"metric_type":"L2","params":{"nprobe":64}},
            limit=100,
            output_fields=[
                "product_code","category_code","category_name","market_product_name",
                "market_price","shipping_fee","shipping_type","max_quantity",
                "composite_options","image_url","manufacturer","model_name",
                "origin","keywords","description","return_shipping_fee",
            ]
        )
        vector_items_50 = []
        for hits in vector_hits_100:
            for h in hits:
                vector_items_50.append(_build_info_from_hit(h))
        print(f"[Top2-Stage] 벡터50개: {len(vector_items_50)}")

        # 기존 변수명 재사용: raw_candidates (100 인터리브 결과)
        raw_candidates = _interleave_unique(direct_items_50, vector_items_50, top_n=100)
        print(f"[Top2-Stage] interleave 후 후보수(raw_candidates): {len(raw_candidates)}")

        # 2) 시즌 필터 적용
        raw_candidates = season_filter_items(raw_candidates, season)

        # 3) 카테고리별 상품 개수 계산 및 Top2 선정
        cat_counts = Counter(p["카테고리"] for p in raw_candidates)
        print("\n[Top2-Stage] 카테고리별 상품 개수:")
        for cat, count in cat_counts.most_common():
            print(f"  {cat}: {count}개")

        # Top2 카테고리 선정
        top2_cats = [cat for cat, _ in cat_counts.most_common(2)]
        if not top2_cats:
            print("⚠️ 카테고리가 없습니다.")
            top2_cats = ["기타"]

        # 선정된 Top2 카테고리의 비율 계산 및 10개 할당
        c1, c2 = top2_cats[0], top2_cats[1] if len(top2_cats) > 1 else None
        count1, count2 = cat_counts[c1], cat_counts[c2] if c2 else 0
        total = count1 + count2

        # 비율에 따른 10개 분배
        if c2 is None:
            quota10 = {c1: 10}
        else:
            q1 = round(10 * (count1 / total))
            q2 = 10 - q1
            # 최소 1개 보장
            if q1 == 0: q1, q2 = 1, 9
            if q2 == 0: q1, q2 = 9, 1
            quota10 = {c1: q1, c2: q2}

        print(f"\n[Top2-Stage] 선정된 Top2 카테고리 (총 {total}개 중):")
        print(f"  1위: {c1} - {count1}개 → {quota10[c1]}개 할당")
        if c2:
            print(f"  2위: {c2} - {count2}개 → {quota10[c2]}개 할당")

        # 4) Top2 풀 구성
        top2_pools = {c: [] for c in top2_cats}
        for it in raw_candidates:
            for c in top2_cats:
                if it["카테고리"].startswith(c):
                    top2_pools[c].append(it)
                    break
        print("\n[Top2-Stage] 카테고리별 풀 크기:", {k: len(v) for k,v in top2_pools.items()})

        # 5) 카테고리별 LLM 재랭킹 → ranked_pools (카테고리 내부 정렬)
        prompt_template = """
        답변은 반드시 "{target_lang}"로 해주세요.
        당신은 고객의 니즈를 정확히 파악하는 프리미엄 쇼핑몰의 VIP 상품 추천 전문가입니다.
        주어진 상품 목록을 바탕으로, 고객이 더 나은 선택을 할 수 있도록 도와주세요.

        [매우 중요]
        - 고객 검색어("{query}")의 의미를 임의로 확장/추정/대체하지 마세요.
        - 아래 '카테고리'는 후보이며, 사실 여부를 먼저 확인하는 질문부터 하세요.
        - 특정 품목/모델로 단정하지 말고, 후보 리스트의 특징을 근거로 질문하세요.

        현재 상황:
        - 고객 검색어: "{query}"
        - 카테고리(후보): "{category}"
        - 현재 시즌: {season}

        후보 상품 목록:
        {products}

        요청사항:
        1) 후보 상품의 제목/설명에서 서로를 구분해주는 특징 키워드 3~5개를 추출하세요.
        - 기능/성능: 방수, 저소음, 고출력, 대용량, 저전력, 무선/유선, 규격(예: 27W, 128GB, 1.5L 등)
        - 재질/마감: 가죽/스테인리스/TPU/친환경 등
        - 구성/형태: 세트/단품, 접이식, 휴대용, 벽걸이, 2+1 구성 등
        - 호환/범용/시즌: 정품/호환, 규격·사이즈, 여름/겨울 등
        2) 그 특징을 활용해 고객 의도 확인과 선호 파악을 위한 '확인형 질문'을 200~250자 내외 한 문단으로 작성하세요.
        - 예시 어조: "후보를 보니 [특징A/특징B/특징C] 옵션이 보이는데, 이런 조건이 필요하신가요?"
        - 본품인지 관련품·소모품·세트·서비스인지도 자연스럽게 확인하세요.
        - 가격대(프리미엄/일반/실속), 디자인, 용도, 재질, 브랜드 선호, 계절감, 옵션 유무를 과도한 나열 없이 자연스럽게 묻으세요.
        3) 상품 코드나 구체 모델명/스펙 나열은 금지합니다. 후보에서 추출한 '특징 키워드'만 요약해 언급하세요.
        4) 친근하면서도 전문적인 대화체를 유지하세요.
        5) 아래 예시는 참고만 하며, 그대로 복사하지 말고 실제 후보의 특징으로 대체하세요.

        답변 형식(단락 예시):
        고객님께서 "{query}"를 찾고 계시다고 이해했습니다. 후보를 보니 [특징A/특징B/특징C] 같은 선택지가 보입니다. 이러한 조건이 필요하신가요, 아니면 심플한 구성이 더 좋으실까요? 사용하실 환경과 선호 가격대(프리미엄/실속), 디자인·재질, 그리고 본품/관련품 중 어떤 쪽을 찾으시는지도 알려주시면 더 정확히 추천해 드리겠습니다.

        """

        ranked_pools = {}
        clean = ""
        for cat in top2_cats:
            pool = top2_pools.get(cat, [])
            k = min(len(pool), 120)  # 안전상 120까지만 LLM에 노출
            
            # 상품 정보 포맷팅
            products_text = "\n".join([
                f"- 코드: {p['상품코드']} | 제목: {p['제목']} | 가격: {p['가격']:,}원 | 카테고리: {p['카테고리']}"
                for p in pool[:k]
            ])
            
            # LLM 프롬프트 구성
            prompt = prompt_template.format(
                target_lang=target_lang,
                query=preprocessed_query,
                category=cat,
                season=season,
                products=products_text
            )
            
            # LLM 호출
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": prompt}],
                temperature=0
            )
            txt1 = response.choices[0].message.content or ""
            clean = txt1.strip()
            print(f"[Top2-Stage] {cat} 추가 질문:\n{clean}\n")
            
            try:
                # JSON 배열 추출
                txt = response.choices[0].message.content.strip()
                m = re.search(r'\[\s*"[^"]+?"(?:\s*,\s*"[^"]+?")*\s*\]', txt)
                if m:
                    ranked_codes = json.loads(m.group(0))
                else:
                    ranked_codes = []
            except Exception as e:
                print(f"[Top2-Stage] LLM 응답 파싱 오류 ({cat}): {e}")
                ranked_codes = []
            
            # 순위가 매겨진 상품 목록 생성
            code2item = {p["상품코드"]: p for p in pool}
            ranked = [code2item[c] for c in ranked_codes if c in code2item]
            seen = set(p["상품코드"] for p in ranked)
            
            # 순위가 매겨지지 않은 상품들 추가
            for p in pool:
                if p["상품코드"] not in seen:
                    ranked.append(p)
                    seen.add(p["상품코드"])
            
            ranked_pools[cat] = ranked
            print(f"[Top2-Stage] {cat} 재랭킹 완료: {len(ranked)}개")
        print("[Top2-Stage] 모든 카테고리 재랭킹 완료:", {k: len(v) for k,v in ranked_pools.items()})

        # 6) 버튼(페이지) 로더 — 페이지마다 quota10(예: 7:3 등) 유지하여 10개씩 공급
        def get_next_products_top2(ranked_pools: Dict[str, List[Dict]], quotas: Dict[str, int], top2_order: List[str], offset: int = 0, batch_size: int = 10):
            """
            Top2 카테고리에서 다음 페이지 상품들을 가져오는 함수
            
            Args:
                ranked_pools: 카테고리별 전체 상품 풀 (LLM 랭킹 적용된 상태)
                quotas: 카테고리별 할당량 (예: {"카테고리A": 7, "카테고리B": 3})
                top2_order: 카테고리 순서 리스트 (예: ["카테고리A", "카테고리B"])
                offset: 현재까지 로드된 상품 수
                batch_size: 한 번에 가져올 상품 수 (기본값 10)
            
            Returns:
                (현재 페이지 상품들, 더 있는지 여부)
            """
            assert len(top2_order) >= 1, "Top2 카테고리가 비어있음"
            
            # 각 카테고리별 시작 위치 계산
            total_products = {cat: len(pool) for cat, pool in ranked_pools.items()}
            used_products = {cat: offset * (quotas.get(cat, 0) / batch_size) 
                            for cat in top2_order}
            
            out = []  # 이번 페이지에 보여줄 상품들
            has_more = False  # 더 보여줄 상품이 있는지 여부
            
            # 1) 각 카테고리별로 할당량만큼 가져오기
            remaining_quotas = quotas.copy()  # 이번 배치의 남은 할당량
            
            for cat in top2_order:
                if not remaining_quotas.get(cat, 0):  # 할당량이 없으면 스킵
                    continue
                    
                pool = ranked_pools.get(cat, [])
                start_idx = int(used_products.get(cat, 0))
                quota = remaining_quotas[cat]
                
                # 이 카테고리에서 가져올 수 있는 만큼 가져오기
                available = pool[start_idx:start_idx + quota]
                out.extend(available)
                
                # 남은 할당량 갱신
                remaining_quotas[cat] -= len(available)
                
                # 더 가져올 수 있는지 체크
                if start_idx + quota < len(pool):
                    has_more = True
            
            # 2) 부족분 보충 - 다른 카테고리에서 채우기
            remaining = batch_size - len(out)
            if remaining > 0:
                for cat in top2_order:
                    if remaining <= 0:
                        break
                        
                    pool = ranked_pools.get(cat, [])
                    start_idx = int(used_products.get(cat, 0)) + quotas.get(cat, 0)
                    
                    # 남은 상품이 있는지 확인
                    if start_idx < len(pool):
                        available = pool[start_idx:start_idx + remaining]
                        out.extend(available)
                        remaining -= len(available)
                        if len(available) > 0:
                            has_more = True
            
            # 3) 모든 카테고리가 소진됐는지 체크
            if not has_more:
                for cat in top2_order:
                    start_idx = int(used_products.get(cat, 0)) + quotas.get(cat, 0)
                    if start_idx < total_products.get(cat, 0):
                        has_more = True
                        break
            
            return out[:batch_size], has_more

        # ===== 방법 2의 페이지 로딩 테스트 =====
        print("\n📖 방법 2의 상품리스트 테스트:")
        print(f"- 카테고리 할당: {quota10}")  # 예: {"카테고리A": 7, "카테고리B": 3}
        print(f"- 각 풀 크기: {[(k, len(v)) for k,v in ranked_pools.items()]}")

        # 페이지 로딩 시뮬레이션
        offset = 0  # 시작 위치
        page = 1    # 페이지 번호 (표시용)
        method2_shown_products = []     # 지금까지 보여준 모든 상품
        method2_seen_keys = set()  # 중복 체크용 키 세트

        for _ in range(10):  # 최대 10번 반복
            # 다음 페이지 가져오기
            batch, more = get_next_products_top2(ranked_pools, quota10, top2_cats, offset, batch_size=10)
            
            # 배치가 비어있으면 종료
            if not batch:
                if offset == 0:
                    print("\n❌ 표시할 상품이 없습니다.")
                else:
                    print("\n✅ 더 불러올 상품이 없습니다.")
                    print(f"   총 {len(method2_shown_products)}개 노출 완료.")
                break

            # 중복 방지(코드+카테고리 단위)
            new_batch = []
            for p in batch:
                key = f"{p['상품코드']}_{p['카테고리']}"
                if key not in method2_seen_keys:
                    method2_seen_keys.add(key)
                    new_batch.append(p)
                    method2_shown_products.append(p)

            # 새로운 상품이 없으면 종료
            if not new_batch:
                print("\n✅ 모든 상품을 이미 표시했습니다.")
                print(f"   총 {len(method2_shown_products)}개 노출 완료.")
                break

            # 새로운 상품 출력
            print(f"\n=== 페이지 {page} (누적 {len(method2_shown_products)}개, offset={offset}) ===")
            for i, p in enumerate(new_batch, 1):
                print(f"[{offset+i:>3}] {p['제목']} | {p['카테고리']} | {p['가격']:,}원 | 코드:{p['상품코드']}")

            # 더 이상 가져올 상품이 없으면 종료
            if not more:
                print("\n✅ 더 불러올 상품이 없습니다.")
                print(f"   총 {len(method2_shown_products)}개 노출 완료.")
                break

            offset += len(new_batch)  # 실제로 추가된 새로운 상품 수만큼만 증가
            page += 1

        final_results = []
        method1_offset = 0
        method2_offset = 0

        print("\n✅ 최종 추천 상품 (방법1과 방법2 번갈아가며):")

        for batch_num in range(10):  # 최대 10개 배치 반복
            # 방법2에서 10개 가져오기
            start_idx = method2_offset
            end_idx = start_idx + 10
            products2 = method2_shown_products[start_idx:end_idx]
            
            # 방법1에서 10개 가져오기
            products1, has_more1 = get_next_products(top3_products, top5_cats, method1_offset)

            # 먼저 방법2의 10개를 추가
            if products2:
                final_results.extend(products2)
                method2_offset = end_idx
                print(f"\n[배치 {batch_num+1}] 방법2: {len(products2)}개 상품 추가 (누적: {method2_offset}개)")
                
            # 그 다음 방법1의 10개를 추가
            if products1:
                # 중복 제거를 위한 임시 세트
                existing_codes = set(p["상품코드"] for p in final_results)
                unique_products1 = [p for p in products1 if p["상품코드"] not in existing_codes]
                
                final_results.extend(unique_products1)
                method1_offset += len(products1)  # 원래 길이만큼 오프셋 증가
                print(f"[배치 {batch_num+1}] 방법1: {len(unique_products1)}개 상품 추가 (누적: {method1_offset}개)")
            
            # 두 방법 모두 더 이상 상품이 없으면 종료
            has_more2 = method2_offset < len(method2_shown_products)
            if not has_more1 and not has_more2:
                print("\n✅ 모든 상품이 추가되었습니다.")
                print(f"최종 결과: 총 {len(final_results)}개 상품")
                break

        # 최종 결과 출력
        print(f"\n총 {len(final_results)}개의 상품이 최종 리스트에 저장되었습니다.")
        print("각 방식 별 상품 수:")
        method1_count = len([p for p in final_results if p in method1_shown_products])
        method2_count = len([p for p in final_results if p not in method1_shown_products])
        print(f"- 방법1 (5:3:2): {method1_count}개")
        print(f"- 방법2 (Top2): {method2_count}개")

        print("\n상품의 상세 정보:")
        for idx, info in enumerate(final_results[:40], start=1):
            PRODUCT_CACHE[info["상품코드"]] = info
            
            print(f"\n[{idx}] {info['제목']}")
            print(f"   카테고리   : {info['카테고리']}")
            # print(f"   상품코드   : {info['상품코드']}")
            print(f"   가격       : {info['가격']}원")
            # print(f"   배송비     : {info['배송비']}원")
            # print(f"   이미지     : {info['이미지']}")
            # print(f"   원산지     : {info['원산지']}")
            # print(f"   상품링크   : {info['상품링크']}")
            # print(f"   옵션       : {info['옵션']}")
            # print(f"   조합형옵션 : {info['조합형옵션']}")
            print(f"   최대구매수량: {info['최대구매수량']}개")
            
        print(f"================================")
        print(f"보여주는 상품의 개수: {len(final_results)}")

        # ✅ 상품 정보에 옵션 관련 필드 확인 및 추가
        for product in final_results:
            if "옵션" not in product:
                product["옵션"] = "없음"
            if "조합형옵션" not in product:
                product["조합형옵션"] = ""
            if "최대구매수량" not in product:
                product["최대구매수량"] = 0       

        result_payload = {
            "query": query,  # 사용자가 입력한 원본 쿼리
            "UserMessage": llm_response,  # 정제된 쿼리
            "RawContext": previous_queries + [query],  # 전체 대화 맥락
            "results": final_results,  # 검색 결과 리스트 (방법 2가 앞쪽, 방법 1이 뒤쪽)
            "combined_message_text": clean,  # 사용자에게 보여줄 메시지
            "message_history": [
                {"type": type(msg).__name__, "content": getattr(msg, "content", "")}
                for msg in session_history.messages
            ]  # 전체 메시지 기록 (디버깅용)
        }
        return result_payload
    
    except Exception as e:
        print(f"❌ external_search_and_generate_response 오류: {e}")
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
        print(f"\n🐞 [DEBUG] option_raw: {option_raw}\n")

        option_display = "없음"
        if option_raw and option_raw.lower() != "nan":
            option_lines = option_raw.splitlines()
            print(f"\n🐞 [DEBUG] option_lines: {option_lines}\n")
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
        updater.set_unique_code("13117409", product.get('상품코드'))
        updater.set_product_name("13117396", product.get('제목'))
        updater.set_option("12953235", option_display)
        updater.set_price("13117479", price)
        updater.set_shipping("13117482", shipping)
        updater.set_product_max_quantity("13117481", product.get('최대구매수량'))

        # ✅ 외부 Flow 트리거 (비동기처럼 요청 보내기)
        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        flow_payload = {
            "subscriber_id": sender_id,
            "flow_ns": "content20250604080355_172315"
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
            "flow_ns": "content20250605003906_502539"
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
                "url": f"{API_URL}/manychat-option-select",
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
                    "url": f"{API_URL}/manychat-option-request",
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
    updater.set_product_selection_option("13117397", selected_option)
    updater.set_extra_price("13117480", extra_price)

    # ✅ 옵션 저장 후 Flow로 이동시키기
    headers = {
        "Authorization": f"Bearer {MANYCHAT_API_KEY}",
        "Content-Type": "application/json"
    }
    flow_payload = {
        "subscriber_id": sender_id,
        "flow_ns": "content20250605003906_502539"
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
                    "text": f"Option selected: {selected_option} (Extra: {extra_price:,})원)"
                }
            ]
        }
    }

class QuantityInput(BaseModel):
    sender_id: str
    product_quantity: int
    product_code: str


def safe_int(val):
    try:
        return int(float(str(val).replace(",", "").replace("원", "").strip()))
    except:
        return 0


@app.post("/calculate_payment")
def calculate_payment(data: QuantityInput):
    try:
        # 1) product_code로 바로 조회
        product = PRODUCT_CACHE.get(data.product_code)
        if not product:
            raise ValueError(f"❌ 상품코드 {data.product_code} 정보가 없습니다.")

        sender_id = data.sender_id
        quantity = data.product_quantity
        if not sender_id:
            raise ValueError("❌ sender_id 누락됨")

        # 2) 기본 정보 추출
        price        = safe_int(product.get("가격", 0))
        extra_price  = safe_int(product.get("추가금액", 0)) if "추가금액" in product else 0
        shipping     = safe_int(product.get("배송비", 0))
        max_quantity = safe_int(product.get("최대구매수량", 0))

        # 3) 총 가격 계산
        total_price = (price + extra_price) * quantity
        if max_quantity == 0:
            shipping_cost = shipping
        else:
            shipping_cost = shipping * math.ceil(quantity / max_quantity)
        total_price += shipping_cost

        # ✅ 천 단위 구분을 위한 포맷팅
        formatted_total_price = "{:,}".format(total_price)
        print(
            f"✅ 계산 완료 → 총금액: {formatted_total_price}원\n"
            f" 상품금액: {price:,}원,\n"
            f" 추가금액: {extra_price:,}원,\n"
            f" 수량: {quantity},\n"
            f" 배송비: {shipping_cost:,}원,\n"
            f" 묶음배송수량: {max_quantity}"
        )

        # ✅ Manychat 필드 업데이트
        updater = ManychatFieldUpdater(sender_id, MANYCHAT_API_KEY)
        updater.set_quantity("13117398", quantity)  # Product_quantity 필드 ID
        updater.set_total_price("13170342", formatted_total_price)  # Total_price 필드 ID - 포맷팅된 값으로 저장

        # ✅ ManyChat 다음 Flow로 이동
        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        flow_payload = {
            "subscriber_id": sender_id,
            "flow_ns": "content20250605012240_150101"
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



# ✅ FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run("BeeMall_Chatbot:app", host="0.0.0.0", port=8011, reload=True)
