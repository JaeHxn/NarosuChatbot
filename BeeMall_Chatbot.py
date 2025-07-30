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

# “카테고리” 목록 로드 (엑셀/CSV)
# CSV_PATH     = "카테고리목록.csv"     # '카테고리목록' 컬럼이 있는 CSV
# df_categories = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
# categories    = df_categories['카테고리목록'].dropna().unique().tolist()

collection_cat = Collection("category_0710")
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
    if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
        return obj.item()
    return obj


def minimal_clean_with_llm(latest_input: str, previous_inputs: List[str]) -> str:
    """
    최신 입력과 Redis에서 가져온 과거 입력을 함께 LLM에게 전달하여,
    최소한의 정제 + 충돌 문맥 제거를 수행한 한 문장 반환
    """
    try:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("❌ [ERROR] OPENAI_API_KEY가 설정되지 않았습니다.")
        API_KEY = os.environ["OPENAI_API_KEY"]

        llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", openai_api_key=API_KEY)

        context_message = "\n".join(previous_inputs)

        system_prompt = (f"""

            System:
        당신은 (1) 검색 엔진의 전처리를 담당하는 AI이자, (2) 쇼핑몰 검색 및 분류 전문가입니다.
        어떤 언어로 입력이 되든 반드시 한국어로 문장 의미에 맞게 번역 먼저 합니다.
        아래는 DB에서 로드된 **가능한 카테고리 목록**입니다.  
        모든 예측은 이 목록 안에서만 이루어져야 합니다:
        
        {categories}
        
        다음 순서대로 응답하세요:
        
        1) **전처리 단계**  
           - 사용자 원문(query)에서 오타를 바로잡고, 중복 표현을 제거한 뒤  
           - 핵심 키워드와 의미만 남긴 깔끔한 검색 쿼리로 바꿔주세요.  
           - 문장의 의미가 맞다면 문장 통으로 입력되어도 괜찮습니다.  
        
        2) **카테고리 예측 단계**  
           - 전처리된 쿼리를 바탕으로 직관적으로 최상위 카테고리 하나를 예측하세요.
        
        3) **검색 결과 재정렬 단계**  
           - 이미 Milvus 벡터 검색을 통해 얻은 TOP N 결과 리스트(search_results)를 입력받아  
           - 각 결과의 메타데이터(id, 상품명, 카테고리, 가격, URL 등)를 활용해  
           - 2번에서 예측한 카테고리와 매칭되거나 인접한 결과를 우선 정렬하세요.
        
        4) **출력 형식**은 반드시 아래와 같습니다:
        
        Raw Query: "<query>"  
        Preprocessed Query: "<전처리된_쿼리>"  
        Predicted Category: "<예측된_최상위_카테고리>" 

            """
        )

        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"이전 대화: {context_message}\n최신 입력: {latest_input}")
        ]) 

        if not hasattr(response, "content") or not isinstance(response.content, str):
            raise ValueError("❌ LLM 응답이 유효하지 않습니다.")

        return response.content.strip()

    except Exception as e: 
        print(f"❌ [ERROR] minimal_clean_with_llm 실패: {e}")
        return latest_input  # 실패 시 최신 입력만 사용


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


def compute_top4_quota(
    candidates: List[Dict[str, Any]],
    max_total: int = max_total,
    min_per_category: int = 1
) -> Dict[str, int]:
    """
    Top4 카테고리 자동 추출 & 비율 기반 quota 계산
    Returns: {카테고리: quota}
    """
    total = len(candidates)
    counts = Counter(item["카테고리"] for item in candidates)
    top4 = [cat for cat, _ in counts.most_common(4)]
    
    # 초기 quota 계산 (floor + 최소 보장)
    quotas = {
        cat: max(math.floor(counts[cat] / total * max_total), min_per_category)
        for cat in top4
    }
    
    # 부족·초과 보정
    diff = max_total - sum(quotas.values())
    if diff > 0:
        # 비중 큰 순서대로 +1
        for cat, _ in counts.most_common():
            if cat in quotas and diff > 0:
                quotas[cat] += 1
                diff -= 1
    elif diff < 0:
        # 비중 작은 순서대로 -1 (min 유지)
        for cat, _ in reversed(counts.most_common()):
            if cat in quotas and quotas[cat] > min_per_category and diff < 0:
                quotas[cat] -= 1
                diff += 1
    
    return quotas

def filter_top4_candidates(
    candidates: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Top4 카테고리만 필터링
    Returns: (filtered_candidates, top4_keys)
    """
    counts = Counter(item["카테고리"] for item in candidates)
    top4_keys = [cat for cat, _ in counts.most_common(4)]
    filtered = [item for item in candidates if item["카테고리"] in top4_keys]
    return filtered, top4_keys

def prepare_recommendation(
    all_candidates: List[Dict[str, Any]],
    max_total: int = max_total,
    min_per_category: int = 1
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    """
    1) Top4 필터링
    2) quota 계산
    Returns: (filtered_candidates, quotas, top4_keys)
    """
    filtered, top4_keys = filter_top4_candidates(all_candidates)
    quotas = compute_top4_quota(filtered, max_total, min_per_category)
    return filtered, quotas, top4_keys

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

        # ✅ LLM으로 정제된 쿼리 생성
        UserMessage = minimal_clean_with_llm(query, previous_queries)
        print("\n🧾 [최종 정제된 문장] →", UserMessage)
        print("📚 [원본 전체 문맥] →", " | ".join(previous_queries + [query]))
        
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
        
        target_lang = lang_map.get(lang_code, "English")
        
        print("[Debug] Detected language →", target_lang)

        llm_response = UserMessage
        print("[Debug] LLM full response:\n", llm_response)  # ← 여기에!   
        
        #LLM 응답 파싱
        lines = [l.strip() for l in llm_response.splitlines() if l.strip()]
        preprocessed_query = next(
            l.split(":",1)[1].strip().strip('"')
            for l in lines if l.lower().startswith("preprocessed query")
        )
        predicted_category = next(
            l.split(":",1)[1].strip().strip('"')
            for l in lines if l.lower().startswith("predicted category")
        )
        # ← 여기에 한 줄 추가
        top_category = predicted_category.split(">")[0]
        
        print("[Debug] Preprocessed Query →", preprocessed_query)
        print("[Debug] top_category →", top_category)


        #최하위 카테고리
        lowest_subcategory = predicted_category.split(">")[-1]
        
        print("[Debug] lowest_subcategory →", lowest_subcategory)
        
        #쿼리 임베딩 생성
        q_vec = embedder.embed_query(preprocessed_query)
        print(f"[Debug] q_vec length: {len(q_vec)}, sample: {q_vec[:5]}")
        
        # ① Stage1: 직접 문자열 검색 (boolean search)
        print("[Stage1] Direct name search 시작")
        
        # “남자용 향수” → ["남자", "향수"] 두 토큰으로 AND 검색
        tokens = [t for t in re.sub(r"[용\s]+", " ", preprocessed_query).split() if t]
        query_expr = " && ".join(f'market_product_name like "%{tok}%"'
            for tok in tokens
        )
        
        print("[Debug] Stage1 expr:", query_expr)
        direct_hits = collection.query(
            expr=query_expr,
            limit=200,
            output_fields = [
            "product_code",
            "category_code",
            "category_name",
            "market_product_name",
            "market_price",
            "shipping_fee",
            "shipping_type",
            "max_quantity",
            "composite_options",
            "image_url",
            "manufacturer",
            "model_name",
            "origin",
            "keywords",
            "description",
            "return_shipping_fee",
        ]
        )
        print("[Stage1] Direct hits count:", len(direct_hits))

        for i, row in enumerate(direct_hits[:7], 1):
            print(f"  [Stage1 샘플 {i}]: 코드={row['product_code']}, 이름={row['market_product_name']}")
        
        
        print("\n[Stage2.5] 직접검색 results 구성 시작")  
        raw_candidates = []
        for row in direct_hits:
            # e = hit.entity
            # 본문 미리보기 링크
            try:
                html_raw = row.get("description", "") or ""
                html_cleaned = clean_html_content(html_raw)
                if isinstance(html_raw, bytes):
                    html_raw = html_raw.decode("cp949")
                encoded_html = base64.b64encode(
                    html_cleaned.encode("utf-8", errors="ignore")
                ).decode("utf-8")
                safe_html = urllib.parse.quote_plus(encoded_html)
                preview_url = f"{API_URL}/preview?html={safe_html}"
            except Exception as err:
                print(f"⚠️ 본문 처리 오류: {err}")
                preview_url = "https://naver.com"
    
            # 상품링크(fallback)
            product_link = row.get("product_link", "")
            if not product_link or product_link in ["링크 없음", "#", None]:
                product_link = preview_url
                                                                                            
            # 옵션 파싱
            option_raw = str(row.get("composite_options", "")).strip()
            option_display = "없음"
            if option_raw.lower() not in ["", "nan"]:
                parsed = []
                for line in option_raw.splitlines():
                    try:
                        name, extra, _ = line.split(",")
                        extra = int(float(extra))
                        parsed.append(
                            f"{name.strip()}{f' (＋{extra:,}원)' if extra>0 else ''}"
                        )
                    except Exception:
                        parsed.append(line.strip())
                option_display = "\n".join(parsed)
    
            # 10개 한글 속성으로 딕셔너리 구성
            result_info = {
                "상품코드":     str(row.get("product_code", "없음")),
                "제목":        row.get("market_product_name", "제목 없음"),
                "가격":        convert_to_serializable(row.get("market_price", 0)),
                "배송비":      convert_to_serializable(row.get("shipping_fee", 0)),
                "이미지":      row.get("image_url", "이미지 없음"),
                "원산지":      row.get("origin", "정보 없음"),
                "상품링크":    product_link,
                "옵션":        option_display,
                "조합형옵션":  option_raw,
                "최대구매수량": convert_to_serializable(row.get("max_quantity", 0)),
                "카테고리":    row.get("category_name", "카테고리 없음"),
            }
            result_info_cleaned = {}
            for k, v in result_info.items():
                if isinstance(v, str):
                    v = v.replace("\n", "").replace("\r", "").replace("\t", "")
                result_info_cleaned[k] = v
            raw_candidates.append(result_info_cleaned)

   
        # ② Stage2: 벡터 유사도 검색
        # expr = f'category_name like "%{top_category}%"'   #최상위 카테고리
        expr = f'category_name like "%{lowest_subcategory}%"'   #최하위 카테고리
        milvus_results = collection.search(
            data=[q_vec],
            anns_field="emb",  # ← 벡터 저장된 필드 이름
            param={"metric_type": "L2", "params": {"nprobe": 1536}},   #유클리드 방식 
            # param={"metric_type": "COSINE", "params": {"nprobe": 128}},   #코사인 방식
            limit=200,
            expr=expr,                              
            output_fields = [
            "product_code",
            "category_code",
            "category_name",
            "market_product_name",
            "market_price",
            "shipping_fee",
            "shipping_type",
            "max_quantity",
            "composite_options",
            "image_url",
            "manufacturer",
            "model_name",
            "origin",
            "keywords",
            "description",
            "return_shipping_fee",
        ]
        )
        print(f"[Stage2] Vector hits count: {len(milvus_results[0])}")

        #  results 생성
        print("\n[Stage2.5] 벡터 esults 구성 시작")  
        # raw_candidates = []
        for hits in milvus_results:
            for hit in hits:
                e = hit.entity
                # 본문 미리보기 링크
                try:
                    html_raw = e.get("description", "") or ""
                    html_cleaned = clean_html_content(html_raw)
                    if isinstance(html_raw, bytes):
                        html_raw = html_raw.decode("cp949")
                    encoded_html = base64.b64encode(
                        html_cleaned.encode("utf-8", errors="ignore")
                    ).decode("utf-8")
                    safe_html = urllib.parse.quote_plus(encoded_html)
                    preview_url = f"{API_URL}/preview?html={safe_html}"
                except Exception as err:
                    print(f"⚠️ 본문 처리 오류: {err}")
                    preview_url = "https://naver.com"
        
                # 상품링크(fallback)
                product_link = e.get("product_link", "")
                if not product_link or product_link in ["링크 없음", "#", None]:
                    product_link = preview_url
        
                # 옵션 파싱
                option_raw = str(e.get("composite_options", "")).strip()
                option_display = "없음"
                if option_raw.lower() not in ["", "nan"]:
                    parsed = []
                    for line in option_raw.splitlines():
                        try:
                            name, extra, _ = line.split(",")
                            extra = int(float(extra))
                            parsed.append(
                                f"{name.strip()}{f' (＋{extra:,}원)' if extra>0 else ''}"
                            )
                        except Exception:
                            parsed.append(line.strip())
                    option_display = "\n".join(parsed)
        
                # 10개 한글 속성으로 딕셔너리 구성
                result_info = {
                    "상품코드":     str(e.get("product_code", "없음")),
                    "제목":        e.get("market_product_name", "제목 없음"),
                    "가격":        convert_to_serializable(e.get("market_price", 0)),
                    "배송비":      convert_to_serializable(e.get("shipping_fee", 0)),
                    "이미지":      e.get("image_url", "이미지 없음"),
                    "원산지":      e.get("origin", "정보 없음"),
                    "상품링크":    product_link,
                    "옵션":        option_display,
                    "조합형옵션":  option_raw,
                    "최대구매수량": convert_to_serializable(e.get("max_quantity", 0)),
                    "카테고리":    e.get("category_name", "카테고리 없음"),

                }
                result_info_cleaned = {}
                for k, v in result_info.items():
                    if isinstance(v, str):
                        v = v.replace("\n", "").replace("\r", "").replace("\t", "")
                    result_info_cleaned[k] = v
                raw_candidates.append(result_info_cleaned)
        
                # 캐시에 안전 저장
                product_code = result_info_cleaned.get("상품코드")



        
        message_history = [
            {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
            for msg in session_history.messages
        ]
        
        # 완료 후: 원본 보관
        original_candidates = raw_candidates.copy()

    
            
        # 개수 및 샘플 확인
        print(f"[Stage2.5] raw_candidates count: {len(raw_candidates)}")
        
        # # ④ Stage4: LLM으로 최종 5개 선택
        # print("[Stage4] LLM 최종 후보 선정 시작")
        # candidate_list = "\n".join(
        #     f"{i+1}. {info['제목']} [{info.get('카테고리', predicted_category)}]"
        #     for i, info in enumerate(raw_candidates)
        # )


        # Top4 필터 + quota 계산
        filtered_cands, quotas, top4_keys = prepare_recommendation(
            all_candidates=original_candidates,
            max_total=max_total,
            min_per_category=1
        )
        
        print("🔝 Top4 카테고리:", top4_keys)
        print("🗂️ 카테고리별 quota:", quotas)
        
        total = len(raw_candidates)
        print(f"🔍 총 후보: {total}개")
        
        # 비율
        props = compute_category_proportions(filtered_cands)
        print("📊 카테고리별 비율:")
        for cat, ratio in props.items():
            print(f"  {cat}: {ratio*100:.1f}%")
        
            
        
        # quota (최종 5개 배정 기준 예시)
        quotas = compute_top4_quota(filtered_cands, max_total=max_total,min_per_category=1)
        print("🗂️ 카테고리별 추천 개수(quota):")
        for cat, q in quotas.items():
            print(f"  {cat}: {q}개")
        
        # Prompt에 quota 가이드 추가 ────────────────────────────────
        def quota_to_text(quota: Dict[str, int]) -> str:
            return "\n".join([f"- {cat}: {num}개" for cat, num in quota.items()])
        
        quota_text = quota_to_text(quotas)
        print(f"quota_text ->   {quota_text}")
        # 후보 리스트(원본 전부)
        
        candidate_list = "\n".join(
            f"{i+1}. {c['제목']} [{c['카테고리']}]"
            for i, c in enumerate(filtered_cands)
        )

        raw_results_json = json.dumps(candidate_list[:5], ensure_ascii=False)
        raw_history_json = json.dumps(message_history, ensure_ascii=False)
        escaped_results = raw_results_json.replace("{", "{{").replace("}", "}}")
        escaped_history = raw_history_json.replace("{", "{{").replace("}", "}}")

        
        print("[Stage4] LLM에 넘길 후보 리스트:\n", candidate_list[:300], "...")  # 앞부분만 출력
        print(f"target_lang 1번째 ----- {target_lang}")
        # ✅ LangChain 기반 프롬프트 및 LLM 실행 설정
        llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
        **답변은 반드시 "{target_lang}"로 해주세요.**
        
        User Query: "{query}"
        # 예측된 카테고리: "{predicted_category}"
        # 아래 후보들은 모두 이 카테고리에 속합니다. 
        
        후보리스트 : {candidate_list}에는 이미 Top4 카테고리만 필터링된 상품들이 포함되어 있습니다.
        
        **지침:**
        
        1. **카테고리별 필터링 적용**  
           - {quota_text}에 명시된 각 카테고리별 할당량만큼, candidate_list에서 반드시 해당 카테고리 상품을 정확히 그 개수만큼 나열하세요.  
           - 예: “패션의류>남성의류>티셔츠: 4개”라면, 후보 리스트 중 해당 카테고리 상품 4개를 출력합니다.
        
        2.  **상품 리스트 절대 출력 금지**  
           - 후보리스트나 카테고리별 항목을 **하나도** 화면에 보여주지 말고, 
           
        3. **추가 탐색 질문 생성 (200~250자)**  
           - 나열된 상품 메타데이터(상품코드, 제목, 가격, 이미지 URL 등)만 참고해서,  
             사용자가 선택 폭을 좁힐 수 있는 자연어 질문(200~250자)만 작성하세요.  
        
        4. **JSON 배열**  
           - 제시된 후보 중 **반드시 {max_total}개의** 사용자의 의도에 가장 적합한 항목의 번호만을 JSON 배열 형태로 반환하세요.  
           - 반드시 **예시와 같은 형식**으로만 출력합니다:  
            
            (예시)
            [1,2,3,4,5,6,7,8,9,10]

         
         """),
            MessagesPlaceholder(variable_name="message_history"),
            ("system", f"[검색 결과 - 내부 참고용 JSON]\n{escaped_results}"),
            ("system", f"[이전 대화 내용]\n{escaped_history}"),
            ("human", query)
        ]) 
    
        runnable = prompt | llm
        with_message_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="message_history",
        )
        
        # ✅ 응답 생성 및 시간 측정
        start_response = time.time()
        # invoke 호출 직전
        print("▶️ [LLM 호출 시작] with_message_history.invoke() 직전")
        print(f"   args = {{'input': {query!r}, 'query': {query!r}, "
              f"'predicted_category': {predicted_category!r}, 'target_lang': {target_lang!r}}}")


        print(f"target_lang 2번째 ----- {target_lang}")

        resp2 = with_message_history.invoke(
            {
              "input": query,                       # MessagesPlaceholder
              "query": query,                       # "{query}" 에 매핑
              "predicted_category": predicted_category,
              "target_lang": target_lang
            },
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"📊 [LLM 응답 시간] {time.time() - start_response:.2f}초")
        print("🤖 응답 결과:", resp2.content)
        
        session_history.add_ai_message(resp2.content)
        selection = resp2.content.strip()

        print("[Stage4] Raw LLM selection:", selection)

        
        
        # # 1) ```json … ``` 마크다운 제거
        # clean = re.sub(r'```.*?\n', '', selection).replace('```','').strip()
        # print("[Stage4] Cleaned selection:", clean)
        
        # match = re.search(r'\[(?:\s*\d+\s*,?)+\s*\]', clean)
        # if match:
        #     arr_text = match.group(0)
        #     try:
        #         chosen_idxs = json.loads(arr_text)
        #     except json.JSONDecodeError:
        #         chosen_idxs = []
        # else:
        #     chosen_idxs = []






                # JSON 배열 위치 찾기
        match = re.search(r'\[\s*(?:\d+\s*,\s*)*\d+\s*\]', selection)
        if match:
            arr_text = match.group(0)
            try:
                chosen_idxs = json.loads(arr_text)
            except json.JSONDecodeError:
                chosen_idxs = []
            start, end = match.span()
            # JSON이 앞에 나오면 뒤쪽, 아니면 앞쪽을 내러티브로
            if start == 0:
                clean = selection[end:].strip()
            else:
                clean = selection[:start].strip()
        else:
            chosen_idxs = []
            clean = selection.strip()
        
        # 결과 출력
        print("\n=== 내러티브 (자연어 질문) ===")
        print(clean)
        
        print("\n=== 선택된 인덱스 리스트 ===")
        print(chosen_idxs)












        max_n = len(filtered_cands)
        valid_idxs = [i for i in chosen_idxs if 1 <= i <= max_n]
        if len(valid_idxs) < len(chosen_idxs):
            print(f"⚠️ 잘못된 인덱스 제거됨: {set(chosen_idxs) - set(valid_idxs)}")
        if not valid_idxs:
            print("⚠️ 유효 인덱스 없음, 상위 10개로 Fallback")
            valid_idxs = list(range(1, min(11, max_n+1)))
        chosen_idxs = valid_idxs
        print("[Stage4] Final chosen indices:", chosen_idxs)
        # ── 여기까지 추가 ──
        
        # 3) 최종 결과 매핑 → raw_candidates 기준
        final_results = [ filtered_cands[i-1] for i in chosen_idxs ]   #10개 제한 시키기
        print("\n✅ 최종 추천 상품:")
        
        # ★ 여기에 10개 이상이면 앞 10개만 사용하도록 자르기 ★
        if len(final_results) > 10:
            final_results = final_results[:10]
        
        for idx, info in enumerate(final_results, start=1):
            PRODUCT_CACHE[info["상품코드"]] = info
            
            print(f"\n[{idx}] {info['제목']}")
            print(f"   상품코드   : {info['상품코드']}")
            print(f"   가격       : {info['가격']}원")
            print(f"   배송비     : {info['배송비']}원")
            print(f"   이미지     : {info['이미지']}")
            print(f"   원산지     : {info['원산지']}")
            print(f"   상품링크   : {info['상품링크']}")
            print(f"   옵션       : {info['옵션']}")
            print(f"   조합형옵션 : {info['조합형옵션']}")
            print(f"   최대구매수량: {info['최대구매수량']}")
        
        # print(f"PRODUCT_CACHE {PRODUCT_CACHE}")


        # ✅ 최종 결과 반환 및 출력 로그
        result_payload = {
            "query": query,  # 사용자가 입력한 원본 쿼리
            "UserMessage": UserMessage,  # 정제된 쿼리
            "RawContext": previous_queries + [query],  # 전체 대화 맥락
            "results": final_results,  # 검색 결과 리스트
            "combined_message_text": clean,  # LLM이 생성한 자연어 응답
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
    uvicorn.run(app, host="0.0.0.0", port=8011)
