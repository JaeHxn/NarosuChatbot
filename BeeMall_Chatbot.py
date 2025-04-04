import asyncio
import base64
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import faiss
import numpy as np
import pandas as pd
import redis
import requests
import uvicorn
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

# ✅ FAISS 인덱스 파일 경로 설정
faiss_file_path = f"03_25_faiss_index_3s.faiss"

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


# 응답 속도 측정을 위한 미들웨어 추가
@app.middleware("http")
async def measure_response_time(request: Request, call_next):
    start_time = time.time()  # 요청 시작 시간
    response = await call_next(request)  # 요청 처리
    process_time = time.time() - start_time  # 처리 시간 계산

    response.headers["ngrok-skip-browser-warning"] = "true"
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

# ✅ Redis 기반 메시지 기록 관리 함수
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
        raise HTTPException(status_code=500, detail="Redis 연결에 문제가 발생했습니다.")

# 요청 모델
class QueryRequest(BaseModel):
    query: str


# ✅ JSON 직렬화를 위한 int 변환 함수
def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
        return obj.item()
    return obj

# ✅ 엑셀 데이터 로드 및 변환 (공백 제거)
def load_excel_to_texts(file_path):
    try:
        data = pd.read_excel(file_path)
        data.columns = data.columns.str.strip()
        texts = [" | ".join([f"{col}: {row[col]}" for col in data.columns]) for _, row in data.iterrows()]
        return texts, data
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
        file_path = os.path.join(os.getcwd(), "db", "ownerclan_인기상품_1만개.xlsx")
        
        # 🔍 엑셀 데이터 로드 확인
        texts, data = load_excel_to_texts(file_path)
        print(data.head())  # 데이터의 첫 5개 행 출력 (엑셀 데이터 확인용)
        
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

        # 기존 대화 이력과 함께 LLM에 전달
        response = llm.invoke([
            SystemMessage(content="사용자의 대화 내역을 반영하여 상품 검색을 위한 핵심 키워드를 추출해주세요. 만약 단어 간에 띄어쓰기가 있다면 하나의 단어 일수도 있습니다 띄어쓰기가 있다면 단어끼리 붙여서도 문장을 분석해보세요. 여러방법,여러 방면으로 생각해서 추출해주세요. 다른 나라 언어로 질문이 들어오면 질문을 먼저 한글로 번역해서 단어를 추출합니다."),
            HumanMessage(content=f"질문: {query} \n ")
        ])

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

store = {}  # 빈 딕셔너리를 초기화합니다.
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # 세션 ID에 해당하는 대화 기록이 저장소에 없으면 새로운 ChatMessageHistory를 생성합니다.
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    # 세션 ID에 해당하는 대화 기록을 반환합니다.
    return store[session_id]

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
async def handle_webhook(request: Request):
    start_time = time.time()

    try:
        # Step 1: 요청 데이터 로드
        data = await request.json()
        parse_time = time.time() - start_time
        logger.info(f"📊 [Parse Time]: {parse_time:.4f} 초")

        # Step 2: 메시지 처리
        process_start = time.time()

        if data.get("field") == "messages":  # field 값이 'messages'인지 확인
            value = data.get("value", {})  # value 필드 가져오기

            # Redis 세션 ID 설정
            sender_id = value.get("sender", {}).get("id")  # 발신자 ID
            # print(f"유저아이디 : {sender_id}")

            # 사용자 메시지 가져오기
            user_message = value.get("message", {}).get("text", "").strip()  # 메시지 텍스트
            # print(f"유저메세지 : {user_message}")
            if sender_id and user_message:
                if user_message.lower() == "reset":
                    print(f"🔄 [RESET] 세션 {sender_id}의 대화 기록 초기화!")
                    clear_message_history(sender_id)  # Redis 대화 기록 초기화
                    return {
                        "version": "v2",
                        "content": {
                            "messages": [
                                {
                                    "type": "text",
                                    "text": f"✅ 세션 {sender_id}의 대화 기록이 초기화되었습니다!"
                                }
                            ]
                        },
                        "message": f"세션 {sender_id}의 대화 기록이 초기화되었습니다."
                    }
                # ✅ AI 응답을 비동기적으로 처리 (별도로 실행)
                asyncio.create_task(process_ai_response(sender_id, user_message))
            
            process_time = time.time() - process_start
            logger.info(f"📊 [Processing Time 메시지 처리 전체 시간]: {process_time:.4f} 초")
        print(data)
        return {
            "version": "v2",

            "content": {
                "messages": [
                    {
                        "type": "text",
                        "text": "입력이 완료 되어 AI가 생각중입니다.."
                    }
                ]
            }
        }    
    
    except Exception as e:
        print(f"❌ 웹훅 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
async def process_ai_response(sender_id: str, user_message: str):
    try:
        print(f"🕒 [AI 처리 시작] 유저 ID: {sender_id}, 메시지: {user_message}")

        # AI 응답 생성 (비동기 처리)
        loop = asyncio.get_running_loop()
        bot_response = await loop.run_in_executor(executor, external_search_and_generate_response, user_message, sender_id)

        # ✅ 응답 확인 및 전송 처리
        if isinstance(bot_response, dict):
            combined_message_text = bot_response.get("combined_message_text", "")
            results = bot_response.get("results", [])

            # ✅ 전송할 메시지 데이터 목록 (이제 리스트가 아님)
            messages_data = []

            # ✅ AI 응답 메시지 추가 (combined_message_text가 있을 경우에만)
            if combined_message_text:
                messages_data.append({
                    "type": "text",
                    "text": combined_message_text
                })

            # ✅ 상품 정보들을 딕셔너리로 추가
            for product in results:
                if product.get("이미지"):
                    messages_data.append({
                        "type": "image",
                        "url": product["이미지"]
                    })
                messages_data.append({
                    "type": "text",
                    "text": f"✨ {product['제목']}\n\n가격: {product['가격']}원\n배송비: {product['배송비']}원\n원산지: {product['원산지']}\n",
                    "buttons": [
                        {"type": "url", "caption": "상품 보러가기", "url": product.get("상품링크", "#"), "webview": "full"},
                        {"type": "url", "caption": "구매하기", "url": product.get("상품링크", "#")}
                    ]
                })

            # ✅ send_message()에 원시 데이터 리스트를 넘김
            send_message(sender_id, messages_data)
            print(f"✅ [Combined 메시지 전송 완료]: {combined_message_text}")

        else:
            print(f"❌ AI 응답 오류 발생")

    except Exception as e:
        print(f"❌ AI 응답 처리 오류: {e}")


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
        # ✅ [Step 3] Redis 메시지 기록 관리
        redis_start = time.time()
        session_history = get_message_history(session_id)
        redis_time = time.time() - redis_start
        print(f"📊 [Step 3] Redis 메시지 기록 관리 시간: {redis_time:.4f} 초")

        # ✅ [Step 4] 기존 대화 내역 확인
        previous_queries = [msg.content for msg in session_history.messages if isinstance(msg, HumanMessage)]
        print(f"🔍 [Step 4] Redis 기존 대화 내역: {previous_queries}")

        # ✅ [Step 5] LLM 키워드 추출
        llm_start = time.time()
        combined_query = " ".join(previous_queries + [query])
        print(f"🔍 [Step 4-1]combined_query: {combined_query}")

        # ✅ extract_keywords_with_llm 실행 전 확인
        if not combined_query or not isinstance(combined_query, str):
            raise ValueError(f"❌ [ERROR] combined_query가 올바른 문자열이 아닙니다: {combined_query} (타입: {type(combined_query)})")


        combined_keywords = extract_keywords_with_llm(combined_query)

        llm_time = time.time() - llm_start

        if not combined_keywords or not isinstance(combined_keywords, str):
            raise ValueError(f"❌ [ERROR] 키워드 추출 실패: {combined_keywords}")
        
        print(f"🔍 [Step 4-2] combined_keywords: {combined_keywords}")
        print(f"✅ [Step 5] 생성된 검색 키워드: {combined_keywords}")
        print(f"📊 [Step 5-1] LLM 키워드 추출 시간: {llm_time:.4f} 초")

        # ✅ [Step 6] Redis에 사용자 입력 추가
        session_history.add_message(HumanMessage(content=query))
        print(f"🔍 [Step 6] Redis 메시지 기록 (변경된 상태): {session_history.messages}")
        
        # ✅ [Step 7] 엑셀 데이터 로드
        excel_start = time.time()
        
        try:
            _, data = load_excel_to_texts("db/ownerclan_인기상품_1만개.xlsx")
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
                    result_info = {
                        "상품코드": str(result_row.get("상품코드", "없음")),
                        "제목": result_row.get("원본상품명", "제목 없음"),
                        "가격": convert_to_serializable(result_row.get("오너클랜판매가", 0)),
                        "배송비": convert_to_serializable(result_row.get("배송비", 0)),
                        "이미지": result_row.get("이미지중", "이미지 없음"),
                        "원산지": result_row.get("원산지", "정보 없음"),
                        "상품링크": result_row.get("본문상세설명", "링크 없음"),
                    }
                    results.append(result_info)
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
            
        message_history=[]
        
        # ✅ [Step 12] LLM 기반 대화 응답 생성
        start_response = time.time()    
        # ✅ ChatPromptTemplate 및 RunnableWithMessageHistory 생성
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
        당신은 쇼핑몰 챗봇으로, 친절하고 인간적인 대화를 통해 고객의 쇼핑 경험을 돕습니다.

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

        # ✅ Redis에 AI 응답 추가
        session_history.add_message(AIMessage(content=response.content))

        # ✅ 메시지 기록을 Redis에서 가져오기
        session_history = get_message_history(session_id)
        message_history = [
            {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
            for msg in session_history.messages
        ]


        # ✅ 출력 디버깅
        print("*** Response:", response)
        print("*** Message History:", message_history)
        print("✅✅✅✅*✅✅✅✅ Results:", results)
        print(f"✅ [Before Send] Results Type: {type(results[:5])}")
        print(f"✅ [Before Send] Results Content: {results[:5]}")

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
        # ✅ ManyChat API URL 및 헤더 설정
        url = "https://api.manychat.com/fb/sending/sendContent"
        headers = {
            "Authorization": f"Bearer {MANYCHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # ✅ 전송 데이터 검증
        if not isinstance(messages, list):
            print(f"❌ [ERROR] messages는 리스트여야 합니다. 전달된 타입: {type(messages)}")
            return
        
        # ✅ 보낼 데이터 형식 확인
        print(f"✅ [Before Send] Messages Content: {messages}")

        # ✅ URL 값 확인 후 변경
        for message in messages:
            if message.get("buttons"):
                for button in message["buttons"]:
                    if button["url"] in ["링크 없음", "#", None, ""]:
                        button["url"] = "https://naver.com"  # 예시 URL로 변경
                        
        # ✅ ManyChat API로 보낼 데이터 구성
        # Step 1: LLM 응답 메시지를 먼저 보내기
        llm_message = {
            "type": "text",
            "text": messages[0]['text']  # LLM 응답 메시지
        }

        # ✅ 보낼 데이터 구성 (ManyChat 형식에 맞게)
        data = {
            "subscriber_id": sender_id,
            "data": {
                "version": "v2",
                "content": {
                    "messages": [llm_message],  # 먼저 LLM 응답 메시지를 보냄
                    "actions": [],
                    "quick_replies": []
                }
            },
            "message_tag": "ACCOUNT_UPDATE"
        }
        
        # ✅ LLM 응답 메시지 보내기
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"✅ [ManyChat LLM 메시지 전송 성공]: {response.json()}")
        else:
            print(f"❌ [ManyChat LLM 메시지 전송 실패] 상태 코드: {response.status_code}, 오류 내용: {response.text}")

        # Step 2: 상품 정보 메시지들 보내기
        for message in messages[1:]:
            data = {
                "subscriber_id": sender_id,
                "data": {
                    "version": "v2",
                    "content": {
                        "messages": [message],  # 개별 상품 메시지
                        "actions": [],
                        "quick_replies": []
                    }
                },
                "message_tag": "ACCOUNT_UPDATE"
            }

            # ✅ JSON 데이터 직렬화 검사
            try:
                json_string = json.dumps(data)  # JSON 직렬화 테스트
                print(f"✅ JSON 직렬화 성공: {json_string[:500]}...")  # 처음 500자만 출력
            except Exception as e:
                print(f"❌ [JSON Error] JSON 데이터 직렬화 오류: {e}")
                continue  # 문제 발생 시 해당 메시지 건너뛰기
            
            # ✅ ManyChat API로 개별 상품 메시지 전송
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                print(f"✅ [ManyChat 개별 메시지 전송 성공]: {response.json()}")
            else:
                print(f"❌ [ManyChat 메시지 전송 실패] 상태 코드: {response.status_code}, 오류 내용: {response.text}")
    
    except Exception as e:
        print(f"❌ ManyChat 메시지 전송 오류: {e}")





# ✅ 루트 경로 - HTML 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def generate_bot_response(user_message: str) -> str:
    """
    사용자의 메시지를 받아 챗봇 응답을 생성합니다.
    """
    try:
        # ✅ Redis를 이용한 세션 관리
        session_id = f"user_{user_message[:10]}"  # 간단한 세션 ID 생성 (필요 시 사용자 ID 사용)
        session_history = get_message_history(session_id)

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
################################################################
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
        session_history = get_message_history(session_id)
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

        _, data = load_excel_to_texts("db/ownerclan_인기상품_1만개.xlsx")

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
        session_history = get_message_history(session_id)
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

# ✅ FastAPI 서버 실행 (포트 고정: 5050)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
    
    