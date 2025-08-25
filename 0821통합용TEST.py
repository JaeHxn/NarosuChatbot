#ownerclan_weekly_0428_전치리 LLM(카테고리를 추측) 추가용
import numpy as np
from langchain_openai import OpenAI, OpenAIEmbeddings
from pymilvus import Collection, connections
import os
from openai import OpenAI as OpenAIClient      # 공식 OpenAI 클라이언트
import json
import re
import base64
import urllib
from langdetect import detect
from collections import defaultdict, Counter
import math

from typing import List, Dict, Any, Tuple
import time
import pandas as pd
from dotenv import load_dotenv

# ── 설정 ─────────────────────────────────────────────────────────


#env
# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
REDIS_URL = "redis://localhost:6379/0"
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
MANYCHAT_API_KEY = os.getenv('MANYCHAT_API_KEY')
LLM_MODEL  = "gpt-4.1-mini"
EMB_MODEL  = "text-embedding-3-small"
COLLECTION = "ownerclan_weekly_0428"            # Milvus 컬렉션 이름
MILVUS_HOST = "114.110.135.96"
MILVUS_PORT = "19530"




# 클라이언트 및 래퍼
client    = OpenAIClient(api_key=API_KEY)
llm       = OpenAI(api_key=API_KEY, model=LLM_MODEL, temperature=0)
embedder  = OpenAIEmbeddings(api_key=API_KEY, model=EMB_MODEL)    # ← embedder 정의 추가
max_total = 10

API_URL = "https://fb-narosu.duckdns.org"  # 예: http://114.110.135.96:8011
# 최종 노출 쿼터
FINAL_QUOTA = {"Top3": 5, "Top4": 4, "Top5": 3}

# 예비까지 미리 뽑는 배수
PREFETCH_MULTIPLIER = 3  # → 15, 12, 9  3배수 

# LLM이 1회에 뽑아야 할 총 개수
PREFETCH_QUOTA = {k: v * PREFETCH_MULTIPLIER for k, v in FINAL_QUOTA.items()}

#가격 조건 파싱 (일단 기능 남겨둠)
pattern = re.compile(r'(\d+)[^\d]*원\s*(이하|미만|이상|초과)')

# 1) Milvus 서버에 먼저 연결
connections.connect(
    alias="default",
    host=MILVUS_HOST,    # 예: "114.110.135.96"
    port=MILVUS_PORT     # 예: "19530"
)
print("✅ Milvus에 연결되었습니다.")


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

collection = Collection(COLLECTION)   #다시 상품 DB 컬렉션으로 연결
def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
        return obj.item()
    return obj

PRODUCT_CACHE = {}

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

    

##############################################################################원본 사용자 쿼리
query = (
    "여름에 쓰는 모자"
)
print("[Debug] Raw query:", query)            # ← 여기에!
lang_code = detect(query)

print("[Debug] lang_code →", lang_code)   # ← 이 줄 추가!
#가격을 이해하는 매핑
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
def detect_season(query: str) -> str:
    summer_keywords = ["여름", "더위", "더운", "시원한", "햇빛"]
    winter_keywords = ["겨울", "추위", "추운", "따뜻한", "방한"]
    
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in summer_keywords):
        return "여름"
    elif any(keyword in query_lower for keyword in winter_keywords):
        return "겨울"
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
system_prompt = (
    f"""System:
당신은 (1) 검색 엔진의 전처리를 담당하는 AI이자, (2) 쇼핑몰 검색 및 분류 전문가입니다.
어떤 언어로 입력이 되든 반드시 한국어로 문장 의미에 맞게 번역 먼저 합니다.

다음 순서대로 응답하세요. 그리고 아래 '출력 형식'을 반드시, 정확히 지키세요.

1) 전처리 단계
   - 사용자 원문(query)에서 오타 교정, 불용어 제거, 중복 표현 제거, 속성 추출(계절/성별/색상/온도 등)
   - 핵심 키워드 하나를 뽑아 표준어/동의어로 치환.

2) 출력 형식(오직 두 줄, 따옴표 포함. 추가 설명/불릿/번호/코드블록 절대 금지):
Raw Query: "<query>"
Preprocessed Query: "<전처리된_쿼리>"
    """    
)

if price_cond:
    system_prompt += f"\n⚠️ 사용자 요청 조건: 가격은 **{amount}원 {comp}** ({price_cond})인 상품만 고려하세요.\n"



resp = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query}
    ],
    temperature=0
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
        param={"metric_type":"L2","params":{"nprobe":128}},
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
# --- (정확 단어 기반) 시즌 보정: 봄/여름/가을/겨울 전부 지원 ---
SPRING_STRONG = (
    "봄", "벚꽃", "꽃샘추위", "춘분", "봄맞이", "봄소풍", "봄나들이"
)

SUMMER_STRONG = (
    "여름", "썸머", "자외선차단", "uv차단", "썬캡", "비치", "바캉스",
    "래시가드", "수영복", "아쿠아슈즈", "쿨링", "냉감", "쿨토시",
    "라피아", "스트로", "스트로우", "비치샌들", "워터레깅스", "햇빛가리개"
)

AUTUMN_STRONG = (
    "가을", "오텀", "단풍", "추석", "한가위", "가을맞이", "수확제"
)

WINTER_STRONG = (
    "겨울", "윈터", "방한", "보온", "기모", "플리스", "보아",
    "다운", "패딩", "롱패딩", "스노우부츠", "핫팩", "바라클라바",
    "넥워머", "귀마개", "목도리", "장갑", "내복", "히트텍"
)

# 시즌별 긍/부정 세트 (겹침 줄이기 위해 '강 키워드'만 사용)
_SEASON_POS = {
    "봄": SPRING_STRONG,
    "여름": SUMMER_STRONG,
    "가을": AUTUMN_STRONG,
    "겨울": WINTER_STRONG,
}
# 봄/가을엔 극여름·극겨울 단어 감점, 여름/겨울엔 서로 반대 시즌만 감점
_SEASON_NEG = {
    "봄": SUMMER_STRONG + WINTER_STRONG,
    "가을": SUMMER_STRONG + WINTER_STRONG,
    "여름": WINTER_STRONG,
    "겨울": SUMMER_STRONG,
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


# 상위 3개 카테고리에 대해 각각 상품 검색
top3_products = []
for cat_name, _ in top5_cats[:3]:
    print(f"\n🔍 카테고리 '{cat_name}' 검색 시작...")
    
    # 1. 직접 검색 (boolean search)
    tokens = [t for t in re.sub(r"[용\s]+", " ", preprocessed_query).split() if t]
    direct_expr = f"category_name like '%{cat_name}%' && " + " && ".join(
        f'market_product_name like "%{tok}%"' for tok in tokens
    )
    
    direct_hits = collection.query(
        expr=direct_expr,
        limit=25,  # 최대 25개
        output_fields=[
            "product_code", "category_code", "category_name", 
            "market_product_name", "market_price", "shipping_fee",
            "shipping_type", "max_quantity", "composite_options",
            "image_url", "manufacturer", "model_name", "origin",
            "keywords", "description", "return_shipping_fee"
        ]
    )
    print(f"  ┣ 직접 검색 결과: {len(direct_hits)}개")
    
    # 2. 벡터 검색
    vector_hits = collection.search(
        data=[q_vec],
        anns_field="emb",
        param={"metric_type": "L2", "params": {"nprobe": 1536}},
        limit=25,  # 최대 25개
        expr=f"category_name like '%{cat_name}%'",
        output_fields=[
            "product_code", "category_code", "category_name", 
            "market_product_name", "market_price", "shipping_fee",
            "shipping_type", "max_quantity", "composite_options",
            "image_url", "manufacturer", "model_name", "origin",
            "keywords", "description", "return_shipping_fee"
        ]
    )
    print(f"  ┗ 벡터 검색 결과: {len(vector_hits[0])}개")
    
    # 결과 통합 및 가공
    cat_products = []
    
    # 직접 검색 결과 처리
    for hit in direct_hits:
        try:
            html_raw = hit.get("description", "") or ""
            html_cleaned = clean_html_content(html_raw)
            if isinstance(html_raw, bytes):
                html_raw = html_raw.decode("cp949")
            encoded_html = base64.b64encode(html_cleaned.encode("utf-8", errors="ignore")).decode("utf-8")
            preview_url = f"{API_URL}/preview?html={urllib.parse.quote_plus(encoded_html)}"
            
            product_info = {
                "상품코드": str(hit.get("product_code", "없음")),
                "제목": hit.get("market_product_name", "제목 없음"),
                "가격": convert_to_serializable(hit.get("market_price", 0)),
                "배송비": convert_to_serializable(hit.get("shipping_fee", 0)),
                "이미지": hit.get("image_url", "이미지 없음"),
                "원산지": hit.get("origin", "정보 없음"),
                "상품링크": preview_url,
                "카테고리": hit.get("category_name", "카테고리 없음"),
                "검색방식": "직접검색"
            }
            cat_products.append(product_info)
        except Exception as e:
            print(f"  ⚠️ 상품 정보 처리 오류: {e}")
    
    # 벡터 검색 결과 처리
    for hits in vector_hits:
        for hit in hits:
            try:
                html_raw = hit.entity.get("description", "") or ""
                html_cleaned = clean_html_content(html_raw)
                if isinstance(html_raw, bytes):
                    html_raw = html_raw.decode("cp949")
                encoded_html = base64.b64encode(html_cleaned.encode("utf-8", errors="ignore")).decode("utf-8")
                preview_url = f"{API_URL}/preview?html={urllib.parse.quote_plus(encoded_html)}"
                
                product_info = {
                    "상품코드": str(hit.entity.get("product_code", "없음")),
                    "제목": hit.entity.get("market_product_name", "제목 없음"),
                    "가격": convert_to_serializable(hit.entity.get("market_price", 0)),
                    "배송비": convert_to_serializable(hit.entity.get("shipping_fee", 0)),
                    "이미지": hit.entity.get("image_url", "이미지 없음"),
                    "원산지": hit.entity.get("origin", "정보 없음"),
                    "상품링크": preview_url,
                    "카테고리": hit.entity.get("category_name", "카테고리 없음"),
                    "검색방식": "벡터검색"
                }
                cat_products.append(product_info)
            except Exception as e:
                print(f"  ⚠️ 상품 정보 처리 오류: {e}")
    
    # 중복 제거 (상품코드 기준)
    seen_codes = set()
    unique_products = []
    for p in cat_products:
        if p["상품코드"] not in seen_codes:
            seen_codes.add(p["상품코드"])
            unique_products.append(p)
    
    print(f"  ✅ 최종 상품 수: {len(unique_products)}개")
    top3_products.extend(unique_products)

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



    
# # ① Stage1: 직접 문자열 검색 (boolean search)
# print("[Stage1] Direct name search 시작")

# # “남자용 향수” → ["남자", "향수"] 두 토큰으로 AND 검색
# tokens = [t for t in re.sub(r"[용\s]+", " ", preprocessed_query).split() if t]
# query_expr = " && ".join(f'market_product_name like "%{tok}%"'
#     for tok in tokens
# )

# print("[Debug] Stage1 expr:", query_expr)
# direct_hits = collection.query(
#     expr=query_expr,
#     limit=200,
#     output_fields=[
#         "product_code",
#         "category_code",
#         "category_name",
#         "market_product_name",
#         "market_price",
#         "shipping_fee",
#         "shipping_type",
#         "max_quantity",
#         "composite_options",
#         "image_url",
#         "manufacturer",
#         "model_name",
#         "origin",
#         "keywords",
#         "description",
#         "return_shipping_fee",
#     ]
# )
# print("[Stage1] Direct hits count:", len(direct_hits))
# # 샘플 출력
# for i, row in enumerate(direct_hits[:7], 1):
#     print(f"  [Stage1 샘플 {i}]: 코드={row['product_code']}, 이름={row['market_product_name']}")


# print("\n[Stage2.5] 직접검색 results 구성 시작")  
# raw_candidates = []
# for row in direct_hits:
#     # e = hit.entity
#     # 본문 미리보기 링크
#     try:
#         html_raw = row.get("description", "") or ""
#         html_cleaned = clean_html_content(html_raw)
#         if isinstance(html_raw, bytes):
#             html_raw = html_raw.decode("cp949")
#         encoded_html = base64.b64encode(
#             html_cleaned.encode("utf-8", errors="ignore")
#         ).decode("utf-8")
#         safe_html = urllib.parse.quote_plus(encoded_html)
#         preview_url = f"{API_URL}/preview?html={safe_html}"
#     except Exception as err:
#         print(f"⚠️ 본문 처리 오류: {err}")
#         preview_url = "https://naver.com"

#     # 상품링크(fallback)
#     product_link = row.get("product_link", "")
#     if not product_link or product_link in ["링크 없음", "#", None]:
#         product_link = preview_url

#     # 옵션 파싱
#     option_raw = str(row.get("composite_options", "")).strip()
#     option_display = "없음"
#     if option_raw.lower() not in ["", "nan"]:
#         parsed = []
#         for line in option_raw.splitlines():
#             try:
#                 name, extra, _ = line.split(",")
#                 extra = int(float(extra))
#                 parsed.append(
#                     f"{name.strip()}{f' (＋{extra:,}원)' if extra>0 else ''}"
#                 )
#             except Exception:
#                 parsed.append(line.strip())
#         option_display = "\n".join(parsed)

#     # 10개 한글 속성으로 딕셔너리 구성
#     result_info = {
#         "상품코드":     str(row.get("product_code", "없음")),
#         "제목":        row.get("market_product_name", "제목 없음"),
#         "가격":        convert_to_serializable(row.get("market_price", 0)),
#         "배송비":      convert_to_serializable(row.get("shipping_fee", 0)),
#         "이미지":      row.get("image_url", "이미지 없음"),
#         "원산지":      row.get("origin", "정보 없음"),
#         "상품링크":    product_link,
#         "옵션":        option_display,
#         "조합형옵션":  option_raw,
#         "최대구매수량": convert_to_serializable(row.get("max_quantity", 0)),
#         "카테고리":    row.get("category_name", "카테고리 없음"),
#     }
#     result_info_cleaned = {}
#     for k, v in result_info.items():
#         if isinstance(v, str):
#             v = v.replace("\n", "").replace("\r", "").replace("\t", "")
#         result_info_cleaned[k] = v
#     raw_candidates.append(result_info_cleaned)

# # ② Stage2: 벡터 유사도 검색
# # expr = f'category_name like "%{top_category}%"'   #최상위 카테고리
# expr = f'category_name like "%{lowest_subcategory}%"'   #최하위 카테고리
# milvus_results = collection.search(
#     data=[q_vec],
#     anns_field="emb",  # ← 벡터 저장된 필드 이름
#     param={"metric_type": "L2", "params": {"nprobe": 1536}},
#     limit=200,
#     expr=expr,                              # ← 이 줄 추가
#     output_fields = [
#     "product_code",
#     "category_code",
#     "category_name",
#     "market_product_name",
#     "market_price",
#     "shipping_fee",
#     "shipping_type",
#     "max_quantity",
#     "composite_options",
#     "image_url",
#     "manufacturer",
#     "model_name",
#     "origin",
#     "keywords",
#     "description",
#     "return_shipping_fee",
# ]
# )
# print(f"[Stage2] Vector hits count: {len(milvus_results[0])}")
# # 샘플 출력



# for i, hit in enumerate(milvus_results[0][:7], 1):
#     e = hit.entity
#     print(f"  [Stage2 샘플 {i}]: 코드={e.get('product_code')}, 이름={e.get('market_product_name')}")

    
# #벡터 검색 담는 부분
# for hits in milvus_results:
#     for hit in hits:
#         e = hit.entity
#         # 본문 미리보기 링크
#         try:
#             html_raw = e.get("description", "") or ""
#             html_cleaned = clean_html_content(html_raw)
#             if isinstance(html_raw, bytes):
#                 html_raw = html_raw.decode("cp949")
#             encoded_html = base64.b64encode(
#                 html_cleaned.encode("utf-8", errors="ignore")
#             ).decode("utf-8")
#             safe_html = urllib.parse.quote_plus(encoded_html)
#             preview_url = f"{API_URL}/preview?html={safe_html}"
#             # preview_url = f"{safe_html}"
#         except Exception as err:
#             print(f"⚠️ 본문 처리 오류: {err}")
#             preview_url = "https://naver.com"

#         # 상품링크(fallback)
#         product_link = e.get("product_link", "")
#         if not product_link or product_link in ["링크 없음", "#", None]:
#             product_link = preview_url

#         # 옵션 파싱
#         option_raw = str(e.get("composite_options", "")).strip()
#         option_display = "없음"
#         if option_raw.lower() not in ["", "nan"]:
#             parsed = []
#             for line in option_raw.splitlines():
#                 try:
#                     name, extra, _ = line.split(",")
#                     extra = int(float(extra))
#                     parsed.append(
#                         f"{name.strip()}{f' (＋{extra:,}원)' if extra>0 else ''}"
#                     )
#                 except Exception:
#                     parsed.append(line.strip())
#             option_display = "\n".join(parsed)

#         # 10개 한글 속성으로 딕셔너리 구성
#         result_info = {
#             "상품코드":     str(e.get("product_code", "없음")),
#             "제목":        e.get("market_product_name", "제목 없음"),
#             "가격":        convert_to_serializable(e.get("market_price", 0)),
#             "배송비":      convert_to_serializable(e.get("shipping_fee", 0)),
#             "이미지":      e.get("image_url", "이미지 없음"),
#             "원산지":      e.get("origin", "정보 없음"),
#             "상품링크":    product_link,
#             "옵션":        option_display,
#             "조합형옵션":  option_raw,
#             "최대구매수량": convert_to_serializable(e.get("max_quantity", 0)),
#             "카테고리":    e.get("category_name", "카테고리 없음"),
#         }

#         # 문자열 정리 후 리스트에 추가
#         result_info_cleaned = {}
#         for k, v in result_info.items():
#             if isinstance(v, str):
#                 v = v.replace("\n", "").replace("\r", "").replace("\t", "")
#             result_info_cleaned[k] = v
#         raw_candidates.append(result_info_cleaned)

#         # 캐시에 안전 저장
#         product_code = result_info_cleaned.get("상품코드")

# # 개수 및 샘플 확인
# print(f"[Stage2.5] raw_candidates count: {len(raw_candidates)}")
# for i, info in enumerate(raw_candidates[:3],1):
#     print(f"  [raw_candidates 샘플 {i}]:", info)


# # Stage2.5 완료 후: 원본 보관
# original_candidates = raw_candidates.copy()

# # ① Top4 필터 + quota 계산
# filtered_cands, quotas, top4_keys = prepare_recommendation(
#     all_candidates=original_candidates,
#     max_total=max_total,
#     min_per_category=1
# )

# print("🔝 Top4 카테고리:", top4_keys)
# print("🗂️ 카테고리별 quota:", quotas)

# total = len(raw_candidates)
# print(f"🔍 총 후보: {total}개")

# # 비율
# props = compute_category_proportions(filtered_cands)
# print("📊 카테고리별 비율:")
# for cat, ratio in props.items():
#     print(f"  {cat}: {ratio*100:.1f}%")

    

# # quota (최종 5개 배정 기준 예시)
# quotas = compute_top4_quota(filtered_cands, max_total=max_total,min_per_category=1)
# print("🗂️ 카테고리별 추천 개수(quota):")
# for cat, q in quotas.items():
#     print(f"  {cat}: {q}개")

# # ── 3) Prompt에 quota 가이드 추가 ────────────────────────────────
# def quota_to_text(quota: Dict[str, int]) -> str:
#     return "\n".join([f"- {cat}: {num}개" for cat, num in quota.items()])

# quota_text = quota_to_text(quotas)
# print(f"quota_text ->   {quota_text}")
# # 후보 리스트(원본 전부)

# candidate_list = "\n".join(
#     f"{i+1}. {c['제목']} [{c['카테고리']}]"
#     for i, c in enumerate(filtered_cands)
# )

# print("[Stage4] LLM에 넘길 후보 리스트:\n", candidate_list[:100], "...")  # 앞부분만 출력

# rank_prompt = f"""
# **답변은 반드시 "{target_lang}"로 해주세요.**

# User Query: "{query}"
# # 예측된 카테고리: "{predicted_category}"
# # 아래 후보들은 모두 이 카테고리에 속합니다. 

# 후보리스트 : {candidate_list}에는 이미 Top4 카테고리만 필터링된 상품들이 포함되어 있습니다.

# **지침:**

# 1. **카테고리별 필터링 적용**  
#    - {quota_text}에 명시된 각 카테고리별 할당량만큼, candidate_list에서 반드시 해당 카테고리 상품을 정확히 그 개수만큼 나열하세요.  
#    - 예: “패션의류>남성의류>티셔츠: 4개”라면, 후보 리스트 중 해당 카테고리 상품 4개를 출력합니다.

# 2.  **상품 리스트 절대 출력 금지**  
#    - 후보리스트나 카테고리별 항목을 **하나도** 화면에 보여주지 말고, 
   
# 3. **추가 탐색 질문 생성 (200~250자)**  
#    - 나열된 상품 메타데이터(상품코드, 제목, 가격, 이미지 URL 등)만 참고해서,  
#      사용자가 선택 폭을 좁힐 수 있는 자연어 질문(200~250자)만 작성하세요.  

# 4. **JSON 배열**  
#    - 제시된 후보 중 **반드시 {max_total}개의** 사용자의 의도에 가장 적합한 항목의 번호만을 JSON 배열 형태로 반환하세요.  
#    - 반드시 **예시와 같은 형식**으로만 출력합니다:  
    
#     (예시)
#     [1,2,3,4,5,6,7,8,9,10]

# """

# resp2 = client.chat.completions.create(
#     model=LLM_MODEL,
#     messages=[{"role":"system","content":rank_prompt}],
#     temperature=0
# )
# selection = resp2.choices[0].message.content.strip()

# # 1) Raw LLM 응답
# print("\n=== [Stage4] Raw LLM selection ===")
# print(selection)

# # 2) JSON 배열 위치 찾기
# match = re.search(r'\[\s*(?:\d+\s*,\s*)*\d+\s*\]', selection)
# if match:
#     arr_text = match.group(0)
#     try:
#         chosen_idxs = json.loads(arr_text)
#     except json.JSONDecodeError:
#         chosen_idxs = []
#     # narrative만 clean에 저장
#     clean = selection[:match.start()].strip()
# else:
#     chosen_idxs = []
#     clean = selection.strip()

# # 3) Narrative (추가 탐색 질문)만 출력
# print("\n=== [Stage4] Cleaned selection (narrative only) ===")
# print(clean)

# # 4) JSON 파싱된 인덱스 리스트 출력
# print("\n=== [Stage4] Chosen indices ===")
# print(chosen_idxs)

# # ── 여기에 추가 ──
# # ②.1 인덱스 범위 검증
# max_n = len(filtered_cands)
# valid_idxs = [i for i in chosen_idxs if 1 <= i <= max_n]
# if len(valid_idxs) < len(chosen_idxs):
#     print(f"⚠️ 잘못된 인덱스 제거됨: {set(chosen_idxs) - set(valid_idxs)}")
# if not valid_idxs:
#     print("⚠️ 유효 인덱스 없음, 상위 10개로 Fallback")
#     valid_idxs = list(range(1, min(11, max_n+1)))
# chosen_idxs = valid_idxs
# print("[Stage4] Final chosen indices:", chosen_idxs)
# # ── 여기까지 추가 ──
# # 3) 최종 결과 매핑 → raw_candidates 기준
# final_results = [ filtered_cands[i-1] for i in chosen_idxs ]   #10개 제한 시키기
# print("\n✅ 최종 추천 상품:")

# # ★ 여기에 10개 이상이면 앞 10개만 사용하도록 자르기 ★
# if len(final_results) > 10:
#     final_results = final_results[:10]

# for idx, info in enumerate(final_results, start=1):
#     PRODUCT_CACHE[info["상품코드"]] = info
    
#     print(f"\n[{idx}] {info['제목']}")
#     print(f"   카테고리   : {info['카테고리']}")
#     print(f"   상품코드   : {info['상품코드']}")
#     print(f"   가격       : {info['가격']}원")
#     print(f"   배송비     : {info['배송비']}원")
#     print(f"   이미지     : {info['이미지']}")
#     print(f"   원산지     : {info['원산지']}")
#     print(f"   상품링크   : {info['상품링크']}")
#     print(f"   옵션       : {info['옵션']}")
#     print(f"   조합형옵션 : {info['조합형옵션']}")
#     print(f"   최대구매수량: {info['최대구매수량']}")
# print(f"================================")
# # print(f"PRODUCT_CACHE {PRODUCT_CACHE}")