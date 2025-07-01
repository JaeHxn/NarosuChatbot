from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from io import BytesIO
import pandas as pd
import numpy as np
import re
import os
import time
import tiktoken        

from openai import OpenAI
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema,
    DataType, Collection
)

app = FastAPI()

# ── 설정 ─────────────────────────────────────────────────────────

# ── 설정 ─────────────────────────────────────────────────────────
API_KEY     = os.getenv('OPENAI_API_KEY')
MODEL       = "text-embedding-3-small"
MILVUS_HOST = '114.110.135.96'
# CHUNK_SIZE  = 2000
MILVUS_PORT = '19530'
COLLECTION  = "ownerclan_weekly_0428"

MAX_TOKENS  = 200_000
QUERY_BATCH = 1000  # 한 번에 조회할 상품코드 개수
# ─────────────────────────────────────────────────────────────

# 1) OpenAI 클라이언트 & Milvus 연결
client = OpenAI(api_key=API_KEY)
connections.connect(alias='default', host=MILVUS_HOST, port=MILVUS_PORT)

# 사전 max_length 계산
excel_paths = glob("db/ownerclan_narosu_오너클랜50만_*.xlsx")

# 한글→영문 필드명 매핑
column_map = {
    "상품코드":        "product_code",
    "카테고리코드":    "category_code",
    "카테고리명":      "category_name",
    "마켓상품명":      "market_product_name",
    "마켓실제판매가":  "market_price",
    "배송비":          "shipping_fee",
    "배송유형":        "shipping_type",
    "최대구매수량":    "max_quantity",
    "조합형옵션":      "composite_options",
    "이미지중":        "image_url",
    "제작/수입사":     "manufacturer",
    "모델명":          "model_name",
    "원산지":          "origin",
    "키워드":          "keywords",
    "본문상세설명":    "description",
    "반품배송비":      "return_shipping_fee",
    "독립형":          "independent_option",
    "조합형":          "composite_flag"
}
numeric_fields = {"market_price", "shipping_fee", "max_quantity", "return_shipping_fee"}

# 임베딩용 텍스트 생성 함수
def make_text(row):
    return " || ".join([
        f"cat:{row['카테고리명']}",
        f"name:{row['마켓상품명']}",
        f"opts:{row['조합형옵션']}"
    ])

# 토큰 기준 배치 분할
def split_batches(texts, max_tokens=MAX_TOKENS):
    enc = tiktoken.get_encoding("cl100k_base")
    batches, batch, tokens = [], [], 0

    for t in texts:
        tk = len(enc.encode(str(t)))
        # 단일 텍스트가 한도 초과하면, 기존 배치 닫고 단독 배치로
        if tk > max_tokens:
            if batch:
                batches.append(batch)
            batches.append([t])
            batch, tokens = [], 0
            continue

        # 기존 배치에 넣으면 초과될 때, 배치 닫고 새로 시작
        if tokens + tk > max_tokens:
            batches.append(batch)
            batch, tokens = [], 0

        batch.append(t)
        tokens += tk

    if batch:
        batches.append(batch)
    return batches


# “max_tokens_per_request” 오류 발생 시 반으로 쪼개 재시도
def try_batch(batch: list[str]) -> list[list[float]]:
    try:
        resp = client.embeddings.create(input=batch, model=MODEL)
        return [d.embedding for d in resp.data]
    except Exception as e:
        msg = getattr(e, "args", [None])[0]
        if "max_tokens_per_request" in str(msg) and len(batch) > 1:
            mid = len(batch) // 2
            return try_batch(batch[:mid]) + try_batch(batch[mid:])
        else:
            # 단일 항목에서도 실패하면 건너뜀
            return []
            


# 5) 컬렉션 로드 or 생성
if not utility.has_collection(COLLECTION):
    fields = [
        FieldSchema("product_code", DataType.VARCHAR, is_primary=True, max_length=200, auto_id=False),
        FieldSchema("emb",  DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema("text", DataType.VARCHAR,      max_length=65535),
        FieldSchema("category_code",       DataType.VARCHAR,      max_length=20),
        FieldSchema("category_name",       DataType.VARCHAR,      max_length=256),
        FieldSchema("market_product_name", DataType.VARCHAR,      max_length=512),
        FieldSchema("market_price",        DataType.INT64),
        FieldSchema("shipping_fee",        DataType.INT64),
        FieldSchema("shipping_type",       DataType.VARCHAR,      max_length=16),
        FieldSchema("max_quantity",        DataType.INT64),
        FieldSchema("composite_options",   DataType.VARCHAR,      max_length=65535),
        FieldSchema("image_url",           DataType.VARCHAR,      max_length=2048),
        FieldSchema("manufacturer",        DataType.VARCHAR,      max_length=128),
        FieldSchema("model_name",          DataType.VARCHAR,      max_length=256),
        FieldSchema("origin",              DataType.VARCHAR,      max_length=100),
        FieldSchema("keywords",            DataType.VARCHAR,      max_length=1024),
        FieldSchema("description",         DataType.VARCHAR,      max_length=65535),
        FieldSchema("return_shipping_fee", DataType.INT64),
        FieldSchema("independent_option",  DataType.VARCHAR,      max_length=8192),   # 텍스트로
        FieldSchema("composite_flag",      DataType.VARCHAR,      max_length=16384),   # 텍스트로
        FieldSchema("embedding_date",      DataType.INT64)  # ← 여기에 추가
    ]

    schema     = CollectionSchema(fields, description="Weekly Top 50k Products")
    collection = Collection(name=COLLECTION, schema=schema)
    collection.create_index(
        field_name="emb",
        index_params={"index_type":"IVF_FLAT","metric_type":"L2","params":{"nlist":128}}
    )
else:
    collection = Collection(name=COLLECTION)
collection.load()

# 6) 엑셀 파일별 Upsert 루프
start_all = time.time()
for path in tqdm(excel_paths, desc="📊 엑셀 처리 진행률", unit="파일"):
    t0 = time.time()
    df = pd.read_excel(path, header=1)

    # (1) 임베딩용 텍스트 미리 계산
    df["__text"] = df.apply(make_text, axis=1)

    # ─────────────────────────────────────────────────────────────────
    # (2) Milvus에 로드 & 기존 저장된 메타데이터 조회
    collection.load()
    codes_all = df["상품코드"].astype(str).tolist()
    expr = "product_code in [" + ",".join(f'"{c}"' for c in codes_all) + "]"

    # 여기에 우리가 비교할 모든 필드를 나열
    compare_fields = ["product_code", "text"] + list(numeric_fields) + [
    "shipping_type", "max_quantity", "composite_options",
    "independent_option", "composite_flag",
    "category_code", "category_name", "market_product_name", "image_url",
    "manufacturer", "model_name", "origin", "keywords", "description"
]

    # existing_rows = collection.query(expr, output_fields=compare_fields)
    
    existing_meta = {}
    for i in range(0, len(codes_all), QUERY_BATCH):
        batch_codes = codes_all[i : i + QUERY_BATCH]
        expr = "product_code in [" + ",".join(f'"{c}"' for c in batch_codes) + "]"
        rows = collection.query(expr, output_fields=compare_fields)
        for r in rows:
            existing_meta[r["product_code"]] = { f: r[f] for f in compare_fields }

    # ─────────────────────────────────────────────────────────────────
    # 신규이거나 메타데이터가 바뀐 행만 남기기
    def has_changed(row):
        code = str(row["상품코드"])
        old = existing_meta.get(code)
        # ① 완전 신규
        if old is None:
            return True
        # ② 텍스트가 변경됐을 때
        if old["text"] != row["__text"]:
            return True
        # ③ 각 메타필드가 변경됐을 때
        for kor, eng in column_map.items():
            if eng == "product_code":
                continue
            if eng in numeric_fields:
                if int(old.get(eng, -1)) != int(row[kor]):
                    return True
            else:
                if str(old.get(eng, "")) != str(row[kor]):
                    return True
        # 모두 같으면 변경 없음
        return False

    mask    = df.apply(has_changed, axis=1)
    proc_df = df[mask]
    to_keep = df[~mask]
    
    print(f"🙅‍♀️ 건너뛴 행 수: {len(df) - len(proc_df)}개")  # 여기에!


    if proc_df.empty:
        print(f"⚡ {os.path.basename(path)}: 변경된 항목 없음, 업서트 생략")
        continue

    #변경된 행만 임베딩 생성
    texts = proc_df["__text"].astype(str).tolist()
    batches = split_batches(texts, max_tokens=MAX_TOKENS)
    embeddings = []
    for idx, batch in enumerate(batches, 1):
        clean = [t for t in batch if t.strip() and t.lower() not in ("nan","none")]
        embs  = try_batch(clean)
        embeddings.extend(embs)
        print(f"✅ 배치 {idx}/{len(batches)} 임베딩 완료 (문장수={len(clean)})")
    embeddings = np.array(embeddings)

    # 2) Milvus upsert (2,000개씩)
    total = len(embeddings)
    date_int = int(date.today().strftime("%Y%m%d"))
    codes = proc_df["상품코드"].astype(str).tolist()[:total]
    txts  = proc_df["__text"].tolist()[:total]
    metas = [
        (proc_df[k].astype(int).tolist() if eng in numeric_fields else proc_df[k].astype(str).tolist())[:total]
        for k, eng in list(column_map.items())[1:]
    ]
    dates = [date_int] * total

    CHUNK = 2000
    for s in range(0, total, CHUNK):
        e = min(s + CHUNK, total)
        entities = [
            codes[s:e],
            embeddings[s:e].tolist(),
            txts[s:e],
            *[m[s:e] for m in metas],
            dates[s:e]
        ]
        collection.upsert(entities)
        collection.flush()
        print(f"✅ Milvus 업서트 완료 {s}–{e}/{total}")

    # 3) 30일 이전 데이터 삭제
    threshold = int((date.today() - timedelta(days=30)).strftime("%Y%m%d"))
    collection.delete(f"embedding_date < {threshold}")
    collection.flush()

    print(f"✅ {os.path.basename(path)} 완료 ({time.time()-t0:.1f}s)")
    print(f"✅ Milvus 삽입 완료 ({total}개, {embeddings.shape})")

    print(len(df))       # 전체 행 수
    print(len(proc_df))  # 변경된 행 수
    print(len(embeddings))  # 생성된 임베딩 수


print(f"\n 전체 완료: {time.time()-start_all:.1f}s")

# 검색 함수
def semantic_search_with_price(query, top_k=5):
    collection.load()
    m = re.search(r'(\d+)\s*원', query)
    price = int(m.group(1)) if m else None
    q_emb = client.embeddings.create(input=[query], model=MODEL).data[0].embedding
    expr = f"market_price <= {price}" if price else None
    results = collection.search(
        data=[q_emb], anns_field="emb",
        param={"metric_type":"L2","params":{"nprobe":30}},
        limit=top_k, expr=expr,
        output_fields=list(column_map.values())+['text']
    )
    for hit in results[0]:
        print(f" • {hit.entity.get('market_product_name')} — {hit.entity.get('market_price')}원")

# 예시
if __name__ == "__main__":
    print("캠핑용품 추천해줘")
    semantic_search_with_price("캠핑용품 추천해줘", 5)
    print("==========================")

    print("여름 티셔츠 중 3000원 이하")
    semantic_search_with_price("여름 티셔츠 중 3000원 이하", 5)
    print("==========================")

    print("큰 가방 추천해줘줘")
    semantic_search_with_price("큰 가방 추천해줘줘", 5)
    print("==========================")

    print("여름용 얇은 강아지 옷 추천받고 싶어요")
    semantic_search_with_price("여름용 얇은 강아지 옷 추천받고 싶어요", 5)
    print("==========================")

    print("겨울용품 추천해줘")
    semantic_search_with_price("겨울용품 추천해줘", 5)
    print("==========================")

    print("썬크림 추천해줘")
    semantic_search_with_price("선크림 추천해줘", 5)
    print("==========================")

    print("시원한 여름용품 추천해줘")
    semantic_search_with_price("시원한 여름용품 추천해줘", 5)
    print("==========================")


# # ── 웹 UI ────────────────────────────────────────────────────────
# @app.get("/", response_class=HTMLResponse)
# async def home():
#     return """
#     <html>
#       <head><title>엑셀 업로드</title></head>
#       <body>
#         <h2>여러 개의 엑셀 파일 업로드</h2>
#         <form action="/upload" enctype="multipart/form-data" method="post">
#           <input name="files" type="file" multiple><br/><br/>
#           <button type="submit">업로드 및 처리</button>
#         </form>
#       </body>
#     </html>
#     """

# @app.post("/upload", response_class=HTMLResponse)
# async def upload(files: list[UploadFile] = File(...)):
#     total = 0
#     for file in files:
#         content = await file.read()
#         df = pd.read_excel(BytesIO(content), header=1)
#         cnt = process_df(df)
#         total += cnt
#     return f"<h3>처리 완료: 총 {total}건 업서트되었습니다 🎉</h3><a href='/'>돌아가기</a>"