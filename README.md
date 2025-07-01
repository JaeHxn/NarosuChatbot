# NarosuChatbot

# ngork은  내 로컬 서버(로컬호스트)를 외부에서 접근할 수 있도록 해주는 터널링 서비스
# 사설 네트워크 내부에서도 외부에서 접속 가능.
# ngrok 은 설치후 설치된 ngrok 창에서 아래 명령어 입력
# ngrok http --url=satyr-inviting-quetzal.ngrok-free.app 5050
(satyr-inviting-quetzal.ngrok-free.app은 ngrok의 개인 도메인주소)

# ngrok http --url=viable-shark-faithful.ngrok-free.app 5050
# (선오 ngork 개인 도메인주소)

# Python 은 3.8.20


column_map = {
    "상품코드":        "product_code",    
    "카테고리코드":    "category_code",   
    "카테고리명":      "category_name",
    "마켓상품명":      "market_product_name",
    "마켓실제판매가":  "market_price",          #숫자
    "배송비":          "shipping_fee",          #숫자
    "배송유형":        "shipping_type",
    "최대구매수량":    "max_quantity",           #숫자
    "조합형옵션":      "composite_options",
    "이미지중":        "image_url",        
    "제작/수입사":     "manufacturer",
    "모델명":          "model_name",
    "원산지":          "origin",
    "키워드":          "keywords",
    "본문상세설명":    "description",       
    "반품배송비":      "return_shipping_fee",    #숫자
    "독립형":          "independent_option",  
    "조합형":          "composite_flag"       
}
총 18개 정제

#임베딩용 열은 [카테고리명,마켓상품명,조합형옵션] 총 4개만 사용.

#숫자 열은 임베딩에 들어 가지 않기 때문에 따로 
numeric_fields = {
    "market_price",
    "shipping_fee",
    "max_quantity",
    "return_shipping_fee"} 로 제외시켜 임베딩 시킴.


오너클랜 오리지널 엑셀에서 1차 전처리 과정을 거쳐 18개 속성을 사용 그 중 "market_price", "shipping_fee", "max_quantity", "return_shipping_fee" 항목은 숫자로 직접 임베딩에 들어 가지는 않음.(원래 임베딩이 숫자는 임베딩 하지 않음.)

최종적으로는
#임베딩용 텍스트 생성
def make_text(row):
    return " || ".join([
        f"cat:{row['카테고리명']}",
        f"name:{row['마켓상품명']}",
        f"opts:{row['조합형옵션']}"
    ])

코드를 보면 카테고리명,마켓상품명,조합형옵션 총 3개의 옵션만 진정으로 임베딩을 실시하여 벡터DB 에 벡터값으로 들어감.(기존 17개 열로 임베딩을 진행했는데 열이 많고 데이터가 너무 길다 보니 벡터검색할때 제대로 된 검색을 하지 못해서 열을 최대한으로 줄여서 진행함. 벡터 검색이 너무 안나올시 추후에 열을 다시 조절 할 필요도 있음)
