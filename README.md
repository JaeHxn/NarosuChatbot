# NarosuChatbot

# ngork은  내 로컬 서버(로컬호스트)를 외부에서 접근할 수 있도록 해주는 터널링 서비스
# 사설 네트워크 내부에서도 외부에서 접속 가능.
# ngrok 은 설치후 설치된 ngrok 창에서 아래 명령어 입력
# ngrok http --url=satyr-inviting-quetzal.ngrok-free.app 5050
(satyr-inviting-quetzal.ngrok-free.app은 ngrok의 개인 도메인주소)

# ngrok http --url=viable-shark-faithful.ngrok-free.app 5050
# (선오 ngork 개인 도메인주소)

# Python 은 3.8.20


#[카테고리코드	카테고리명	마켓상품명	마켓실제판매가	배송비	배송유형	최대구매수량	조합형옵션	이미지중	제작/수입사	모델명	원산지	키워드	본문상세설명	반품배송비	독립형	조합형]
 ->총 17개 정제

#임베딩용 열은 [카테고리명,마켓상품명,키워드,조합형옵션] 총 4개만 사용.

#숫자 열은 임베딩에 들어 가지 않기 때문에 따로 
numeric_fields = {
    "market_price",
    "shipping_fee",
    "max_quantity",
    "return_shipping_fee"} 로 제외시켜 임베딩 시킴.
