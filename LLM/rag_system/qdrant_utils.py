#########################
# Qdrant Collection 생성 #
#########################
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import logging
import os

# 로깅 설정: 파일 + 콘솔 출력
log_dir = "/Users/ai/deep-learning/LLM/rag_system/logs"
os.makedirs(log_dir, exist_ok=True) # 로그 디렉토리 확인 및 생성

logging.basicConfig(
    level=logging.INFO, # INFO 이상 레벨 기록
    format="%(asctime)s [%(levelname)s] %(message)s", # 출력 형식
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "rag_app.log"), encoding="utf-8"), # 파일 저장
        logging.StreamHandler() # 콘솔 출력
    ]
)
logger = logging.getLogger(__name__)

# Qdrant 연결
qdrant = QdrantClient(host="localhost", port=6333)

# Qdrant Collecton 생성
if not qdrant.collection_exists("news_articles"):
    qdrant.create_collection(
        collection_name="news_articles", # 컬렉션 명
        vectors_config=VectorParams( # 벡터 설정 정의
            size=384, # 모델 MiniLM-L12-v2 임베딩 차원 수(384차원)
            distance=Distance.COSINE # 벡터 유사도 계산 방식(COSINE 추천, 의미 유사도 검색에 가장 많이 사용), DOT(내적 기반), EUCLID(거기 기반)
        )
    )
    logger.info(f"Qdrant Collection 'news_articles' 생성 완료")
else:
    logger.info(f"Qdrant Collection 'news_articles' 존재 합니다.")