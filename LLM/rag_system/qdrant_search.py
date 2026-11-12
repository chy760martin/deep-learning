#####################
# Qdrant 검색(Query) #
#####################
import torch
from sentence_transformers import SentenceTransformer # Hugging Face sentence-transformer 라이브러리 사용
from qdrant_client import QdrantClient
import logging
import os
import gc

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

# 디바이스 설정
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# logger.info(f"Pytorch Version : {torch.__version__}, Device : {device}")

# 다국어 임베딩 모델 로드
embedder_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Qdrant 서버 연결
qdrant = QdrantClient(host="localhost", port=6333)

# 임베딩 함수 생성
def generate_embedding(text: str):
    # 본문(content)을 벡터로 변환
    embedding = embedder_model.encode(text)
    return embedding.tolist() # Qdrant에 넣기 위해 list 변환

# Qdrant 검색 함수
def search_qdrant(query: str, top_k: int = 5):
    # 질의 임베딩 생성
    query_vector = generate_embedding(query)

    # qdrant에서 검색
    result = qdrant.query_points(
        collection_name="news_articles",
        query=query_vector,
        limit=top_k
    )
    # result → QueryResponse 객체, 검색 결과를 감싸는 객체
    # result.points → ScoredPoint 리스트 : 실제 결과는 result.points에 들어 있는 ScoredPoint 리스트, score, .payload 등을 꺼내 쓰면 된다
    return result.points # ScoredPoint 리스트 반환

# qdrant 검색 함수 호출
search_result = search_qdrant("이란 전쟁 상황 알려줘")

# 결과 출력
# for r in search_result:
#     logger.info(
#         f"Score : {r.score:.4f}, Title : {r.payload.get('title')}, "
#         f"Source : {r.payload.get('source_name')} , URL : {r.payload.get('url')} "
#     )

# FastAPI 서비스
def run_search(query: str) -> dict: # 딕셔너리 데이터 형태로 값을 반환
    # Qdrant 검색 로직
    search_result = search_qdrant(query=query)

    # 검새 결과 반환값 정리
    answers = []
    sources = []
    for r in search_result:
        answers.append(r.payload.get("content", "")) # 본문 내용
        sources.append({
            "title": r.payload.get("title"),
            "url": r.payload.get("url"),
            "source_name": r.payload.get("source_name"),
            "score": r.score
        })

    return {
        "answer": " ".join(answers[:1]), # 가장 관련 높은 문서 내용 반환
        "sources": sources
    }

# 메모리 정리: 객체 삭제, 메모리 반환
# del embedder_model

gc.collect()
# GPU cuda 사용시 메모리 반환
if device=="cuda":
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()