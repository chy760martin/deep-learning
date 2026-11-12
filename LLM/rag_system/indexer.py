##################
# 색인용 데이터 추출 #
##################
import psycopg2
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

# DB 연결
conn = psycopg2.connect(
    dbname="newsdb", 
    user="newsuser", 
    password="1234", 
    host="localhost", 
    port="5432"
)

# cursor 객체 생성
cur = conn.cursor()

try:
    # 색인용 데이터 조회, 최근 데이터 100개 기준
    cur.execute(""" 
                SELECT id, title, content, url, published_at, source_name
                FROM news_articles
                ORDER BY published_at DESC
                LIMIT 100;
    """)
    articles = cur.fetchall()
    logger.info("색인용 데이터 %s개 가져옴:", len(articles))

    # Qdrant에 넣을 준비: 리스트 형태로 변환
    index_data = []
    for row in articles:
        record = {
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "url": row[3],
            "published_at": row[4],
            "source_name": row[5]
        }
        index_data.append(record)

except Exception as e:
    logger.error("색인용 조회 에러 발생: %s", e)

# DB connect 종료
cur.close()
conn.close()
logger.info("색인 데이터 조회 완료 및 DB 연결 종료")


############################################
# Qdrant news_articles 컬렉션 update/insert #
############################################
import torch
from sentence_transformers import SentenceTransformer # Hugging Face sentence-transformer 라이브러리 사용
from qdrant_client import QdrantClient
from qdrant_client.http import models # qdrant 컬렉션 및 검색 관련 설정 정의하는 데이터 모델(구조체) 로드

# 디바이스 설정
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"Pytorch Version : {torch.__version__}, Device : {device}")

# 다국어 임베딩 모델 로드
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Qdrant 서버 연결
qdrant = QdrantClient(host="localhost", port=6333)

# 임베딩 함수 생성
def generate_embedding(text: str):
    # 본문(content)을 벡터로 변환
    embedding = model.encode(text)
    return embedding.tolist() # Qdrant에 넣기 위해 list 변환

# 데이터 저장 형식 설명
# qdrant.upsert()는 삽입(insert)+업데이트(update) 기능을 동시에 수행, 같은 ID가 있으면 덮어쓰고, 없으면 새로 추가한다
    # { 
    #     "id": 1,
    #     "vector": [0.123, -0.456, 0.789, ...],
    #     "payload": {
    #         "title": "AI 의료 활용",
    #         "content": "AI가 의료 분야에서 활용되는 사례...",
    #         "date": "2026-03-11",
    #         "author": "홍길동"
    #     }
    # }
# Qdrant update/insert 함수
def insert_to_qdrant(data):
    ids = []
    vectors = []
    payloads = []
    
    for item in data:
        # 본문만 임베딩
        text = item["content"]
        embedding = generate_embedding(text=text) # 임베딩 함수 호출

        ids.append(item["id"]) # ID 값
        vectors.append(embedding) # 임베딩 처리 후 벡터 값
        payloads.append(item) # 원본 데이터
    
    qdrant.upsert(
        collection_name="news_articles",
        points=models.Batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads # paylods(JSON) 형태로 저장
        )
    )
    logger.info("본문 임베딩 및 Qdrant 저장 완료")
    

# Batch 단위로 Qdrant update/insert
batch_size = 30
total = len(index_data)

for i in range(0, len(index_data), batch_size):
    chunk = index_data[i : i + batch_size] # 2개씩 잘라서 가져온다
    insert_to_qdrant(chunk) # 잘라낸 청크를 Qdrant에 전달

    # 진행 상황 계산
    start_id = i
    end_id = i + len(chunk) - 1
    processed = end_id + 1
    progress = (processed / total) * 100
    remaining = total - processed

    logger.info(
        f"Batch {start_id} ~ {end_id} 인덱싱 완료" 
        f"(총 {len(chunk)}건), 진행률 : {progress:.2f}%, 남은 데이터 : {remaining}건"
    )

# 최종 데이터 개수 확인
count_result = qdrant.count(collection_name="news_articles")
logger.info(f"Qdrant news_articles 컬렉션 최종 저장 건수 : {count_result}건")