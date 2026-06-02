####################################################
# 요약 모델 적용 : embedding + qdrant + Summary Model #
####################################################

import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
from qdrant_client import QdrantClient
import logging
import os
import re
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
# logger.info(f"PyTorch Version : {torch.__version__}, Device : {device}")

# embedding(sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
# 기반 구조 : Microsoft의 MiniLM을 기반으로 한 경량 Transformer 모델
# 임베딩 크기 : 384차원 벡터로 문장르 표현 -> 빠르고 메모리 효율적
# 환용 분야 : 문장 유사도 계산, 의미 기반 검색(Semantic Search), RAG에서 질의와 문서 매칭, 클러스터링, 추천 시스템
embedder_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedder_model = SentenceTransformer(embedder_name)

# 모델 로드(gogamza/kobart-summarization)
# Facebook BART를 한국어에 맞게 변형한 KoBart 기반 모델
summarizer_name = "gogamza/kobart-summarization"
summarizer_tokenizer = PreTrainedTokenizerFast.from_pretrained(summarizer_name)
summarizer_model = BartForConditionalGeneration.from_pretrained(summarizer_name)

# Qdrant 서버 연결
qdrant = QdrantClient(host="localhost", port=6333)


# embedding 함수
def generate_embedding(text: str):
    embedding = embedder_model.encode(text).tolist() # 임베딩 벡터 변환
    return embedding

# qdrant 검색 함수
def search_qdrant(query: str, top_k: int = 5):
    query_vector = generate_embedding(query) # 질의 임베딩 생성

    # Qdrant 검색
    qdrant_result = qdrant.query_points(
        collection_name="news_articles",
        query=query_vector,
        limit=top_k
    )
    # query_result : QueryResponse 객체, 검색 결과를 감싸는 객체
    # query_result.points : ScoredPoint 리스트 실제 결과는 query_result.points에 들어 있는 ScoredPoint 리스트, score, .payload 등을 꺼내 쓰면 된다
    return qdrant_result.points

# 요약 함수: 질문 + 문서 합친 context → 모델이 질문을 중심으로 맥락을 재구성 → 잡음이 줄고 핵심 요약 생성 
# 여러 문서를 요약하려면:
# 문서들을 합쳐서 하나의 문자열(" ".join(...))로 만든 뒤 전달
# 질문과 문서를 합쳐서 "질문: ...\n답변: ..." 형태로 전달 --> context
def summarize_text(context, max_length=50, min_length=10):
    # 요약 토크나이저
    inputs = summarizer_tokenizer(
        [context],
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    # logging.info(f"요약 토크나이저 결과: {inputs}")
    # logging.info(f"max_length, min_length: {max_length}, {min_length}")


    # 요약 생성
    summary_ids = summarizer_model.generate(
        # 요약 기본 옵션
        inputs["input_ids"], # 토크나이저가 변환한 입력 문장 토큰, 모델이 요약을 생성할때 조건으로 사용
        num_beams=4, # Beam Search 탐색 개수(값이 클수록 다양한 후보를 탐색해 더 좋은 요약을 얻을수 있으나 속도가 느려진다)
        max_length=max_length, # 생성할 요약의 최대 토큰 길이
        min_length=min_length, # 생성할 요약의 최소 토큰 길이
        early_stopping=True, # 모든 Beam이 EOS(End of Sentence) 토큰에 도달하면 탐색을 조기 종료

        # 반복 결과 제한 옵션
        no_repeat_ngram_size=3, # 특정 길이의 n-gram(연속된 단어 묶음)이 반복되지 않도록 제한
        repetition_penalty=1.5, # 동일 단어를 반복할 경우 확률을 낮추는 패널티 적용(1.2~2.0)
        length_penalty=2.0, # 너무 짧거나 긴 문장을 방지하기 위해 길이에 따른 점수 조정(>1.0 더 긴문장 선호, <1.0 짧은 문장 선호)

        # 창의적, 다양성 옵션
        do_sample=True, # 확률 분포에서 샘플링(결과가 매번 달라 질 수 있고, 창의적/다양한 요약 생성 가능)
        temperature=0.9, # 확률 분포 조정, 1.0(원래 확률 분포 그대로)
        top_p=0.85 # 누적 확률이 85%에 해당하는 상위 토큰 집합에서만 샘플링

        # 보수적, 옵션
        # do_sample=False, # 확률 분포에서 샘플링(결과가 매번 달라 질 수 있고, 창의적/다양한 요약 생성 가능)
        # temperature=1.0, # 확률 분포 조정, 1.0(원래 확률 분포 그대로)
        # top_p=1.0 # 누적 확률이 85%에 해당하는 상위 토큰 집합에서만 샘플링
    )

    # 요약 Decode 한글로 변환
    summary = summarizer_tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary

# 후처리 로직: 불필요한 문자열 제거
def clean_content(content: str):
    content = re.sub(r'(https?://)?\S+\.(com|net|co\.kr)', '', content).strip()
    return content

def clean_answer(answer: str):
    text = answer.strip()
    
    # 질문 문구 제거
    text = text.replace(query, "").strip()

    # 출처명 제거
    text = re.sub(r'(연합뉴스|네이트|전자신문|Google 뉴스|경향신문|포토뉴스|매일경제|KBS 뉴스|조선일보|한겨레)', '', text)

    # URL 제거
    text = re.sub(r'(https?://)?\S+\.(com|net|co\.kr)', '', text).strip()

    # 잡음 패턴 제거
    text = re.sub(r'헤드라인 및 의견 더보기', '', text)
    text = re.sub(r'기사 더보기', '', text)
    text = re.sub(r'신문에서', '', text)
    text = re.sub(r'Google 뉴스에서', '', text)
    text = re.sub(r'(보기|더보기|뉴스|Google|Goog gle|헤드라인)', '', text)
    text = re.sub(r'(^|\s)에서(\s|$)', ' ', text) # 문장 중간의 "서울에서"는 유지, 단독 "에서"는 제거

    # 너무 긴 답변은 앞 200자 까지만 사용
    if len(text) > 200:
        text = text[:200] + "..."

    # 최종 안전장치
    if not text or not text.strip():
        text = "내용없음"

    return text

# FastAPI 서비스
def run_summary(query: str) -> dict:
    # Qdrant 검색 호출
    search_result = search_qdrant(query=query)

    # 검색된 문서들을 후처리
    contexts = [
        {
            "content": clean_content(r.payload.get("content", "")),
            "source": r.payload.get("source_name", "출처 없음"),
            "url": r.payload.get("url", ""),
            "title": r.payload.get("title", ""),
            "score": r.score
        }
        for r in search_result
    ]

    # 문서별 요약 후 병합
    summaries = []
    for doc in contexts:
        if doc["content"].strip(): # content가 공백이 아닐 때만 요약
            doc_summary = summarize_text(context=doc["content"], max_length=80, min_length=30)
            summaries.append(clean_answer(doc_summary))
    
    # 중복 문장 제거
    def deduplicate_sentences(text: str):
        # 문장 단위로 분리 (마침표, 물음표, 느낌표 기준)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        unique_sentences = list(dict.fromkeys([s.strip() for s in sentences if s.strip()]))
        return " ".join(unique_sentences)

    answer = deduplicate_sentences("\n".join(summaries)) if summaries else "내용 없음"
    logging.info(f"answer: {answer}")

    # 출처 매핑
    summary_sources = [ # 데이터 타입: list
        {
            "title": doc["title"],
            "url": doc["url"],
            "source_name": doc["source"],
            "score": float(doc["score"]) if doc["score"] else 0.0
        }
        for doc in contexts
    ]

    return {
        "answer": answer, # 문자열 반환
        "sources": summary_sources # 문서별 dict 리스트 반환
    }


# Qdrant 검색
query = "이란 전쟁 상황에 대해 알려줘"
search_result = search_qdrant(query=query)
# for r in search_result:
#     logger.info(
#         f"Score: {r.score:.4f}, "
#         f"Title: {r.payload.get('title')}, "
#         f"Source: {r.payload.get('source_name')}, "
#         f"URL: {r.payload.get('url')}, "
#         f"Content: {r.payload.get('content')}"
#     )

# 검색된 문서들을 context로 합치기
contexts = [
    {
        "content": clean_content(r.payload.get("content", "")),
        "source": r.payload.get("source_name", "출처 업음"),
        "url": r.payload.get("url", "")
    }
    for r in search_result
]

# 여러 문서들을 하나의 문자열로 합치기
context = " ".join(doc["content"] for doc in contexts)
# logger.info(f"여러 문서들을 하나의 문자열로: {context}")

# 요약: 검색문서들을 요약
summary = summarize_text(context=context, max_length=100, min_length=40)
# logger.info(f"검색문서들을 요약: {summary}")

# 요약: 요약 문서들을 후처리
summary = clean_answer(summary)

# 출처 정보
source_info = [doc.get("source", "출처 없음") for doc in contexts]

# URL 정보
url_info = [doc.get("url", "") for doc in contexts]

# 출처 URL 정보 병합
source_url_info = [
    f"{doc.get('source', '출처 없음')} ({doc.get('url', '')})"
    for doc in contexts
]

# logger.info(f"검색 문서들을 요약: {summary}\n\n출처 정보: {', '.join(source_url_info)}")

# 요약: 검색 질의문 + 검색 결과
combined_input = f"질문: {query}\n답변: {context}"
qdrant_summary = summarize_text(context=combined_input, max_length=150, min_length=50)
qdrant_summary = clean_answer(qdrant_summary)
# logger.info(f"검색 질의문 + 검색 결과들을 요약: {qdrant_summary}\n\n출처 정보: {', '.join(source_url_info)}")

# 메모리 정리: 객체 삭제, 메모리 반환
# del summarizer_tokenizer
# del summarizer_model
# del embedder_model

gc.collect()
# GPU cuda 사용시 메모리 반환
if device=="cuda":
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()