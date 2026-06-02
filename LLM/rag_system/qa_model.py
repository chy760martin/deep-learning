##############################################
# QA 모델 적용 : embedding + qdrant + QA Model #
##############################################

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import logging
import os
import gc
import re

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

# embedding
embedder_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedder_model = SentenceTransformer(embedder_name)

# KorQuAD (Korean Question Answering Dataset) : 한국어 위키 문서를 기반으로 만든 QA 데이터셋으로 질문과 답변이 문맥 내에서 매핑되어 있다
qa_model_name = "monologg/koelectra-base-v3-finetuned-korquad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name) # QA Tokenizer
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name) # QA Model

# Qdrant 서버 연결
qdrant = QdrantClient(host="localhost", port=6333)

# embedding 함수
def generate_embedding(text: str):
    embedding = embedder_model.encode(text) # 임베딩 벡터 변환
    return embedding.tolist()

# qdrant 검색 함수
def search_qdrant(query: str, top_k: int = 5):
    # 질의 임베딩 생성
    query_vector = generate_embedding(query)

    # qdrant 검색
    query_result = qdrant.query_points(
        collection_name="news_articles",
        query=query_vector,
        limit=top_k
    )
    # query_result : QueryResponse 객체, 검색 결과를 감싸는 객체
    # query_result.points : ScoredPoint 리스트 실제 결과는 query_result.points에 들어 있는 ScoredPoint 리스트, score, .payload 등을 꺼내 쓰면 된다
    return query_result.points

# QA 모델 파이프라인: 
# QA 토크나이저 > QA 모델 > 답변 시작/끝 위치 > 토큰 ID를 토큰 문자열 변환 > 특수 토큰 제거 > 토큰을 문자열로 변환
def qa_pipeline(question, context, top_k=3, max_answer_len=100):
    # 토크나이징: 질문과 문서를 토큰 ID로 변환
    inputs = qa_tokenizer(
        question, # 질문
        context, # 문서
        return_tensors="pt", # 파이토치 형태로 리턴
        max_length=512, # 입력 최대 길이
        truncation=True, # 길면 잘라내기
        padding="max_length"
    ).to(device)

    # QA 모델 추론
    with torch.no_grad():
        # QA 모델 추론
        outputs = qa_model(**inputs) # **inputs 파라미터 형태로 입력

    # 확률 분포 계산
    # outputs.start_logits 답변이 여기서 시작할 확률(logit 확률로 변환하기 전 값)
    # dim=-1 마지막 차원(토큰 차원)을 기준으로 softmax() 적용
    # 결과적으로 각 토큰 위치에서 대해 0~1 사이의 확률 값이 나오고, 전체 합은 1이 된다
    start_probs = torch.nn.functional.softmax(outputs.start_logits, dim=-1) # 각 토큰이 답변 시작 위치일 확률 분포
    end_probs = torch.nn.functional.softmax(outputs.end_logits, dim=-1) # 각 토큰이 답변 끝 위치일 확률 분포

    # 상위 후보 추출: top_k 개수 만큼 후보(복원) 추출
    start_top = torch.topk(start_probs, k=top_k)
    end_top = torch.topk(end_probs, k=top_k)

    # QA모델이 예측한 시작/끝 위치 확률에서 실제 답변 span 복원 후보를 뽑는 과정
    answers = []
    for i in range(top_k):
        for j in range(top_k):
            # start_top.indices[0][i].item() 답변 시작 위치 후보 중 i번째 토큰 인덱스
            start_idx = start_top.indices[0][i].item()
            # end_top.indices[0][j].item() + 1 답변 끝 위치 후보 중 j번째 토큰 인덱스
            end_idx = end_top.indices[0][j].item() + 1 # 실제 끝 토큰까지 포함하기 위해 +1 해준다

            # end_idx <= start_idx 끝 위치가 시작보다 앞에 있으면 잘못된 span -> 무시
            # end_idx - start_idx > max_answer_len 답변 길이가 너무 길면 -> 무시
            if end_idx <= start_idx or end_idx - start_idx > max_answer_len:
                continue

            # 선택된 토큰 구간(start_idx:end_idx)을 실제 문자열로 반환
            # skip_special_tokens=True [CLS],[SEP],[PAD] 같은 특수 토큰을 제거
            answer = qa_tokenizer.decode(
                inputs["input_ids"][0][start_idx:end_idx],
                skip_special_tokens=True
            )

            # 시작 위치 확률 x 끝 위치 확률 계산 -> span(복원)의 신뢰도
            score = start_top.values[0][i].item() * end_top.values[0][j].item()

            # (답변 문자열, 점수) 형태로 후보 리스트에 추가한다
            answers.append((answer, score))

    # 확률 점수가 높은 순으로 정렬
    answers = sorted(answers, key=lambda x: x[1], reverse=True)

    return answers

# 후처리 로직: 불필요한 문자열 제거
def clean_content(content: str):
    content = re.sub(r'(https?://)?\S+\.(com|net|co\.kr)', '', content)
    return content.strip()

def clean_answer(answer: str):
    text = answer.strip()
    
    # 질문 문구 제거
    text = text.replace(query, "").strip()

    # 출처명 제거
    text = re.sub(r'(연합뉴스|네이트|전자신문|Google 뉴스|경향신문|포토뉴스|매일경제|KBS 뉴스|조선일보|한겨레)', '', text)

    # URL 제거
    text = re.sub(r'(https?://)?\S+\.(com|net|co\.kr)', '', text).strip()

    # 너무 긴 답변은 앞 200자 까지만 사용
    if len(text) > 200:
        text = text[:200] + "..."

    # 최종 안전장치
    if not text or not text.strip():
        text = "내용없음"

    return text

# FastAPI 서비스
def run_qa(query: str) -> dict: # 딕셔너리 데이터 형태로 값을 반환
    # Qdrant 검색 호출
    search_result = search_qdrant(query)

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

    # 여러 문서들을 하나의 문자열로 합친다
    context = " ".join(doc["content"] for doc in contexts)

    # QA 모델: 질문(question)/문서(context)를 입력으로 받아서 답변을 복원 또는 생성
    # Qdrant 의미 기반 검색 -> QA 모델 답변 생성 + 출처 정보 + URL 정보
    qa_answers = []
    qa_sources = []
    for i, doc in enumerate(contexts):
        # qa 모델 파이프라인(질문, 문서), 대상 문서에 따라서 top_k 후보를 20까지 늘려 복원의 다양성 및 품질 개선
        ans = qa_pipeline(question=query, context=doc["content"], top_k=5)

        if ans:
            # top_k 만큼의 후보내에서 필터링: 최소 3글자 이상, 질문과 동일하지 않은 답변
            valid_answers = [
                a for a, s in ans 
                if len(a.strip()) >= 3 # 최소 3글자 이상
                and query.strip() not in a.strip() # 답변에 질문 문장이 포함되지 않음
                and not a.strip().startswith(query.strip()) # 답변이 질문으로 시작하지 않음
            ]
            # logging.info(f"valid_answers : {valid_answers}")

            if not valid_answers:
                # fallback: 검색 문서 첫 문장
                best_answer = doc["content"].split(".")[0].strip()
            else:
                # 가장 긴 답변 선택
                best_answer = max(valid_answers, key=len)
            
            # 최종 안전장치
            if not best_answer or not best_answer.strip():
                best_answer = "내용 없음"
            
            # 출처 정보 병합
            source_info = doc.get("source", "출처 없음")

            # URL 정보 병합
            url_info = doc.get("url", "")

            # Title 정보 병합
            title_info = doc.get("title", "")

            # Score 정보 병합
            score_info = doc.get("score", "")
            
            qa_answers.append(f"{i+1} {clean_answer(answer=best_answer)}")

            # qa_answers.append(f"{i+1} {clean_answer(answer=best_answer)} (출처: {source_info}) (URL: {url_info})")

            qa_sources.append({
                "title": title_info,
                "url": url_info,
                "source_name": source_info,
                "score": score_info
            })

    # 최종 답변 선택
    if qa_answers:
        final_qa_answers = qa_answers
    else:
        # QA 모델 답변 실패시 검색 문서 전체를 fallback으로 사용
        final_qa_answers = context

    # 반환값 리스트 문자열 구분
    if isinstance(final_qa_answers, list):
        # 리스트일 경우 -> 문자열을 합친다
        answer = " ".join(final_qa_answers) # 리스트 -> 문자열 변환
    else:
        # 문자열일 경우 그대로 사용
        answer = final_qa_answers
    
    sources = qa_sources
    return {
        "answer": answer,
        "sources": sources 
    }

# qdrant 검색 결과
query = "이란 전쟁 상황에 대해 알려줘"
search_result = search_qdrant(query)
# for r in search_result:
#     logger.info(
#         f"Score: {r.score:.4f}, "
#         f"Title: {r.payload.get('title')}, "
#         f"Source: {r.payload.get('source_name')}, "
#         f"URL: {r.payload.get('url')}, "
#         f"Content: {r.payload.get('content')}"
#     )

# 검색된 문서들을 후처리
contexts = [
    {
        "content": clean_content(r.payload.get("content", "")),
        "source": r.payload.get("source_name", "출처 없음"),
        "url": r.payload.get("url", ""),
        "title": r.payload.get("title", ""),
        "score": r.payload.get("score", "")
    }
    for r in search_result
]

# 여러 문서들을 하나의 문자열로 합친다
context = " ".join(doc["content"] for doc in contexts)
# logger.info(f"context : {context}")

# QA 모델: 질문(question)/문서(context)를 입력으로 받아서 답변을 복원 또는 생성
# Qdrant 의미 기반 검색 -> QA 모델 답변 생성 + 출처 정보 + URL 정보
qa_answers = []
qa_sources = []
for i, doc in enumerate(contexts):
    # qa 모델 파이프라인(질문, 문서), 대상 문서에 따라서 top_k 후보를 20까지 늘려 복원의 다양성 및 품질 개선
    ans = qa_pipeline(question=query, context=doc["content"], top_k=5)

    if ans:
        # top_k 만큼의 후보내에서 필터링: 최소 3글자 이상, 질문과 동일하지 않은 답변
        valid_answers = [
            a for a, s in ans 
            if len(a.strip()) >= 3 # 최소 3글자 이상
            and query.strip() not in a.strip() # 답변에 질문 문장이 포함되지 않음
            and not a.strip().startswith(query.strip()) # 답변이 질문으로 시작하지 않음
        ]
        # logging.info(f"valid_answers : {valid_answers}")

        if not valid_answers:
            # fallback: 검색 문서 첫 문장
            best_answer = doc["content"].split(".")[0].strip()
        else:
            # 가장 긴 답변 선택
            best_answer = max(valid_answers, key=len)
        
        # 최종 안전장치
        if not best_answer or not best_answer.strip():
            best_answer = "내용 없음"
        
        # 출처 정보 병합
        source_info = doc.get("source", "출처 없음")

        # URL 정보 병합
        url_info = doc.get("url", "")

        # Title 정보 병합
        title_info = doc.get("title", "")

        # Score 정보 병합
        score_info = doc.get("score", "")
        
        qa_answers.append(f"{i+1} {clean_answer(answer=best_answer)}")

        # qa_answers.append(f"{i+1} {clean_answer(answer=best_answer)} (출처: {source_info}) (URL: {url_info})")

        qa_sources.append({
            "title": title_info,
            "url": url_info,
            "source_name": source_info,
            "score": score_info
        })

# logging.info(f"qa_answers : {qa_answers}")

# 최종 답변 선택
if qa_answers:
    final_qa_answers = qa_answers
else:
    # QA 모델 답변 실패시 검색 문서 전체를 fallback으로 사용
    final_qa_answers = context
# logger.info(f"질문 : {query}")
# logger.info(f"답변 : {final_qa_answers}")

# 메모리 정리: 객체 삭제, 메모리 반환
# del qa_tokenizer
# del qa_model
# del embedder_model

gc.collect()
# GPU cuda 사용시 메모리 반환
if device=="cuda":
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()