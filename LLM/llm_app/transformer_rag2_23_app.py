# 서비스: 구글, 네이버 RAG 시스템 구성
# - /llm_app/transformer_rag2_23_app.py
# - FastAPI 구동 정보: 터미널에서 구동, uvicorn transformer_rag2_23_app:app --reload, 경로 포함 uvicorn LLM.llm_app.transformer_rag2_23_app:app --reload
# - FastAPI 서비스: /search, 입력: 질의문, 출력: QA + 요약 결과 + 출처 정보
# - 1)  [사용자 질의]
# - 2)  [FastAPI 엔드포인트: /search]
# - 3)  [SentenceTransformer: 임베딩]
# - 4)  [Qdrant 의미 기반 검색]
# - 5)  [검색 결과 문서]
# - 6)  [KoELECTRA QA 모델 + MeCab 후처리(형태소 분석: 한국어 처리 강화)]
# - 7)  [KoBART Summarization 모델 + clean_summary]
# - 8)  [응답 + 출처 표시]
# - 9)  [최종 응답 LLM 모델은  외부 서비스 연계 검토: 자연스러운 문장]
# - 10) [최종 사용자 응답]

from fastapi import FastAPI # FastAPI
from pydantic import BaseModel # type 지정, api 요청시 자동으로 검증
from typing import List # Python 타입 힌트 모듈, List[str], List[int] 같은 형식으로 리스트 안의 데이터 타입을 지정
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from mecab import MeCab

# 디바이스 설정
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"PyTorch Version: {torch.__version__}, Device: {device}")

# Qdrant 서버 연결
qdrant = QdrantClient(host="localhost", port=6333)

# 임베딩 모델 로드
# - 여기서는 paraphrase-multilingual-MiniLM-L12-v2 모델을 사용했는데, 다국어(한국어 포함) 문장 의미를 잘 반영하는 임베딩을 생성한다
# - 문장을 입력하면 의미 공간에서 가까운 벡터로 변환
# embedder = SentenceTransformer( # 온라인 상태(단, 로컬캐시에 저장)
#     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#     device=device
# )

# 오프라인 사용 준비: 온라인 환경에서 모델 다운로드
# - git lfs install
# - git clone https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
embedder = SentenceTransformer( # 오프라인 상태(경로 지정에 저장)
    "D:/AI/Pytorch/deep-learning/LLM/offline_models/paraphrase-multilingual-MiniLM-L12-v2",
    device=device
)

# QA모델: 특정 도메인으로 파인튜닝된 QA 모델로 교체 검토
qa_model_name = "monologg/koelectra-base-v3-finetuned-korquad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name) # KoELECTRA 토크나이저 로드
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name).to(device)

# QA 모델 파이프라인: 
# - QA 토크나이저->QA 모델 추론->답변 시작/끝 위치->토큰ID를 토큰 문자열 변환->특수 토큰 제거-> 토큰을 문자열로 변환
def qa_pipeline(question, context):
    # 토크나이징: 질문과 문서를 토큰ID 변환
    inputs = qa_tokenizer(
        question, # 질문: 의미 기반 검색 질의
        context, # 문서: 의미 기간 검색 결과 건의 배열 값
        return_tensors="pt"
    ).to(device)

    # 추론 실행
    with torch.no_grad():
        outputs = qa_model(**inputs)

    # 답변 시작/끝 위치 예측하는 확률 분포
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # 시작/끝 위치를 잡을때, 최소/최대 길이를 강제로 설정: 최소 5토큰 이상, 최대 50토큰 이하
    if answer_end - answer_start < 5:
        answer_end = answer_start + 5
    elif answer_end - answer_start > 50:
        answer_end = answer_start + 50

    # 토큰ID -> 토큰 문자열 변환
    tokens = qa_tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0][answer_start:answer_end]
    )
    # 특수 토큰 제거: [CLS],[SEP],[PAD] 특수 토큰이 제외한다
    tokens = [ t for t in tokens if t not in qa_tokenizer.all_special_tokens ]
    # 토큰 -> 최종적으로 사람이 읽을수 있는 문자열 복원
    answer = qa_tokenizer.convert_tokens_to_string(tokens)    
    return answer

# 형태소 분석
mecab = MeCab()
# 후처리 함수: 형태소 분석 포함
def clean_result(text: str) -> str:
    text = text.strip() # 불필요한 공백제거    
    text = text.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "") # 특수문자/토큰 정리

    morphs = mecab.pos(text)
    if not morphs:
        return text
    
    last_word, last_pos = morphs[-1] # 마지막 단어/품사
    # 동사/형용사 처리
    if last_pos.startswith("VV") or last_pos.startswith("VA") or "XSV" in last_pos:
        if not text.endswith("다."):
            if text.endswith("다"):
                text += "."
            else:
                text += "다."
    # 명사 처리
    elif last_pos.startswith("NN"):
        if not text.endswith("이다."):
            text += "이다."
    # 조사로 끝나는 경우 -> 직전 형태소 확인
    elif last_pos.startswith("J"):
        prev_word, prev_pos = morphs[-2]
        if prev_pos.startswith("NN"):
            text += "이다."
        elif prev_pos.startswith("VV") or prev_pos.startswith("VA"):
            text += "다."
    # 기본 처리
    else:
        if not text.endswith("다."):
            if text.endswith("다"):
                text += "."
            else:
                text += "다."
    return text


# 요약 토크나이저 + 요약 모델 로드 , 요약 모델: 특정 도메인으로 파인튜닝된 QA 모델로 교체 검토
summarizer_model_name = "gogamza/kobart-summarization"
summarizer_tokenizer = PreTrainedTokenizerFast.from_pretrained(summarizer_model_name)
summarizer_model = BartForConditionalGeneration.from_pretrained(summarizer_model_name).to(device)

# 요약 함수 생성
def summarize_text(context, max_length=50, min_length=10):
    # 요약 토크나이저
    inputs = summarizer_tokenizer(
        [context],
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # 요약 생성
    summary_ids = summarizer_model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        min_length=min_length,
        early_stopping=True,
        
        no_repeat_ngram_size=3,
        repetition_penalty=2.5,   # 반복 억제 강화
        length_penalty=1.5, # 문장 길이 조절

        do_sample=True,
        temperature=0.9, # 다양성 확보
        top_p=0.85 # 안정성 확보
    )
    
    # 요약 Decode 한글로 변환
    summary = summarizer_tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )
    return clean_summary(summary)

# 후처리 로직: 요약 결과 문자열에서 바로 앞 단어가 반복되는 경우 제거
def clean_summary(summary: str) -> str:
    # 같은 구절 반복 제거
    tokens = summary.split()          # 1. 문자열을 공백 기준으로 단어 단위 분리
    cleaned = []                      # 2. 최종 결과를 담을 리스트 생성
    for t in tokens:                  # 3. 분리된 단어들을 순서대로 확인
        if not cleaned or cleaned[-1] != t:  
            # 4. 리스트가 비어있거나, 마지막에 추가된 단어와 현재 단어가 다르면
            cleaned.append(t)         #    현재 단어를 결과 리스트에 추가
    return " ".join(cleaned)          # 5. 중복 제거된 단어들을 다시 문자열로 합침


# FastAPI
app = FastAPI(title="RAG Search API", description="Qdrant 의미 기반 검색 + QA 모델 + 요약 모델 적용")

# 요청 데이터 구조 정의
class QueryRequest(BaseModel):
    query: str
# 응답 데이터 구조 정의
class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
# 엔드포인트 정의
@app.post("/search", response_model=QueryResponse)
def search_rag(request: QueryRequest):
    query = request.query

    # 1. 질의문 임베딩: Qdrant 의미 기반 검색시 입력값으로 벡터값이 리스트 형태로 입력
    query_vector = embedder.encode(query).tolist()

    # 2. Qdrant 검색
    results = qdrant.query_points(
        collection_name="news_collection",
        query=query_vector,   # 그냥 벡터 리스트를 직접 전달
        limit=3,
        with_payload=True
    )

    # 3. QA 모델 적용
    contexts = [ r.payload["content"] for r in results.points ]
    context = " ".join(contexts)
    
   # 의미 기반 검색 결과의 각 문서마다 복원값(답변) 추출
    qa_answers = []
    for i, doc in enumerate(contexts):
        ans = qa_pipeline(question=query, context=doc) # QA 모델 파이프라인: 건별 복원값
        if ans and ans not in ["[CLS]", ""]:
            qa_answers.append(f"{i+1} {clean_result(ans)}")
    # 최종 답변 선택
    if qa_answers:
        shortest_answer = min(qa_answers, key=len) # 가장 짧은 답변
        longest_answer = max(qa_answers, key=len) # 가장 긴 답변
        # final_qa_answer = {
        #     "shortest": shortest_answer,
        #     "longest": longest_answer,
        #     "all": qa_answers
        # }
        final_qa_answer = qa_answers
    else:
        # QA 모델 답변 실패시 검색 문서 전체를 fallback으로 사용
        final_qa_answer = context

    # 4. 요약 모델 적용
    # 요약: 검색 질의문 + QA 답변 + 검색 결과
    qa_texts = " ".join(final_qa_answer) # 문자열로 합쳐서 입력값으로 전달
    combined_input = f"질문: {query}\n답변: {qa_texts}\n문서: {context}"
    qa_summary = summarize_text(combined_input, max_length=60, min_length=20)

    # 5. 출처 표시
    sources = [ f"ID={r.id}, 제목={r.payload['title']}" for r in results.points ]
    final_answer = f"{qa_summary}\n\n출처: " + "; ".join(sources)

    return QueryResponse(query=query, answer=final_answer, sources=sources)