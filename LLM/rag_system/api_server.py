##################################
# FastAPI/Flask 기반 REST API 서버 #
##################################

from fastapi import FastAPI    # 웹 서버 프레임워크
from pydantic import BaseModel # 요청/응답 데이터 구조 정의
import uvicorn # ASGI 서버, FastAPI 실행 엔진
from qdrant_search import run_search # Qdrant 의미 기반 검색
from qa_model import run_qa # QA 모델 답변
from summary_model import run_summary # 요약 모델 답변
from typing import Optional

# 객체 타입 리스트 정의
class Source(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    source_name: Optional[str] = None
    score: Optional[float] = None

# 3.10 이상 버전 코드
# class Source(BaseModel):
#     title: str | None = None
#     url: str | None = None
#     source_name: str | None = None
#     score: float | None = None

# 요청/응답 데이터 모델 정의: 클라이언트가 보내는 요청(JSON)구조, {"query": "..."}
class QueryRequest(BaseModel):
    query: str

# 요청/응답 데이터 모델 정의: 서버가 반환하는 요청(JSON)구조, {"answer": "...", "sources": []"..."]}
class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = [] # 응답 반환시 dict -> Source 객체 매핑
    # sources: list[str] # 문자열만 사용시

# FastAPI 앱 생성: 앱 이름과 버전을 지정, Swagger UI 자동 제공(http://localhost:8000/docs)
app = FastAPI(title="RAG System API", version="1.0.0")

# 앤드포인트: 검색(Qdrant): qdrant_search.py 호출, 의미 기반 검색 결과 반환
# /search: Qdrant 검색 결과 반환 예정
@app.post("/search", response_model=QueryResponse)
def search_endpoint(request: QueryRequest):
    # qdrant_search.py 내에 run_search() 함수 실행
    result = run_search(request.query)
    # result["sources"] 사용 가능하나, 검색 결과가 비어있거나 "sources" 키가 누락될 가능성을 고려하면 get(..., []) 방식이 더 견고하다
    return QueryResponse(answer=result["answer"], sources=result.get("sources", []))

# 앤드포인트: QA 모델: qa_model.py 호출, 질의응답 결과 반화
# /qa: QA 모델 결과 반환 예정
@app.post("/qa", response_model=QueryResponse)
def qa_endpoint(request: QueryRequest):
    # qa_model.py 내에 run_qa() 함수 실행
    result = run_qa(request.query)
    # result["sources"] 사용 가능하나, 검색 결과가 비어있거나 "sources" 키가 누락될 가능성을 고려하면 get(..., []) 방식이 더 견고하다
    return QueryResponse(answer=result["answer"], sources=result.get("sources", []))

# 앤드포인트: 요약 모델: summary_model.py 호출, 요약 결과 반환
# /summary: 요약 모델 결과 반환 에정
@app.post("/summary", response_model=QueryResponse)
def summary_endpoint(request: QueryRequest):
    # summary_model.py 내에 run_summary() 함수 실행
    result = run_summary(request.query)
    # result["sources"] 사용 가능하나, 검색 결과가 비어있거나 "sources" 키가 누락될 가능성을 고려하면 get(..., []) 방식이 더 견고하다
    return QueryResponse(answer=result["answer"], sources=result.get("sources", []))

# 실행 코드: uvicorn 으로 FastAPI 실행, reload=True 코드 수정시 자동 재시작, http://localhost:8000 에서 API 접근 가능
if __name__ == "__main__":
    # uvicorn.run("api_server:app", host="0.0.0.0", port=8000)
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)