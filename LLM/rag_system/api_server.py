##################################
# FastAPI/Flask 기반 REST API 서버 #
##################################

from fastapi import FastAPI    # 웹 서버 프레임워크
from pydantic import BaseModel # 요청/응답 데이터 구조 정의
import uvicorn # ASGI 서버, FastAPI 실행 엔진

# 요청/응답 데이터 모델 정의: 클라이언트가 보내는 요청(JSON)구조, {"query": "..."}
class QueryRequest(BaseModel):
    query: str
# 요청/응답 데이터 모델 정의: 서버가 반환하는 요청(JSON)구조, {"answer": "...", "sources": []"..."]}
class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# FastAPI 앱 생성: 앱 이름과 버전을 지정, Swagger UI 자동 제공(http://localhost:8000/docs)
app = FastAPI(title="RAG System API", version="1.0.0")

# 앤드포인트: 검색(Qdrant): qdrant_search.py 호출, 의미 기반 검색 결과 반환
# /search: Qdrant 검색 결과 반환 예정
@app.post("/search", response_model=QueryResponse)
def search_endpoint(request: QueryRequest):
    # TODO: qdrant_search.py 불러와서 실행
    answer = f"Search 결과: {request.query}"
    sources = ["출처 예시"]
    return QueryResponse(answer=answer, sources=sources)

# 앤드포인트: QA 모델: qa_model.py 호출, 질의응답 결과 반화
# /qa: QA 모델 결과 반환 예정
@app.post("/qa", response_model=QueryResponse)
def qa_endpoint(request: QueryRequest):
    # TODO: qa_model.py 불러와서 실행
    answer = f"QA 결과: {request.query}"
    sources = ["출처 예시"]
    return QueryResponse(answer=answer, sources=sources)

# 앤드포인트: 요약 모델: summary_model.py 호출, 요약 결과 반환
# /summary: 요약 모델 결과 반환 에정
@app.post("/summary", response_model=QueryResponse)
def summary_endpoint(request: QueryRequest):
    answer = f"Summary 결과: {request.query}"
    sources = ["출처 예시"]
    return QueryResponse(answer=answer, sources=sources)

# 실행 코드: uvicorn 으로 FastAPI 실행, reload=True 코드 수정시 자동 재시작, http://localhost:8000 에서 API 접근 가능
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)