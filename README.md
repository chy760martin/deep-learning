# Deep Learning & LLM Projects

이 저장소는 딥러닝 및 대형 언어 모델(LLM) 관련 학습과 실험을 기록하기 위해 만들어졌습니다.  
MLP, CNN, RNN, LSTM, GRU 같은 기본 신경망부터 LLM 응용, Transformer RAG 파이프라인까지 다양한 예제를 포함합니다.

---

## 💻 My Tech Stack
<div align="center">
    <img src="https://img.shields.io/badge/java-007396?style=for-the-badge&logo=OpenJDK&logoColor=white">
    <img src="https://img.shields.io/badge/springboot-6DB33F?style=for-the-badge&logo=springboot&logoColor=white">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">
    <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black"/>
</div>

---

## 📂 학습 프로젝트

### 46. Transformer Mathematical Models
- **폴더**: [LLM/mathematical-models](https://github.com/chy760martin/deep-learning/tree/main/LLM/mathematical-models)
- **학습 목표**: 트랜스포머 모델의 핵심 수학적 구조와 단계별 계산 흐름 이해
- **구성 요소**: Linear Layer, MLP, Encoder, Decoder, 최종 Linear, Softmax
  1. **Linear Layer**  
     - 공식: \(y = xW + b\)  
     - 입력 벡터를 선형 변환
  2. **MLP 2계층**  
     - 구조: Linear → ReLU → Linear  
     - 공식: \(h = \max(0, xW_1+b_1), \; y = hW_2+b_2\)
  3. **MLP 3계층**  
     - 구조: Linear → ReLU → Linear → ReLU → Linear  
     - 단계별 활성화와 선형 변환
  4. **Transformer Encoder**
     - 입력(단어토큰) -> 입력 임베딩 + 위치 인코딩
     - Multi-Head Attention
     - Residual Connection + Layer Normalization
     - FFN(Feed Forward Network)
     - 2차 Residual Connection + Layer Normalization
  5. **Transformer Decoder**
     - 출력(Target) -> 출력 임베딩 + 위치 인코딩
     - Masked Multi-Head Attention
     - Residual Connection + Layer Normalization
     - Cross-Attention (인코더 + 디코더 출력 결합)
     - 2차 Residual Connection + Layer Normalization
     - FFN(Feed Forward Network)
     - 3차 Residual Connection + Layer Normalization
  6. **최종 Linear Layer**  
     - 디코더 출력 → 어휘 공간 매핑
  7. **최종 Softmax**  
     - 점수 벡터 → 확률 분포 변환

```mermaid
flowchart TD
    A[입력 토큰] --> B[Encoder]
    B --> C[Decoder]
    C --> D[Linear Layer]
    D --> E[Softmax]
    E --> F[다음 단어 확률 분포]

### 45. Transformer RAG (Qdrant 기반)
- **파일**: `LLM/23.transformer_rag2.ipynb`, `LLM/llm_app/transformer_rag2_23_app.py`
- **학습 목표**: 실무형 RAG 파이프라인 이해 및 적용
- **구성 요소**:
  1. **데이터 수집 및 저장**  
     - PostgreSQL 테이블 생성 및 입력  
     - DB 조회 + 로깅 설정으로 데이터 관리  
  2. **Qdrant 의미 기반 검색 구축**  
     - 뉴스 컬렉션 생성  
     - Qdrant 서버 실행: `.\LLM\qdrant\qdrant.exe`  
     - API 테스트: `curl http://localhost:6333/collections`  
  3. **임베딩 생성 및 삽입**  
     - SentenceTransformer 임베딩 생성  
     - Batch 단위 Insert/Update  
  4. **QA 처리**  
     - KoELECTRA QA 모델 + MeCab 형태소 분석  
     - 입력: `input_ids`, `attention_mask`  
     - 출력: 자연어 응답 복원  
  5. **요약 처리**  
     - KoBART Summarization 모델  
     - 반복 억제, 길이 조절, 다양성 확보  
     - 후처리: `clean_summary`  
  6. **서비스 구성**  
     - FastAPI 실행:  
       ```bash
       uvicorn LLM.llm_app.transformer_rag2_23_app:app --reload
       ```  
     - 엔드포인트: `/search`  
     - 출력: QA + 요약 결과 + 출처 정보  

```
flowchart TD
    A[사용자 질의] --> B[FastAPI 엔드포인트 /search]
    B --> C[SentenceTransformer 임베딩 생성]
    C --> D[Qdrant 의미 기반 검색]
    D --> E[관련 문서 반환]
    E --> F[KoELECTRA QA 모델 + MeCab 형태소 분석]
    F --> G[KoBART Summarization + clean_summary]
    G --> H[응답 + 출처 표시]
    H --> I[외부 LLM 서비스 연계 검토]
    I --> J[최종 사용자 응답]
```

### 44. Transformer RAG (FAISS 기반)
- **파일**: `LLM/22.transformer_rag.ipynb`
- **학습 목표**: 실무형 RAG 파이프라인 이해 및 적용
- **구성 요소**:
  1. **FAISS 메모리 기반 검색엔진 구축**
     - 라이브러리: `faiss`
     - 단일 머신 메모리 기반, 10만 건 미만 소규모 데이터 적합
  2. **QA 처리**
     - 모델: `monologg/koelectra-base-v3-finetuned-korquad`
     - 특정 도메인으로 파인튜닝된 QA 모델 교체 검토
  3. **요약 처리**
     - 모델: `gogamza/kobart-summarization`
     - 토크나이저: `PreTrainedTokenizerFast` (한글 지원)
     - 특정 도메인 요약 모델 교체 검토
  4. **LLM 모델**
     - 현재 로컬 GPU 장비로는 생성형 LLM 모델 파인튜닝 불가
     - 추론 모델 교체도 현재는 어려움

### 43. Transformer Dialogue Chatbot
- **파일**: `LLM/21.transformer_dialogue_chatbot.ipynb`, `LLM/llm_app/transformer_dialogue_chatbot_21_app.py`
- **학습 목표**: 실무형 대화형 챗봇 파이프라인 이해 및 적용
- **구성 요소**:
  1. **모델 분석**
     - Shape 변화 과정 확인
  2. **데이터셋 준비**
     - AI Hub 한국어 SNS 멀티턴 대화 데이터
     - Train/Validation 분리 및 정상 파일 추출
  3. **전처리**
     - JSON 파싱 → 사전 토크나이징 및 저장
     - Dataset 클래스 정의 및 DataLoader 생성
  4. **모델 정의**
     - Feature Extraction + LoRA Fine-tuning 조합
     - 최적화 설정: Optimizer, GradScaler, autocast
     - Early Stopping 클래스 정의 및 최적 모델 가중치 저장
  5. **학습/검증 루프**
     - 딕셔너리 형태 학습데이터를 그대로 모델에 전달
     - Early Stopping 객체 적용
     - AMP `torch.float32` 사용 (메모리 증가, `torch.float16` 사용 시 loss 문제 발생)
  6. **추론 및 서비스**
     - 멀티 답변 생성
     - FastAPI 추론 서비스 실행:
       ```bash
       uvicorn transformer_dialogue_chatbot_21_app:app --reload
       ```
     - 엔드포인트: `http://127.0.0.1:8000/predict`
     - Postman 및 API 코드(Python, Java 등)로 테스트 가능
  7. **추가 검토 사항**
     - 히스토리 관리: `session_id` + 최근 5회 대화 유지
     - 오래된 대화는 요약 후 삭제 (예: “사용자는 여행 관련 질문을 자주 한다”)
     - 메모리 DB 및 벡터 데이터베이스(Vector Store) 활용
     - 하이브리드 전략: 최근 대화는 그대로 유지 + 오래된 대화는 요약/검색으로 관리

### 42. Transformer QA 모델
- **파일**: `LLM/20.transformer_qa.ipynb`, `LLM/llm_app/transformer_qa_20_app.py`
- **학습 목표**: 실무형 질의응답(QA) 파이프라인 이해 및 적용
- **구성 요소**:
  1. **QA Pre-trained 모델 테스트**
     - 다양한 사전학습 모델 선별 및 성능 확인
  2. **데이터셋 준비**
     - 데이터셋 로드 및 Train/Validation 분리
  3. **전처리**
     - 질문 + 문맥 토큰화
     - 정답 스팬(offsets 위치 정보) 매핑
     - `batched=True` 적용
  4. **DataLoader 구성**
     - `collate_fn` 정의: batch → tensor 변환
     - DataLoader 생성
  5. **모델 정의**
     - Feature Extraction + LoRA Fine-tuning 조합
     - 최적화 설정: Optimizer, GradScaler, autocast
     - Early Stopping 클래스 정의 및 최적 모델 가중치 저장
  6. **학습/검증 루프**
     - 딕셔너리 형태 학습데이터를 그대로 모델에 전달
     - Early Stopping 객체 적용
  7. **평가 파이프라인**
     - F1/EM 평가 지표 활용
  8. **추론**
     - 단일 테스트 및 다중 테스트
     - 문장 추론: FastAPI 호출
  9. **서비스 구성**
     - FastAPI 실행:
       ```bash
       uvicorn transformer_qa_20_app:app --reload
       ```
     - 엔드포인트: `http://127.0.0.1:8000/qa`
     - Postman 및 API 코드(Python, Java 등)로 테스트 가능

### 41. Transformer News Summary
- **파일**: `LLM/19.transformer_summary_news.ipynb`, `LLM/llm_app/transformer_summary_news_19_app.py`
- **학습 목표**: 실무형 뉴스 요약 파이프라인 이해 및 적용
- **구성 요소**:
  1. **데이터셋 준비**
     - JSON 파일 로드 및 본문/요약 추출
  2. **전처리**
     - 데이터 정제 및 토크나이저 적용
     - Hugging Face `BartTokenizer` 기반 모델 사용
     - `collate_fn` 적용 후 DataLoader 생성
  3. **모델 정의**
     - Feature Extraction + LoRA Fine-tuning 조합
     - Early Stopping 클래스 정의
  4. **학습 루프**
     - `autocast` 적용 (속도 향상)
     - `GradScaler` 적용 (안정적 학습)
  5. **테스트 및 평가**
     - 최적 모델 로드 후 실제 요약 생성
     - ROUGE 주요 지표 활용
  6. **서비스 구성**
     - FastAPI 실행:
       ```bash
       uvicorn transformer_summary_news_19_app:app --reload
       ```
     - 엔드포인트: `http://127.0.0.1:8000/summarize`
     - Postman 및 API 코드(Python, Java 등)로 테스트 가능

### 40. Transformer Sentiment Classifier
- **파일**: `LLM/18.transformer_classifier_sentiment.ipynb`, `LLM/llm_app/transformer_classifier_sentiment_18_app.py`
- **학습 목표**: 실무형 감정 분류 파이프라인 이해 및 적용
- **구성 요소**:
  1. **데이터 준비**
     - 데이터 로드 및 결측치 제거 (None, "")
  2. **토크나이저 적용**
     - Hugging Face `DistilBertTokenizer` 사용
  3. **DataLoader 변환**
     - 토크나이저에서 바로 DataLoader 생성
     - Pre-trained 모델에서는 Custom Dataset 불필요
  4. **모델 정의**
     - 베이스 모델: `DistilBertForSequenceClassification` (`distilbert-base-uncased`)
     - 클래스 수: 2 (긍정/부정)
     - 본체 동결(Feature Extraction) + LoRA Fine-tuning 조합
     - EarlyStopping 클래스 정의 및 최적 모델 가중치 저장
  5. **학습/검증 루프**
     - 최적화 설정: `autocast` (속도 향상), `GradScaler` (안정적 학습)
     - EarlyStopping 적용
  6. **모델 로드 및 추론**
     - GPU 설정 후 검증/추론 모드 적용
  7. **평가**
     - 사이킷런 평가 지표: 정확도, 정밀도, 재현율, F1-score
     - Confusion Matrix 분석 및 Heatmap 시각화
  8. **테스트**
     - 단일 문장 및 여러 문장 추론
  9. **서비스 구성**
     - FastAPI 실행:
       ```bash
       uvicorn transformer_classifier_sentiment_18_app:app --reload
       ```
     - 엔드포인트:
       - 단일 문장: `http://127.0.0.1:8000/predict`
       - 여러 문장: `http://127.0.0.1:8000/predict_batch`
     - 윈도우 PowerShell 예시:
       ```powershell
       Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"text":"I really love this movie, it was fantastic!"}'
       ```
     - Postman 및 API 코드(Python, Java 등)로 테스트 가능

### 39. Transformer Self-Attention 기반 감정 분류 모델
- **파일**: `LLM/17.transformer_self_attention.ipynb`
- **학습 목표**: Transformer 구조 이해 및 감정 분류 모델 구축
- **구성 요소**:
  1. **Encoder 모델 구축**
     - Scaled Dot-Product Attention
     - Multi-Head Attention
     - Transformer Encoder Block (Attention → FFN → Residual → LayerNorm)
     - Positional Encoding
     - Transformer Encoder 전체 구조
  2. **Decoder 모델 구축**
     - Masked Multi-Head Attention
     - Cross Attention
     - Transformer Decoder Block (Masked Attention → Cross Attention → Residual → LayerNorm)
     - Positional Encoding
     - Transformer Decoder 전체 구조
  3. **Classifier 모델 구축**
     - 입력 문장을 기반으로 긍정/부정 감정 분류
     - Transformer Encoder/Decoder를 활용한 분류기 설계

### 38. Transformer Word Embedding 학습
- **파일**: `LLM/16.transformer_word_embedding.ipynb`
- **학습 목표**: Transformer 모델 내 워드 임베딩 처리 및 학습 이해
- **핵심 개념**:
  - 각 단어마다 vocab 전체와 확률 비교 → 정답과 비교 → 손실 계산 → 파라미터 업데이트 → logits 생성
  - 학습 과정에서 임베딩이 점점 의미를 반영 → 비슷한 단어끼리 가까워지는 성질 발생
  - 임베딩 행렬의 각 벡터가 학습을 통해 의미 공간에서 위치를 바꿈
- **구성 요소**:
  1. **토크나이저 → 인덱스 변환**
     - 텍스트를 토큰 단위로 분리 (WordPiece, BPE, SentencePiece 등)
     - 각 토큰을 고유 인덱스로 매핑
  2. **임베딩 레이어 생성**
     - PyTorch `nn.Embedding` 사용
     - 인덱스를 고정 길이 벡터로 변환
     - 학습 가능한 파라미터로 초기화 → 학습 과정에서 업데이트
  3. **학습 방식**
     - 랜덤 초기화 후 학습: 모델 학습 과정에서 의미를 점차 학습
     - 사전학습 임베딩 활용: Word2Vec, GloVe, FastText 등
     - Transformer 기반 임베딩: BERT, GPT 등 사전학습 모델의 임베딩 레이어를 가져와 파인튜닝

### 37. Transformer 생성형 모델 & 파인튜닝 (GPT-2 기반)
- **파일**: `LLM/15.transformer_gpt-2.ipynb`
- **학습 목표**: 구조 최적화 및 파이프라인 단순화
- **구성 요소**:
  1. **데이터셋 준비**
     - AI HUB 금융 분야 다국어 말뭉치 데이터셋 적용
     - 금융 학술논문 데이터셋 변환 및 전처리
  2. **토크나이징**
     - 입력 문장 토크나이징 및 전처리
  3. **베이스 모델 로드**
     - Hugging Face GPT-2 기반 모델 불러오기
  4. **LoRA(Low-Rank Adaptation) 적용**
     - 특정 레이어에 저차원 행렬(랭크 r) 삽입하여 학습
     - 메모리 효율성, 빠른 학습, 도메인 적용 가능
     - Base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
  5. **학습 설정**
     - 학습 인자(args) 정의
     - Trainer 객체 생성 및 실행
  6. **모델 저장 및 불러오기**
     - LoRA 적용된 모델 및 토크나이저 저장
     - 베이스 모델 + LoRA 모델 + 토크나이저 불러오기

### 36. Transformer 요약 모델 & 파인튜닝 (뉴스 데이터셋 기반)
- **파일**: `LLM/14.transformer(summary_news).ipynb`
- **학습 목표**: 구조 최적화 및 파이프라인 단순화
- **구성 요소**:
  1. **데이터셋 준비**
     - AI HUB 요약문 및 레포트 뉴스(news) 데이터셋 전처리
     - 병렬 문장쌍 데이터셋 변환
  2. **토크나이징**
     - 입력 문장 토크나이징 및 전처리
  3. **베이스 모델 로드**
     - Hugging Face 기반 요약 모델 불러오기
  4. **LoRA(Low-Rank Adaptation) 적용**
     - 특정 레이어에 저차원 행렬(랭크 r) 삽입하여 학습
     - 메모리 효율성, 빠른 학습, 도메인 적용 가능
     - Base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
  5. **학습 설정**
     - 학습 인자(args) 정의
     - Trainer 객체 생성 및 실행
  6. **모델 저장 및 불러오기**
     - LoRA 적용된 모델 및 토크나이저 저장
     - 베이스 모델 + LoRA 모델 + 토크나이저 불러오기

### 35. Transformer 다국어 번역 + 금융 분야 분류 모델
- **파일**: `LLM/13.transformer(translation_with_finance_classification).ipynb`
- **학습 목표**: 구조 최적화 및 파이프라인 단순화
- **구성 요소**:
  1. **데이터셋 준비**
     - AI HUB 금융 학술논문/공시정보/뉴스/규정/보고서 다국어 번역 데이터셋 활용
     - 학습 보강을 통한 데이터 품질 개선
  2. **토크나이징**
     - 입력 문장 토크나이징 및 전처리
  3. **모델 구성**
     - 입력 문장의 언어 분류
     - 문장 유형 분류: 학술논문(0), 공시정보(1), 뉴스(2), 규정(3), 보고서(4)
     - 해당 유형에 맞는 기계 번역 모델 선택 및 적용
  4. **LoRA(Low-Rank Adaptation) 적용**
     - LoRA 적용된 모델 불러오기
     - 베이스 모델 + LoRA 모델 + 토크나이저 조합

### 34. Transformer 다국어 번역 모델 분류기 (금융 데이터셋 기반)
- **파일**: `LLM/12.transformer(translation_finance_classification).ipynb`
- **학습 목표**: 구조 최적화 및 파이프라인 단순화
- **구성 요소**:
  1. **데이터셋 준비**
     - AI HUB 금융 학술논문/공시정보/뉴스/규정/보고서 데이터셋 활용
     - 데이터 전처리 및 병렬 문장쌍 변환
  2. **토크나이징**
     - 입력 문장 토크나이징 및 전처리
  3. **베이스 모델 로드**
     - Hugging Face 기반 다국어 번역 모델 불러오기
  4. **LoRA(Low-Rank Adaptation) 적용**
     - 특정 레이어에 저차원 행렬(랭크 r) 삽입하여 학습
     - 메모리 효율성, 빠른 학습, 도메인 적용 가능
     - Base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
  5. **학습 설정 및 실행**
     - 학습 인자(args) 정의
     - Trainer 객체 생성 및 실행
  6. **모델 저장 및 불러오기**
     - LoRA 적용된 모델 및 토크나이저 저장
     - 베이스 모델 + LoRA 모델 + 토크나이저 불러오기
  7. **문장 분류**
     - 입력 문장을 학술논문(0), 공시정보(1), 뉴스(2), 규정(3), 보고서(4)로 분류
     - 분류 결과에 따라 해당 번역 모델 적용

### 33. Transformer 다국어 번역 모델 (금융 공시 정보 데이터셋 기반)
- **파일**: `LLM/11.transformer(translation_finance_disclosure).ipynb`
- **학습 목표**: 구조 최적화 및 파이프라인 단순화
- **구성 요소**:
  1. **데이터셋 준비**
     - AI HUB 금융 공시 정보 데이터셋 활용
     - 데이터 전처리 및 병렬 문장쌍 변환
  2. **토크나이징**
     - 입력 문장 토크나이징 및 전처리
  3. **베이스 모델 로드**
     - Hugging Face 기반 다국어 번역 모델 불러오기
     - 영어 ↔ 한국어 번역 지원
  4. **LoRA(Low-Rank Adaptation) 적용**
     - 특정 레이어에 저차원 행렬(랭크 r) 삽입하여 학습
     - 메모리 효율성, 빠른 학습, 도메인 적용 가능
     - Base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
  5. **학습 설정 및 실행**
     - 학습 인자(args) 정의
     - Trainer 객체 생성 및 실행
  6. **모델 저장 및 불러오기**
     - LoRA 적용된 모델 및 토크나이저 저장
     - 베이스 모델 + LoRA 모델 + 토크나이저 불러오기

### 32. Transformer 다국어 번역 모델 (금융 뉴스 데이터셋 기반)
- **파일**: `LLM/10.transformer(translation_finance_news).ipynb`
- **학습 목표**: 구조 최적화 및 파이프라인 단순화
- **구성 요소**:
  1. **데이터셋 준비**
     - AI HUB 금융 뉴스 데이터셋 활용
     - 데이터 전처리 및 병렬 문장쌍 변환
  2. **토크나이징**
     - 입력 문장 토크나이징 및 전처리
  3. **베이스 모델 로드**
     - Hugging Face 기반 다국어 번역 모델 불러오기
     - 영어 ↔ 한국어 번역 지원
  4. **LoRA(Low-Rank Adaptation) 적용**
     - 특정 레이어에 저차원 행렬(랭크 r) 삽입하여 학습
     - 메모리 효율성, 빠른 학습, 도메인 적용 가능
     - Base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
  5. **학습 설정 및 실행**
     - 학습 인자(args) 정의
     - Trainer 객체 생성 및 실행
  6. **모델 저장 및 불러오기**
     - LoRA 적용된 모델 및 토크나이저 저장
     - 베이스 모델 + LoRA 모델 + 토크나이저 불러오기

### 31. Transformer 다국어 번역 모델 (금융 보고서 데이터셋 기반)
- **파일**: `LLM/09.transformer(translation_finance_report).ipynb`
- **학습 목표**: 구조 최적화 및 파이프라인 단순화
- **구성 요소**:
  1. **데이터셋 준비**
     - AI HUB 금융 보고서 데이터셋 활용
     - 데이터 전처리 및 병렬 문장쌍 변환
  2. **토크나이징**
     - 입력 문장 토크나이징 및 전처리
  3. **베이스 모델 로드**
     - Hugging Face 기반 다국어 번역 모델 불러오기
     - 영어 ↔ 한국어 번역 지원
  4. **LoRA(Low-Rank Adaptation) 적용**
     - 특정 레이어에 저차원 행렬(랭크 r) 삽입하여 학습
     - 메모리 효율성, 빠른 학습, 도메인 적용 가능
     - Base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
  5. **학습 설정 및 실행**
     - 학습 인자(args) 정의
     - Trainer 객체 생성 및 실행
  6. **모델 저장 및 불러오기**
     - LoRA 적용된 모델 및 토크나이저 저장
     - 베이스 모델 + LoRA 모델 + 토크나이저 불러오기

### 30. Transformer 다국어 번역 모델 (금융 규제 정보 데이터셋 기반)
- **파일**: `LLM/08.transformer(translation_finance_regulation).ipynb`
- **학습 목표**: 구조 최적화 및 파이프라인 단순화
- **구성 요소**:
  1. AI HUB 금융 규제 정보 데이터셋 전처리
  2. 병렬 문장쌍 데이터셋 변환
  3. 토크나이징 및 전처리
  4. 베이스 모델 로드
  5. LoRA 적용 및 학습
  6. Trainer 실행
  7. LoRA 모델/토크나이저 저장 및 불러오기
  8. 영어 ↔ 한국어 번역 지원

### 29. Transformer 다국어 번역 모델 (금융 학술 논문 데이터셋 기반)
- **파일**: `LLM/07.transformer(translation_finance_article).ipynb`
- **학습 목표**: 구조 최적화 및 파이프라인 단순화
- **구성 요소**:
  1. AI HUB 금융 학술 논문 데이터셋 전처리
  2. 병렬 문장쌍 데이터셋 변환
  3. 토크나이징 및 전처리
  4. 베이스 모델 로드
  5. LoRA 적용 및 학습
  6. Trainer 실행
  7. LoRA 모델/토크나이저 저장 및 불러오기
  8. 영어 ↔ 한국어 번역 지원

### 28. Transformer 다국어 번역 모델 (방송 데이터셋 기반)
- **파일**: `LLM/06.transformer(translation_broadcast).ipynb`
- **학습 목표**: 구조 최적화 및 파이프라인 단순화
- **구성 요소**:
  1. AI HUB 방송 다국어 번역 데이터셋 전처리
  2. 병렬 문장쌍 데이터셋 변환
  3. 토크나이징 및 전처리
  4. 베이스 모델 로드
  5. LoRA 적용 및 학습
  6. Trainer 실행
  7. LoRA 모델/토크나이저 저장 및 불러오기
  8. 영어 ↔ 한국어 번역 지원

### 27. Transformer News Analysis (AI HUB 뉴스 기사 기반)
- **파일**: `LLM/05_transformer(news_analysis_aihub_news).ipynb`
- **학습 목표**: 뉴스 카테고리 분류 모델 구축
- **구성 요소**:
  - 입력 문장을 정치/경제/사회/문화/IT·과학/스포츠 카테고리로 분류
  - 다양한 Transformer 아키텍처(BERT, RoBERTa, ELECTRA 등) 비교
  - 성능 지표: Macro F1, Accuracy, Recall 최적화
  - Attention 가중치 분석을 통한 모델 해석
  - 확장: 멀티레이블 분류, 다국어 뉴스 분류

### 26. Transformer News Analysis (AG News 데이터셋 기반)
- **파일**: `LLM/04_transformer(news_analysis_ag).ipynb`
- **학습 목표**: 뉴스 카테고리 분류 모델 구축
- **구성 요소**:
  - 입력 문장을 정치/경제/과학·기술/스포츠 카테고리로 분류
  - 다양한 Transformer 아키텍처 비교 및 성능 최적화
  - 데이터 증강 및 정규화 적용
  - Attention 가중치 분석을 통한 모델 해석
  - 확장: 멀티레이블 분류, 다국어 뉴스 분류

### 25. Transformer Sentiment Analysis (Naver 영화 리뷰, 다국어)
- **파일**: `LLM/03_transformer(sentiment_analysis_naver_xlm-roberta).ipynb`
- **학습 목표**: 긍정/부정 감정 분류 모델 구축
- **구성 요소**:
  - 입력 문장을 긍정(Positive) 또는 부정(Negative)으로 자동 분류
  - 문맥적 의미와 뉘앙스를 고려한 감정 해석
  - 일반화 성능 확보: 새로운 문장에서도 정확한 분류 수행

### 24. Transformer Sentiment Analysis (Naver 영화 리뷰, 한국어)
- **파일**: `LLM/02_transformer(sentiment_analysis_naver).ipynb`
- **학습 목표**: 한국어 영화 리뷰 기반 감정 분류
- **구성 요소**:
  - 긍정/부정 자동 분류
  - 문맥적 의미와 뉘앙스를 고려한 감정 해석
  - 과적합 방지 및 일반화 성능 확보

### 23. Transformer Sentiment Analysis (IMDB 리뷰, 영어)
- **파일**: `LLM/01_transformer(sentiment_analysis_imdb).ipynb`
- **학습 목표**: 영어 영화 리뷰 기반 감정 분류
- **구성 요소**:
  - 긍정/부정 자동 분류
  - 문맥적 의미와 뉘앙스를 고려한 감정 해석
  - 다양한 표현 방식 이해 및 일반화 성능 확보

### 22. Hybrid CNN + Attention Image Captioning (COCO 데이터셋)
- **파일**: `22_hybrid_coco_attention.ipynb`
- **학습 목표**: Attention 기반 이미지 캡션 생성
- **구성 요소**:
  - Encoder: CNN(ResNet-50)으로 이미지 특징 추출
  - Decoder: Attention 기반 시퀀스 생성
  - 매 시점마다 이미지의 다른 위치에 집중하여 단어 생성
  - Attention Map 시각화로 단어-이미지 위치 관계 확인

### 21. Hybrid CNN + RNN Image Captioning (COCO 데이터셋)
- **파일**: `21_hybrid_coco.ipynb`
- **학습 목표**: CNN-RNN 하이브리드 구조로 이미지 캡션 생성
- **구성 요소**:
  - Encoder: CNN으로 이미지 특징 추출
  - Decoder: RNN(LSTM/GRU)으로 시퀀스 생성
  - 학습 데이터셋: MS COCO
  - 손실 함수: `nn.CrossEntropyLoss()`
  - 옵티마이저: `torch.optim.Adam`

### 20. Hybrid CNN + RNN (EMNIST 손글씨 숫자+알파벳)
- **파일**: `20_deep_learning_hybrid_emnist.ipynb`
- **웹앱 구조**: `hybrid-emnist-streamlit/src/`
- **학습 목표**: EMNIST 데이터셋 기반 CNN+RNN 하이브리드 모델 구축
- **구성 요소**:
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리 (train, evaluate, test)
  - 모델 저장 및 불러오기
  - Confusion Matrix 및 오차 분석
  - Streamlit 웹앱 데모: 무작위 이미지, 업로드, 직접 그리기 입력 지원

### 19. Hybrid CNN + RNN (MNIST 손글씨 숫자)
- **파일**: `19_deep_learning_hybrid.ipynb`
- **웹앱 구조**: `hybrid-mnist-streamlit/src/`
- **학습 목표**: MNIST 데이터셋 기반 CNN+RNN 하이브리드 모델 구축
- **구성 요소**:
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Confusion Matrix 및 오차 분석
  - Streamlit 웹앱 데모: 무작위 이미지, 업로드, 직접 그리기 입력 지원

### 18. Transfer Learning (GTSRB 교통 표지판 인식)
- **파일**: `18_transfer_learning_gtsrb_traffic_sign_detection.ipynb`
- **웹앱 구조**: `gtsrb-traffic-sign-detection-streamlit/src/`
- **학습 목표**: 교통 표지판 이미지 분류 모델 구축
- **구성 요소**:
  - Pre-trained 모델 기반 전이학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Streamlit 앱: 이미지 업로드/웹캠 입력/멀티 이미지 지원

### 17. Transfer Learning (Kaggle Surface Crack Detection)
- **파일**: `17_transfer_learning_kaggle_surface_crack_detection.ipynb`
- **웹앱 구조**: `surface_crack-detection-streamlit/src/`
- **학습 목표**: 콘크리트 표면 결함 예측 모델 구축
- **구성 요소**:
  - Pre-trained 모델 기반 전이학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Streamlit 앱: 이미지 업로드/웹캠 입력/멀티 이미지 지원

### 16. Transfer Learning (Kaggle Breast Ultrasound Detection)
- **파일**: `16_transfer_learning_kaggle_breast_ultrasound_detection.ipynb`
- **웹앱 구조**: `breast-detection-streamlit/src/`
- **학습 목표**: 유방암 초음파 이미지 분류 모델 구축
- **구성 요소**:
  - Pre-trained 모델 기반 전이학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Streamlit 앱: 이미지 업로드/웹캠 입력/멀티 이미지 지원

### 15. Transfer Learning (COVID-19 Detection)
- **파일**: `15_transfer_learning_kaggle_covid19_detection.ipynb`
- **웹앱 구조**: `covid19-detection-streamlit/src/`
- **학습 목표**: COVID-19 감염 예측 이미지 분류 모델 구축
- **구성 요소**:
  - Pre-trained 모델 기반 전이학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Streamlit 앱: 이미지 업로드/웹캠 입력/멀티 이미지 지원

### 14. Transfer Learning (Face Emotion Detection)
- **파일**: `14_transfer_learning_kaggle_emotion_detection.ipynb`
- **웹앱 구조**: `face-emotion-streamlit/src/`
- **학습 목표**: 얼굴 감정 이미지 분류 모델 구축
- **구성 요소**:
  - Pre-trained 모델 기반 전이학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Streamlit 앱: 얼굴 감정 예측 데모

### 13. Transfer Learning (Face Mask Detection)
- **파일**: `13_transfer_learning_kaggle_face_mask_detection.ipynb`
- **웹앱 구조**: `face-mask-streamlit/src/`
- **학습 목표**: 얼굴 마스크 착용 여부 분류 모델 구축
- **구성 요소**:
  - Pre-trained 모델 기반 전이학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Streamlit 앱: 얼굴 마스크 착용 예측 데모

### 12. Transfer Learning (Brain Tumor MRI Detection)
- **파일**: `12_transfer_learning_kaggle_brain_tumor_mri.ipynb`
- **웹앱 구조**: `brain-tumor-streamlit/src/`
- **학습 목표**: 뇌종양 MRI 이미지 분류 모델 구축
- **구성 요소**:
  - Pre-trained 모델 기반 전이학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Streamlit 앱: 뇌종양 예측 데모

### 10. Transfer Learning (Dog Emotion Detection)
- **파일**: `11_transfer_learning_vit_dog_emotion_gpu.ipynb`
- **웹앱 구조**: `dogs-image-streamlit/src/`
- **학습 목표**: 강아지 감정 분류 모델 구축
- **구성 요소**:
  - Pre-trained 모델 기반 전이학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Streamlit 앱: 강아지 감정 예측 데모

### 9. Transfer Learning (Dog Breed Classification)
- **파일**: `10_transfer_learning_vit_custom_image_gpu.ipynb`
- **웹앱 구조**: `dogs-image-streamlit/src/`
- **학습 목표**: 강아지 종 분류 모델 구축
- **구성 요소**:
  - Pre-trained 모델 기반 전이학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Streamlit 앱: 강아지 종 예측 데모

### 8. Transfer Learning (Cats vs Dogs Classification)
- **파일**: `09_transfer_learning_cats_dogs_gpu.ipynb`
- **웹앱 구조**: `cats-dogs-streamlit/src/`
- **학습 목표**: 고양이 vs 강아지 이미지 분류 모델 구축
- **구성 요소**:
  - Pre-trained 모델 기반 전이학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Streamlit 앱: 고양이/강아지 분류 데모

### 7. Deep CNN (CIFAR10 Dataset)
- **파일**: `07_deep_cnn_cifar10_gpu.ipynb`
- **웹앱 구조**: `app_07_deep_cnn_cifar10.py`
- **학습 목표**: CIFAR10 이미지 분류 모델 구축
- **구성 요소**:
  - Deep CNN 기반 학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 학습/평가 함수 분리
  - 모델 저장 및 불러오기
  - Streamlit 앱: CIFAR10 이미지 분류 데모

### 6. CNN (CIFAR10 Dataset)
- **파일**: `06_cnn_cifar10_gpu.ipynb`
- **웹앱 구조**: `app_06_cnn_cifar10.py`
- **학습 목표**: CIFAR10 이미지 분류 모델 구축
- **구성 요소**:
  - CNN 기반 학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - StepLR, EarlyStopping 적용
  - 성능 지표: Accuracy, Confusion Matrix, Precision, Recall, F1-score
  - 모델 저장 및 불러오기
  - Streamlit 앱: CIFAR10 이미지 분류 데모

### 5. CNN (Fashion-MNIST Dataset)
- **파일**: `05_cnn_fashion_mnist_gpu.ipynb`
- **웹앱 구조**: `app_05_cnn_fashion_mnist.py`
- **학습 목표**: Fashion-MNIST 이미지 분류 모델 구축
- **구성 요소**:
  - CNN 기반 학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - StepLR, EarlyStopping 적용
  - 성능 지표: Accuracy, Confusion Matrix, Precision, Recall, F1-score
  - 모델 저장 및 불러오기
  - Streamlit 앱: Fashion-MNIST 이미지 분류 데모

### 4. CNN (MNIST Dataset)
- **파일**: `04_cnn_mnist_gpu.ipynb`
- **웹앱 구조**: `app_04_cnn_mnist.py`
- **학습 목표**: MNIST 손글씨 이미지 분류 모델 구축
- **구성 요소**:
  - CNN 기반 학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - StepLR, EarlyStopping 적용
  - 성능 지표: Accuracy, Confusion Matrix, Precision, Recall, F1-score
  - 모델 저장 및 불러오기
  - Streamlit 앱: 무작위 이미지, 업로드, 직접 그리기 입력 지원

### 3. MLP (Fashion-MNIST Dataset)
- **파일**: `03_mlp_fashion_mnist_gpu.ipynb`
- **웹앱 구조**: `app_03_mlp_fashion_mnist.py`
- **학습 목표**: Fashion-MNIST 이미지 분류 모델 구축
- **구성 요소**:
  - MLP 기반 학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - StepLR, EarlyStopping 적용
  - 성능 지표: Accuracy, Confusion Matrix, Precision, Recall, F1-score
  - 모델 저장 및 불러오기
  - Streamlit 앱: Fashion-MNIST 이미지 분류 데모

### 2. MLP (MNIST Dataset)
- **파일**: `02_mlp_mnist_gpu.ipynb`
- **웹앱 구조**: `app_02_mlp_mnist_model.py`, `app_02_mlp_mnist_model_image_upload.py`
- **학습 목표**: MNIST 손글씨 이미지 분류 모델 구축
- **구성 요소**:
  - MLP 기반 학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 성능 지표: Accuracy, Confusion Matrix, Precision, Recall, F1-score
  - 모델 저장 및 불러오기
  - Streamlit 앱: 무작위 이미지, 업로드, 직접 그리기 입력 지원

### 1. MLP (Basic Dataset)
- **파일**: `01_mlp.ipynb`
- **웹앱 구조**: `app_01_mlp_model.py`, `app_01_mlp_model_csv_upload.py`, `app_01_mlp_model_csv_upload_download.py`
- **학습 목표**: 기본 데이터셋 기반 MLP 이진 분류기 구축
- **구성 요소**:
  - MLP 기반 학습
  - Dataset 및 DataLoader 활용
  - 하이퍼파라미터 튜닝
  - 성능 지표: Accuracy, Confusion Matrix, Precision, Recall, F1-score
  - 모델 저장 및 불러오기
  - Streamlit 앱: 숫자 입력, CSV 업로드/다운로드 지원