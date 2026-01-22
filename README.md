<h2> My tech stack </h2>
<div align=center>
        <img src="https://img.shields.io/badge/springboot-6DB33F?style=for-the-badge&logo=springboot&logoColor=white">
        <img src="https://img.shields.io/badge/Spring-6DB33F?style=for-the-badge&logo=Spring&logoColor=white">
        <img src="https://img.shields.io/badge/java-007396?style=for-the-badge&logo=OpenJDK&logoColor=white">
        <img src="https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E">
        <img src="https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jQuery&logoColor=white"/>
        <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
        <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">
        <img src="https://img.shields.io/badge/Anaconda-44A833?style=for-the-badge&logo=Anaconda&logoColor=white"/>
        <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black"/>
        <img src="https://img.shields.io/badge/Apache Tomcat-F8DC75?style=for-the-badge&logo=apachetomcat&logoColor=black"/>
        <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white">
        <img src="https://img.shields.io/badge/ORACLE-F80000?style=for-the-badge&logo=oracle&logoColor=white"/>
        <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=MySQL&logoColor=white">
    <br>
</div>

<br>
<h2> Deep Learning </h2>

---
### 40. Transformer 모델 구축 - Transformer Sentiment Classifier 감정 분류 모델, Model - LLM/18.transformer_classifier_sentiment.ipynb
> Transformer 모델 구축 - Transformer Sentiment Classifier 감정 분류 모델
>> 학습 목표 - 실무에서 사용되는 파이프라인 이해 및 적용
> 1. 데이터 로드 & 확인
> - 결측치 제거(None, "")
> 2. 토크나이저 적용
> - Hugging Face DistilBertTokenizer 베이스 모델 사용
> 3. 데이터셋 -> DataLoader 변환
> - DistilBertTokenizer 베이스 모델 토크나이저에서 DataLoader 로 바로 변환
> - Hugging Face Pre-Trained Model에서는 Custom Dataset 필요 없음
> 4. 모델정의 & GPU설정 & 전이학습 & 본체 동결(Feature Extraction) + LoRA Fine-tuning 조합 & EarlyStopping 클래스 정의
> - 전이 학습: DistilBertForSequenceClassification 베이스 모델(distilbert-base-uncased), num_labels=2 긍정/부정 2개 클래스
> - 본체 동결: model.distilbert.parameters()는 사전학습된 본체(embedding + transformer 블록)의 모든 파라미터를 의미, 따라서 학습은 classifier 레이어(pre_classifier, classifier)만 진행된다
> - LoRA: Attention의 q_lin, v_lin 레이어에서 LoRA가 768차원 → 8차원 축소 → 768차원 복원 과정을 거쳐 업데이트를 추가하는 구조
> - EarlyStopping: - 과적합 방지 + 최적 모델 확보 + 자원 절약 + EarlyStopping 발동 시점에서 최적 모델 가중치를 자동 저장
> 5. 최적화 설정 & 학습 루프 & 검증 루프 EarlyStopping 클래스 적용
> - 최적화 설정: autocast(속도 향상) GradScaler(안정적 학습) 적용
> 6. 최적 모델 로드: GPU 설정, 검증/추론 모드 적용
> 7. 테스트 데이터 평가: 사이킷런 평가 지표 적용
> - classification_report 정확도/정밀도/재현율/F1-socre 확인
> - confusion_matrix 오분류 패턴 분석
> 8. Confusion Matrix Heatmap: Confusion Matrix를 Heatmap 그래프로 시각화 적용, 테스트 데이터 평가 데이터를 활용
---
### 39. Transformer 구조 이해 및 모델 구축 - Transformer Classifier 감정 분류 모델, Model - LLM/17.transformer_self_attention.ipynb
> Transformer 구조 이해 및 모델 구축 - Transformer Classifier 감정 분류 모델
> 1. Encoder 모델 구축
> - Scaled Dot-Product Attention
> - Multi-Head Attention
> - Transformer Encoder Block(Attention -> FFN -> Residual -> LayerNorm 구조)
> - Positional Encoding
> - Transformer Encoder
> 2. Decoder 모델 구축
> - Masked Multi-Head Attention
> - Cross Attention
> - Transformer Decoder Block(Masked Attention -> Cross Attention -> Residual -> LayerNorm 구조)
> - Positional Encoding
> - Transformer Decoder
> 3. Transformer Classifier  모델 구축
> - Transformer Classifier 감정 분류 모델(문장을 입력 받아 긍정/부정 감정 분류)
---
### 38. Transformer 모델내에서 사용되는 워드 임베딩 처리 및 학습, Model - LLM/16.transformer_word_embedding.ipynb
> Transformer 모델내에서 사용되는 워드 임베딩 처리 및 학습
>> 학습 목표 
> - 각 단어마다 vocab 전체와 확률 비교 → 정답과 비교 → 손실 계산 → 파라미터 업데이트 → logits 생성이라는 흐름으로 학습
> - 그 과정에서 임베딩이 점점 의미를 반영하게 되고, 비슷한 단어끼리 가까워지는 성질이 생긴다
> - 임베딩 weight 업데이트, 임베딩 행렬의 각 벡터가 학습을 통해 의미 공간에서 위치를 바꾸는 것이다
> 1. 토크나이저 -> 인덱스 변환 
> - 텍스트를 토큰 단위로 분리 (WordPiece, BPE, SentencePiece 등) 
> - 각 토큰을 고유 인덱스로 매핑
> 2. 임베딩 레이어 생성 
> - PyTorch의 nn.Embedding을 사용해 인덱스를 고정 길이 벡터로 변환 
> - 학습 가능한 파라미터로 초기화 -> 학습 과정에서 업데이트 됨 
> 3. 학습 루프 및 임베딩 학습 방식 
> - 랜덤 초기화 후 학습 : 모델 학습 과정에서 임베딩이 점차 의미를 학습 
> - 사전학습 임베딩 활용 : Word2Vec, GloVe, FastText 같은 사전학습 벡터를 초기값으로 사용 
> - Transformer 기반 임베딩 : BERT, GPT 등 사전학습 모델의 임베딩 레이어를 가져와 파인 튜닝
---
### 37. Transformer 생성형 모델 & 파인 튜닝 - Hugging Face 라이브러리 적용, AI HUB 금융 분야 다국어 말뭉치 데이터셋 적용, Model - LLM/15.transformer_gpt-2.ipynb
> Transformer 생성형 모델 & 파인 튜닝
> - Hugging Face 라이브러리 적용
> - 입력된 문장을 생성형 모델을 통한 문장 생성
>> 학습 목표
> - 구조 최적화 및 파이프라인 단순화
> 1. AI HUB 금융 분야 다국어 말뭉치 데이터셋 적용
> 2. 금융 학술논문 데이터셋 변환 전처리
> 3. 토크나이징 및 토크나이징 전처리
> 4. 베이스 모델 로드
> 5. LoRA(Low-Rank Adaptation) 설정, 특정 레이어에 작은 저차원 행렬(랭크 r)을 삽입해서 학습
> - LoRA(Low-Rank Adaptation) 모델, 메모리 효율성/빠른 학습/도메인 적용, base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
> 6. 학습 args 설정
> 7. Trainer 정의
> 8. Trainer 실행
> 9. LoRA 적용된 모델 저장, LoRA모델/토크나이저
> 10. LoRA 적용된 모델 불러오기, 베이스모델/LoRA모델/토크나이저
---
### 36. Transformer 요약 모델 & 파인 튜닝 - Hugging Face 라이브러리 적용, AI HUB 요약문 및 레포트 뉴스(news) 데이터셋 적용, Model - LLM/14.transformer(summary_news).ipynb
> Transformer 요약 모델 & 파인 튜닝
> - Hugging Face 라이브러리 적용
> - 입력된 문장을 요약 모델을 통한 뉴스(news) 원문 요약
>> 학습 목표
> - 구조 최적화 및 파이프라인 단순화
> 1. AI HUB 요약문 및 레포트 뉴스(news) 데이터셋 전처리
> 2. 병렬 문장쌍 데이터셋 변환 전처리
> 3. 토크나이징 및 토크나이징 전처리
> 4. 베이스 모델 로드
> 5. LoRA(Low-Rank Adaptation) 설정, 특정 레이어에 작은 저차원 행렬(랭크 r)을 삽입해서 학습
> - LoRA(Low-Rank Adaptation) 모델, 메모리 효율성/빠른 학습/도메인 적용, base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
> 6. 학습 args 설정
> 7. Trainer 정의
> 8. Trainer 실행
> 9. LoRA 적용된 모델 저장, LoRA모델/토크나이저
> 10. LoRA 적용된 모델 불러오기, 베이스모델/LoRA모델/토크나이저
---
### 35. Transformer 다국어 기계 번역 모델 분류기 및 다국어 기계 번역 - Hugging Face 라이브러리 적용, AI Hub 금융 학술논문/공시정보/뉴스/규정/보고서 데이터셋 적용, Model - LLM/13.transformer(translation_with_finance_classification).ipynb
> Transformer 다국어 기계 번역 모델 분류기 및 다국어 기계 번역
> - Hugging Face 라이브러리 적용
> - 입력된 문장을 다국어 기계 번역 모델 분류기을 통한 학술논문(0)/공시정보(1)/뉴스(2)/규정(3)/보고서(4) 분류
> - 입력 문장의 언어 분류 -> 입력 문장에 해당하는 기계 번역 모델 분류 -> 입력 문장에 해당하는 기계 번역
>> 학습 목표
> - 구조 최적화 및 파이프라인 단순화
> 1. AI HUB 금융 학술논문/공시정보/뉴스/규정/보고서 다국어 번역 데이터셋 학습 보강
> 2. 토크나이징 및 토크나이징 전처리
> 3. LoRA 적용된 모델 불러오기, 베이스모델/LoRA모델/토크나이저
---
### 34. Transformer 다국어 기계 번역 모델 분류기 - Hugging Face 라이브러리 적용, AI Hub 금융 학술논문/공시정보/뉴스/규정/보고서 데이터셋 적용, Model - LLM/12.transformer(translation_finance_classification).ipynb
> Transformer 다국어 기계 번역 모델 분류기 & 파인 튜닝
> - Hugging Face 라이브러리 적용
> - AI HUB 금융 학술논문/공시정보/뉴스/규정/보고서 데이터셋 적용
> - 입력된 문장을 다국어 기계 번역 모델 분류기을 통한 학술논문(0)/공시정보(1)/뉴스(2)/규정(3)/보고서(4) 분류
>> 학습 목표
> - 구조 최적화 및 파이프라인 단순화
> 1. AI HUB 금융 학술논문/공시정보/뉴스/규정/보고서 데이터셋 전처리
> 2. 병렬 문장쌍 데이터셋 변환 전처리
> 3. 토크나이징 및 토크나이징 전처리
> 4. 베이스 모델 로드
> 5. LoRA(Low-Rank Adaptation) 설정, 특정 레이어에 작은 저차원 행렬(랭크 r)을 삽입해서 학습
> - LoRA(Low-Rank Adaptation) 모델, 메모리 효율성/빠른 학습/도메인 적용, base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
> 6. 학습 args 설정
> 7. Trainer 정의
> 8. Trainer 실행
> 9. LoRA 적용된 모델 저장, LoRA모델/토크나이저
> 10. LoRA 적용된 모델 불러오기, 베이스모델/LoRA모델/토크나이저
---
### 33. Transformer 다국어 기계 번역 모델 - Hugging Face 라이브러리 적용, AI Hub 금융 공시 정보 데이터셋 적용, Model - LLM/11.transformer(translation_finance_disclosure).ipynb
> Transformer 다국어 기계 번역 모델 & 파인 튜닝
> - Hugging Face 라이브러리 적용
> - AI HUB 금융 공시 정보 데이터셋 적용
> - 입력된 문장을 다국어 기계 번역 모델을 통한 영어->한국어, 한국어->영어 번역
>> 학습 목표
> - 구조 최적화 및 파이프라인 단순화
> 1. AI HUB 금융 공시 정보 데이터셋 전처리
> 2. 병렬 문장쌍 데이터셋 변환 전처리
> 3. 토크나이징 및 토크나이징 전처리
> 4. 베이스 모델 로드
> 5. LoRA(Low-Rank Adaptation) 설정, 특정 레이어에 작은 저차원 행렬(랭크 r)을 삽입해서 학습
> - LoRA(Low-Rank Adaptation) 모델, 메모리 효율성/빠른 학습/도메인 적용, base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
> 6. 학습 args 설정
> 7. Trainer 정의
> 8. Trainer 실행
> 9. LoRA 적용된 모델 저장, LoRA모델/토크나이저
> 10. LoRA 적용된 모델 불러오기, 베이스모델/LoRA모델/토크나이저
---
### 32. Transformer 다국어 기계 번역 모델 - Hugging Face 라이브러리 적용, AI Hub 금융 뉴스 데이터셋 적용, Model - LLM/10.transformer(translation_finance_news).ipynb
> Transformer 다국어 기계 번역 모델 & 파인 튜닝
> - Hugging Face 라이브러리 적용
> - AI HUB 금융 뉴스 데이터셋 적용
> - 입력된 문장을 다국어 기계 번역 모델을 통한 영어->한국어, 한국어->영어 번역
>> 학습 목표
> - 구조 최적화 및 파이프라인 단순화
> 1.  AI HUB 금융 뉴스 데이터셋 전처리
> 2. 병렬 문장쌍 데이터셋 변환 전처리
> 3. 토크나이징 및 토크나이징 전처리
> 4. 베이스 모델 로드
> 5. LoRA(Low-Rank Adaptation) 설정, 특정 레이어에 작은 저차원 행렬(랭크 r)을 삽입해서 학습
> - LoRA(Low-Rank Adaptation) 모델, 메모리 효율성/빠른 학습/도메인 적용, base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
> 6. 학습 args 설정
> 7. Trainer 정의
> 8. Trainer 실행
> 9. LoRA 적용된 모델 저장, LoRA모델/토크나이저
> 10. LoRA 적용된 모델 불러오기, 베이스모델/LoRA모델/토크나이저
---
### 31. Transformer 다국어 기계 번역 모델 - Hugging Face 라이브러리 적용, AI Hub 금융 보고서 데이터셋 적용, Model - LLM/09.transformer(translation_finance_report).ipynb
> Transformer 다국어 기계 번역 모델 & 파인 튜닝
> - Hugging Face 라이브러리 적용
> - AI HUB 금융 보고서 데이터셋 적용
> - 입력된 문장을 다국어 기계 번역 모델을 통한 영어->한국어, 한국어->영어 번역
>> 학습 목표
> - 구조 최적화 및 파이프라인 단순화
> 1. AI HUB 금융 보고서 데이터셋 전처리
> 2. 병렬 문장쌍 데이터셋 변환 전처리
> 3. 토크나이징 및 토크나이징 전처리
> 4. 베이스 모델 로드
> 5. LoRA(Low-Rank Adaptation) 설정, 특정 레이어에 작은 저차원 행렬(랭크 r)을 삽입해서 학습
> - LoRA(Low-Rank Adaptation) 모델, 메모리 효율성/빠른 학습/도메인 적용, base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
> 6. 학습 args 설정
> 7. Trainer 정의
> 8. Trainer 실행
> 9. LoRA 적용된 모델 저장, LoRA모델/토크나이저
> 10. LoRA 적용된 모델 불러오기, 베이스모델/LoRA모델/토크나이저
---
### 30. Transformer 다국어 기계 번역 모델 - Hugging Face 라이브러리 적용, AI Hub 금융 규제 정보 데이터셋 적용, Model - LLM/08.transformer(translation_finance_regulation).ipynb
> Transformer 다국어 기계 번역 모델 & 파인 튜닝
> - Hugging Face 라이브러리 적용
> - AI HUB 금융 규제 정보 데이터셋 적용
> - 입력된 문장을 다국어 기계 번역 모델을 통한 영어->한국어, 한국어->영어 번역
>> 학습 목표
> - 구조 최적화 및 파이프라인 단순화
> 1. AI HUB 금융 규제 정보 데이터셋 전처리
> 2. 병렬 문장쌍 데이터셋 변환 전처리
> 3. 토크나이징 및 토크나이징 전처리
> 4. 베이스 모델 로드
> 5. LoRA(Low-Rank Adaptation) 설정, 특정 레이어에 작은 저차원 행렬(랭크 r)을 삽입해서 학습
> - LoRA(Low-Rank Adaptation) 모델, 메모리 효율성/빠른 학습/도메인 적용, base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
> 6. 학습 args 설정
> 7. Trainer 정의
> 8. Trainer 실행
> 9. LoRA 적용된 모델 저장, LoRA모델/토크나이저
> 10. LoRA 적용된 모델 불러오기, 베이스모델/LoRA모델/토크나이저
---
### 29. Transformer 다국어 기계 번역 모델 - Hugging Face 라이브러리 적용, AI Hub 금융 학술 논문 데이터셋 적용, Model - LLM/07.transformer(translation_finance_article).ipynb
> Transformer 다국어 기계 번역 모델 & 파인 튜닝
> - Hugging Face 라이브러리 적용
> - AI HUB 금융 학술 논문 데이터셋 적용
> - 입력된 문장을 다국어 기계 번역 모델을 통한 영어->한국어, 한국어->영어 번역
>> 학습 목표
> - 구조 최적화 및 파이프라인 단순화
> 1. AI HUB 금융 학술 논문 데이터셋 전처리
> 2. 병렬 문장쌍 데이터셋 변환 전처리
> 3. 토크나이징 및 토크나이징 전처리
> 4. 베이스 모델 로드
> 5. LoRA(Low-Rank Adaptation) 설정, 특정 레이어에 작은 저차원 행렬(랭크 r)을 삽입해서 학습
> - LoRA(Low-Rank Adaptation) 모델, 메모리 효율성/빠른 학습/도메인 적용, base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
> 6. 학습 args 설정
> 7. Trainer 정의
> 8. Trainer 실행
> 9. LoRA 적용된 모델 저장, LoRA모델/토크나이저
> 10. LoRA 적용된 모델 불러오기, 베이스모델/LoRA모델/토크나이저
---
### 28. Transformer 다국어 기계 번역 모델 - Hugging Face 라이브러리 적용, AI HUB 방송 다국어 번역 데이터셋 적용, Model - LLM/06.transformer(translation_broadcast).ipynb
> Transformer 다국어 기계 번역 모델 & 파인 튜닝
> - Hugging Face 라이브러리 적용
> - AI HUB 방송 다국어 번역 데이터셋 적용
> - 입력된 문장을 다국어 기계 번역 모델을 통한 영어->한국어, 한국어->영어 번역
>> 학습 목표
> - 구조 최적화 및 파이프라인 단순화
> 1. AI HUB 방송 다국어 번역 데이터셋 전처리
> 2. 병렬 문장쌍 데이터셋 변환 전처리
> 3. 토크나이징 및 토크나이징 전처리
> 4. 베이스 모델 로드
> 5. LoRA(Low-Rank Adaptation) 설정, 특정 레이어에 작은 저차원 행렬(랭크 r)을 삽입해서 학습
> - LoRA(Low-Rank Adaptation) 모델, 메모리 효율성/빠른 학습/도메인 적용, base 모델에 여러 LoRA 모듈을 붙였다 떼었다 할 수 있음
> 6. 학습 args 설정
> 7. Trainer 정의
> 8. Trainer 실행
> 9. LoRA 적용된 모델 저장, LoRA모델/토크나이저
> 10. LoRA 적용된 모델 불러오기, 베이스모델/LoRA모델/토크나이저
---
### 27. Transformer News Analysis(뉴스 카테고리) 분류 모델 - AI HUB 뉴스 기사 기계독해 데이터 셋 적용, 다국어 뉴스 카테고리 분류, Model - LLM/05_transformer(news_analysis_aihub_news).ipynb
> Transformer News Analysis(뉴스 카테고리) 분류 모델
> - 입력된 문장을 AI HUB 뉴스 기사 카테고리 분류(정치, 경제, 사회, 문화/문화생활, IT/과학, 스포츠)
> - 뉴스 분류는 문서의 의미를 파악해 사전에 정의된 카테고리로 매핑하는 작업
>> 학습 목표
> - 성능 극대화 -> 다양한 Transformer 아키텍처(BERT, RoBERTa, ELECTRA 등) 비교, Macro F1, Accuracy, Recall 등 지표 최적화
> - 일반화 능력 확보 -> 새로운 뉴스 데이터에도 잘 작동하도록 데이터 증강 및 정규화 적용
> - 모델 이해 -> Attention 가중치 분석을 통해 어떤 단어·문맥이 분류에 중요한지 해석 가능성 확보
> - 실험적 확장 -> 멀티레이블 분류(한 뉴스가 여러 카테고리에 속할 수 있음), 다국어 뉴스 분류(한국어 + 영어)
> 1. 텍스트 입력 → 토크나이저로 단어를 토큰화
> 2. Transformer 인코더 → 문맥적 의미를 벡터로 변환
> 3. 분류 레이어(Softmax) → 각 카테고리 확률 계산
> 4. 출력 → 가장 높은 확률의 카테고리를 최종 결과로 반환
---
### 26. Transformer News Analysis(뉴스 카테고리) 분류 모델 - AG News 데이터셋 적용, 다국어 뉴스 카테고리 분류, Model - LLM/04_transformer(news_analysis_ag).ipynb
> Transformer News Analysis(뉴스 카테고리) 분류 모델
> - 입력된 문장을 뉴스 카테고리 분류(정치, 경제, 과학/기술, 스포츠)
> - 뉴스 분류는 문서의 의미를 파악해 사전에 정의된 카테고리로 매핑하는 작업
>> 학습 목표
> - 성능 극대화 -> 다양한 Transformer 아키텍처(BERT, RoBERTa, ELECTRA 등) 비교, Macro F1, Accuracy, Recall 등 지표 최적화
> - 일반화 능력 확보 -> 새로운 뉴스 데이터에도 잘 작동하도록 데이터 증강 및 정규화 적용
> - 모델 이해 -> Attention 가중치 분석을 통해 어떤 단어·문맥이 분류에 중요한지 해석 가능성 확보
> - 실험적 확장 -> 멀티레이블 분류(한 뉴스가 여러 카테고리에 속할 수 있음), 다국어 뉴스 분류(한국어 + 영어)
> 1. 텍스트 입력 → 토크나이저로 단어를 토큰화
> 2. Transformer 인코더 → 문맥적 의미를 벡터로 변환
> 3. 분류 레이어(Softmax) → 각 카테고리 확률 계산
> 4. 출력 → 가장 높은 확률의 카테고리를 최종 결과로 반환

---
### 25. Transformer Sentiment Analysis(긍정/부정) 분류 모델 - Naver 영화 리뷰 데이터셋 적용, 다국어 영화 리뷰(긍정/부정), Model - LLM/03_transformer(sentiment_analysis_naver_xlm-roberta).ipynb
> Transformer Sentiment Analysis(긍정/부정) 분류 모델
> - 입력된 문장(리뷰, 댓글, 트윗 등)을 긍정(Positive) 또는 **부정(Negative)**으로 자동 분류하는 것.
> - 예: "이 영화 정말 재미있었어요!" → 긍정, "스토리가 지루하고 너무 길었어요." → 부정
> 1. 언어 이해 능력 강화
> - 모델이 문장의 어휘, 문맥, 뉘앙스를 파악해 감정을 올바르게 해석하도록 학습.
> - 단순히 키워드만 보는 것이 아니라, 문맥적 의미까지 고려해야 함.
> - "너무 무섭게 재미있었다" → 긍정
> - "재미있긴 했지만 너무 길었다" → 혼합적 뉘앙스 → 최종적으로 부정으로 분류될 수 있음
> 2. 일반화 성능 확보
> - 학습 데이터에만 맞추는 것이 아니라, 새로운 문장에서도 정확히 분류할 수 있어야 함.
> - 즉, 과적합을 피하고 다양한 표현 방식을 이해하는 능력을 키우는 것.
---
### 24. Transformer Sentiment Analysis(긍정/부정) 분류 모델 - Naver 영화 리뷰 데이터셋 적용, 한국어 영화 리뷰(긍정/부정), Model - LLM/02_transformer(sentiment_analysis_naver).ipynb
> Transformer Sentiment Analysis(긍정/부정) 분류 모델
> - 입력된 문장(리뷰, 댓글, 트윗 등)을 긍정(Positive) 또는 **부정(Negative)**으로 자동 분류하는 것.
> - 예: "이 영화 정말 재미있었어요!" → 긍정, "스토리가 지루하고 너무 길었어요." → 부정
> 1. 언어 이해 능력 강화
> - 모델이 문장의 어휘, 문맥, 뉘앙스를 파악해 감정을 올바르게 해석하도록 학습.
> - 단순히 키워드만 보는 것이 아니라, 문맥적 의미까지 고려해야 함.
> - "너무 무섭게 재미있었다" → 긍정
> - "재미있긴 했지만 너무 길었다" → 혼합적 뉘앙스 → 최종적으로 부정으로 분류될 수 있음
> 2. 일반화 성능 확보
> - 학습 데이터에만 맞추는 것이 아니라, 새로운 문장에서도 정확히 분류할 수 있어야 함.
> - 즉, 과적합을 피하고 다양한 표현 방식을 이해하는 능력을 키우는 것.
---
### 23. Transformer Sentiment Analysis(긍정/부정) 분류 모델 - IMDB 리뷰 데이터셋 적용, 영어 영화 리뷰(긍정/부정), Model - LLM/01_transformer(sentiment_analysis_imdb).ipynb
> Transformer Sentiment Analysis(긍정/부정) 분류 모델
> - 입력된 문장(리뷰, 댓글, 트윗 등)을 긍정(Positive) 또는 **부정(Negative)**으로 자동 분류하는 것.
> - 예: "이 영화 정말 재미있었어요!" → 긍정, "스토리가 지루하고 너무 길었어요." → 부정
> 1. 언어 이해 능력 강화
> - 모델이 문장의 어휘, 문맥, 뉘앙스를 파악해 감정을 올바르게 해석하도록 학습.
> - 단순히 키워드만 보는 것이 아니라, 문맥적 의미까지 고려해야 함.
> - "너무 무섭게 재미있었다" → 긍정
> - "재미있긴 했지만 너무 길었다" → 혼합적 뉘앙스 → 최종적으로 부정으로 분류될 수 있음
> 2. 일반화 성능 확보
> - 학습 데이터에만 맞추는 것이 아니라, 새로운 문장에서도 정확히 분류할 수 있어야 함.
> - 즉, 과적합을 피하고 다양한 표현 방식을 이해하는 능력을 키우는 것.
---
### 22. Deep Learning Hybrid(CNN + Attention) Image Captioning(CNN+Attention) 최종 문장생성 모델 - COCO 이미지 캡셔닝 데이터셋 사용, Model - 22_hybrid_coco_attention.ipynb
> 핵심 차이: "어디를 보고 말하는가?"
> - 기존 모델은 이미지 전체를 한 번에 요약해서 디코더에 넘긴다(마치 사진을 한 번 보고 기억으로 문장을 만드는 느낌)
> - Attention 모델은 문장을 생성할 때마다 이미지의 다른 위치를 다시 본다(마치 사진을 계속 보면서 "곰 얼굴", "잔디", "침대" 등 필요한 부분에 집중하는 느낌)
>> 학습 목표 : 이미지 캡션 생성을 위한 CNN-Attention 하이브리드 모델 구현 목표
> 1. Encoder : 이미지 → CNN으로 특징 추출 (ResNet-50 → [batch_size, 14×14, 2048] 공간 특징 유지)
> 2. Decoder : 이미지 특징 → Attention으로 시퀀스 생성 (이미지의 각 위치에 대한 attention + 캡션 시퀀스)
> 3. 문장 생성 방식 : 매 시점마다 이미지의 다른 위치에 집중하며 단어 생성
> 4. 표현력 : 객체, 배경, 위치 등 세부 정보 및 자연스럽고 정확한 문장 생성 가능
> 5. 시각화 : 각 단어가 이미지의 어느 부분을 보고 생성됐는지 시각화 가능 (attention map)
---
### 21. Deep Learning Hybrid(CNN + RNN) Image Captioning(CNN+RNN) 최종 문장생성 모델 - COCO 이미지 캡셔닝 데이터셋 사용, Model - 21_hybrid_coco.ipynb
> 하이브리드 구조 개념
> - CNN (Convolutional Neural Network): 이미지나 공간적 데이터를 처리하여 특징(feature)을 추출합니다.
> - RNN (Recurrent Neural Network) 또는 LSTM/GRU: 시계열적 특성을 가진 데이터를 처리하거나 CNN이 추출한 특징을 시퀀스로 간주해 순차적으로 처리합니다.
> Model - 21_hybrid_coco.ipynb
>> 학습 목표 : 이미지 캡션 생성을 위한 CNN-RNN 하이브리드 모델을 통한 최종 문장생성
> 1. CNN-RNN 하이브리드 이미지 캡셔닝 학습 팁
> 2. 이미지 → CNN으로 특징 추출
> 3. 캡션 → RNN으로 시퀀스 생성
> 4. 학습 데이터셋: MS COCO 추천
> 5. 손실 함수: nn.CrossEntropyLoss()
> 6. 옵티마이저: torch.optim.Adam
> 7. MS COCO 데이터셋은 이미지와 캡션이 포함되어 있음
---
### 20. Deep Learning Hybrid(CNN + RNN) Model - EMNIST 손글씨 숫자 + 알파벳 이미지 데이터셋 사용, Model - 20_deep_learning_hybrid_emnist.ipynb
> 하이브리드 구조 개념
> - CNN (Convolutional Neural Network): 이미지나 공간적 데이터를 처리하여 특징(feature)을 추출합니다.
> - RNN (Recurrent Neural Network) 또는 LSTM/GRU: 시계열적 특성을 가진 데이터를 처리하거나 CNN이 추출한 특징을 시퀀스로 간주해 순차적으로 처리합니다.
> Model - 20_deep_learning_hybrid_emnist.ipynb
> Streamlit 웹앱 기본 구조 
> - hybrid-emnist-streamlit/src/app_20_deep_learning_model_hybrid_emnist.py
> - hybrid-emnist-streamlit/src/model_utils.py
> - hybrid-emnist-streamlit/src/labels_map.json
> - hybrid-emnist-streamlit/models/model_hybrid_emnist.pt
1. Deep Learning Hybrid(CNN + RNN) 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. 웹에서 EMNIST 숫자 + 알파벳 분류기 웹앱 데모 - 사용자 입력 방식(테스트셋에서 무작위 이미지 선택, 사용자가 직접 이미지 업로드, 사용자가 직접 그리기), 모델 추론(학습된 Hybrid(CNN + RNN) 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력 (정답 vs 예측), Confusion Matrix 및 오차 분석, 틀린 예측 샘플 시각화))
---
### 19. Deep Learning Hybrid(CNN + RNN) Model - MNIST 손글씨 이미지 데이터셋 사용
> 하이브리드 구조 개념
> - CNN (Convolutional Neural Network): 이미지나 공간적 데이터를 처리하여 특징(feature)을 추출합니다.
> - RNN (Recurrent Neural Network) 또는 LSTM/GRU: 시계열적 특성을 가진 데이터를 처리하거나 CNN이 추출한 특징을 시퀀스로 간주해 순차적으로 처리합니다.
> Model - 19_deep_learning_hybrid.ipynb
> Streamlit 웹앱 기본 구조 
> - hybrid-mnist-streamlit/src/app_19_deep_learning_model_hybrid_mnist.py
> - hybrid-mnist-streamlit/src/model_utils.py
> - hybrid-mnist-streamlit/src/labels_map.json
> - hybrid-mnist-streamlit/models/model_hybrid_mnist.pt
1. Deep Learning Hybrid(CNN + RNN) 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. 웹에서 MNIST 숫자 분류기 웹앱 데모 - 사용자 입력 방식(테스트셋에서 무작위 이미지 선택, 사용자가 직접 이미지 업로드, 사용자가 직접 그리기), 모델 추론(학습된 Hybrid(CNN + RNN) 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력 (정답 vs 예측), Confusion Matrix 및 오차 분석, 틀린 예측 샘플 시각화))
---
### 18. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : GTSRB (German Traffic Sign Recognition Benchmark) - 교통 표지판(Traffic sign) 이미지 분류 이미지 데이터셋 사용
> Model - 18_transfer_learning_gtsrb_traffic_sign_detection.ipynb
> Streamlit 웹앱 기본 구조 
> - gtsrb-traffic-sign-detection-streamlit/src/app_18_transfer_learning_model_gtsrb_traffic_sign_detection.py
> - gtsrb-traffic-sign-detection-streamlit/src/model_utils.py
> - gtsrb-traffic-sign-detection-streamlit/src/labels_map.json
> - gtsrb-traffic-sign-detection-streamlit/models/model_transfer_learning_gtsrb_traffic_detection.pt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - 교통 표지판(Traffic sign) 예측 이미지 분류 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 교통 표지판(Traffic sign) 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 17. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : Kaggle Surface Crack Detection - 콘크리트 표면 결함 예측 이미지 데이터셋 사용
> Model - 17_transfer_learning_kaggle_surface_crack_detection.ipynb
> Streamlit 웹앱 기본 구조 
> - surface_crack-detection-streamlit/src/app_17_transfer_learning_model_surface_crack_detection.py
> - surface_crack-detection-streamlit/src/model_utils.py
> - surface_crack-detection-streamlit/src/labels_map.json
> - surface_crack-detection-streamlit/models/model_transfer_learning_surface_crack_detection.pt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - 콘크리트 표면 결함 예측 이미지 분류 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 콘크리트 표면 결함 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 16. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : Kaggle Breast Ultrasound Detection - 유방암 예측 이미지 데이터셋 사용
> Model - 16_transfer_learning_kaggle_breast_ultrasound_detection.ipynb
> Streamlit 웹앱 기본 구조 
> - breast-detection-streamlit/src/app_16_transfer_learning_model_breast_ultrasound_detection.py
> - breast-detection-streamlit/src/model_utils.py
> - breast-detection-streamlit/src/labels_map.json
> - breast-detection-streamlit/models/model_transfer_learning_covid19_detection.pt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - 유방암 예측 이미지 분류 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 유방암 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 15. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : Kaggle COVID19 Detection - COVID19 감염 예측 이미지 데이터셋 사용
> Model - 15_transfer_learning_kaggle_covid19_detection.ipynb
> Streamlit 웹앱 기본 구조 
> - covid19-detection-streamlit/src/app_15_transfer_learning_model_covid19_detection.py
> - covid19-detection-streamlit/src/model_utils.py
> - covid19-detection-streamlit/src/labels_map.json
> - covid19-detection-streamlit/models/model_transfer_learning_covid19_detection.pt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - COVID19 감염 예측 이미지 분류 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 COVID19 감염 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 14. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : Kaggle Face Emotion Detection Classification - Kaggle 얼굴에 감정 이미지 분류 데이터셋 사용
> Model - 14_transfer_learning_kaggle_emotion_detection.ipynb
> Streamlit 웹앱 기본 구조 
> - face-emotion-streamlit/src/app_14_transfer_learning_model_emotion_detection.py
> - face-emotion-streamlit/src/model_utils.py
> - face-emotion-streamlit/src/labels_map.json
> - face-emotion-streamlit/models/model_transfer_learning_emotion_detection.pt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - Kaggle 얼굴 감정 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 얼굴 감정 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 13. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : Kaggle Face Mask Detection Classification - Kaggle 얼굴에 마스크(face mask detection) 이미지 분류 데이터셋 사용
> Model - 13_transfer_learning_kaggle_face_mask_detection.ipynb
> Streamlit 웹앱 기본 구조 
> - face-mask-streamlit/src/app_13_transfer_learning_model_face_mask_detection.py
> - face-mask-streamlit/src/model_utils.py
> - face-mask-streamlit/src/labels_map.json
> - face-mask-streamlit/models/model_transfer_learning_face_mask_detection.ckpt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - Kaggle 얼굴에 마스크(face mask detection) 착용 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 얼굴에 마스크 착용 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 12. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : Kaggle brain tumor Image Classification (MRI) - Kaggle 뇌종양(Brain Tumor) 이미지 분류 데이터셋 사용
> Model - 12_transfer_learning_kaggle_brain_tumor_mri.ipynb
> Streamlit 웹앱 기본 구조 
> - brain-tumor-streamlit/src/app_12_transfer_learning_model_brain_tumor.py
> - brain-tumor-streamlit/src/model_utils.py
> - brain-tumor-streamlit/src/labels_map.json
> - brain-tumor-streamlit/models/model_transfer_learning_brain_tumor_mri.ckpt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - Kaggle 뇌종양(Brain Tumor) 이미지 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 뇌종양(brain tumor) 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 10. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : 강아지 감정 Dataset 사용
> Model - 11_transfer_learning_vit_dog_emotion_gpu.ipynb
> Streamlit 웹앱 기본 구조 
> - dogs-image-streamlit/src/app_11_transfer_learning_model_dog_emotion.py
> - dogs-image-streamlit/src/model_utils.py
> - dogs-image-streamlit/src/labels_map.json
> - dogs-image-streamlit/models/model_transfer_learning_dog_emotion.ckpt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - 강아지 감정 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 강아지 감정 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 9. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : 강아지 종 Dataset 사용
> Model - 10_transfer_learning_vit_custom_image_gpu.ipynb
> Streamlit 웹앱 기본 구조 
> - dogs-image-streamlit/src/app_10_transfer_learning_model_dog_image.py
> - dogs-image-streamlit/src/model_utils.py
> - dogs-image-streamlit/src/labels_map.json
> - dogs-image-streamlit/models/model_transfer_learning_dog_image.ckpt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - 강아지 종 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 무슨 강아지인지 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 8. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : 고양이와 강아지 Dataset 사용
> Model - 09_transfer_learning_cats_dogs_gpu.ipynb
> Streamlit 웹앱 기본 구조 
> - cats-dogs-streamlit/src/app_08_transfer_learning_model_cats_dogs.py
> - cats-dogs-streamlit/src/model_utils.py
> - cats-dogs-streamlit/models/transfer_learning_model_cats_dogs.pth
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - 고양이와 강아지 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 고양이인지 강아지인지 예측
---
### 7. Deep CNN(Convolution Neural Network) : CIFAR10 Dataset 사용
> Model - 07_deep_cnn_cifar10_gpu.ipynb
> Streamlit 웹앱 - app_07_deep_cnn_cifar10.py
1. Deep CNN 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. 웹에서 CIFAR10 숫자 분류기 웹앱 데모 - 사용자 입력 방식(사용자가 직접 이미지 업로드), 모델 추론(학습된 CNN 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력))
---
### 6. CNN(Convolution Neural Network) : CIFAR10 Dataset 사용
> Model - 06_cnn_cifar10_gpu.ipynb
> Streamlit 웹앱 - app_06_cnn_cifar10.py
1. CNN 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 학습률 개선, StepLR(일정 에폭마다 학습률을 감소), EarlyStopping(일정 에폭 동안 성능 향상이 없을 경우 학습을 조기 중단, 과적합 방지, 학습 시간 절약) 
6. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
7. 모델 저장 및 불러오기
8. 테스트 및 시각화
9. 웹에서 CIFAR10 숫자 분류기 웹앱 데모 - 사용자 입력 방식(사용자가 직접 이미지 업로드), 모델 추론(학습된 CNN 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력))
---
### 5. CNN(Convolution Neural Network) : Fashion MNIST Dataset 사용
> Model - 05_cnn_fashion_mnist_gpu.ipynb
> Streamlit 웹앱 - app_05_cnn_fashion_mnist.py
1. CNN 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 학습률 개선, StepLR(일정 에폭마다 학습률을 감소), EarlyStopping(일정 에폭 동안 성능 향상이 없을 경우 학습을 조기 중단, 과적합 방지, 학습 시간 절약) 
6. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
7. 모델 저장 및 불러오기
8. 테스트 및 시각화
9. 웹에서 MNIST 숫자 분류기 웹앱 데모 - 사용자 입력 방식(사용자가 직접 이미지 업로드), 모델 추론 학습된 CNN 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력 (정답 vs 예측))
---
### 4. CNN(Convolution Neural Network) : 필기체손글씨 MNIST Dataset 사용
> Model - 04_cnn_mnist_gpu.ipynb
> Streamlit 웹앱 - app_04_cnn_mnist.py
1. CNN 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 학습률 개선, StepLR(일정 에폭마다 학습률을 감소), EarlyStopping(일정 에폭 동안 성능 향상이 없을 경우 학습을 조기 중단, 과적합 방지, 학습 시간 절약) 
6. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
7. 모델 저장 및 불러오기
8. 테스트 및 시각화
9. 웹에서 MNIST 숫자 분류기 웹앱 데모 - 사용자 입력 방식(테스트셋에서 무작위 이미지 선택, 사용자가 직접 이미지 업로드, 사용자가 직접 그리기), 모델 추론(학습된 CNN 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력 (정답 vs 예측), Confusion Matrix 및 오차 분석, 틀린 예측 샘플 시각화))
---
### 3. MLP(Multi-Layer Perceptron) DeepLearning : Fashion MNIST Dataset 사용
> Model - 03_mlp_fashion_mnist_gpu.ipynb
> Streamlit 웹앱 - app_03_mlp_fashion_mnist.py
1. MLP 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 학습률 개선, StepLR(일정 에폭마다 학습률을 감소), EarlyStopping(일정 에폭 동안 성능 향상이 없을 경우 학습을 조기 중단, 과적합 방지, 학습 시간 절약) 
6. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
7. 모델 저장 및 불러오기
8. 테스트 및 시각화
9. 웹에서 Fashion MNIST 숫자 분류기 웹앱 데모 - 사용자 입력 방식(테스트셋에서 무작위 이미지 선택), 모델 추론(학습된 MLP 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력 (정답 vs 예측), Confusion Matrix 및 오차 분석, 틀린 예측 샘플 시각화))
---
### 2. MLP(Multi-Layer Perceptron) DeepLearning : 필기체손글씨 MNIST Dataset 사용
> Model - 02_mlp_mnist_gpu.ipynb
> Streamlit 웹앱 - app_02_mlp_mnist_model.py, app_02_mlp_mnist_model_image_upload.py
1. MLP 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가 
5. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
6. 모델 저장 및 불러오기
7. 테스트 및 시각화
8. 웹에서 MNIST 숫자 분류기 웹앱 데모 - 사용자 입력 방식(테스트셋에서 무작위 이미지 선택, 사용자가 직접 이미지 업로드, 사용자가 직접 그리기), 모델 추론(학습된 MLP 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력 (정답 vs 예측), Confusion Matrix 및 오차 분석, 틀린 예측 샘플 시각화))
---
### 1. MLP(Multi-Layer Perceptron) DeepLearning : basic dataset 사용
> Model - 01_mlp.ipynb
> Streamlit 웹앱 - app_01_mlp_model.py, app_01_mlp_model_csv_upload.py, app_01_mlp_model_csv_upload_download.py
1. MLP 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. 웹에서 MLP 이진 뷴류기 데모 - 사용자가 숫자를 입력하면 예측 결과가 바로 표시되고, 입력값에 해당하는 위치에 빨간 선이 그려진 예측 곡선이 함께 나타남, CSV 업로드를 통한 배치 예측 및 시각화, 예측 결과를 CSV 파일로 다운로드할 수 있도록 기능, 이렇게 하면 사용자가 업로드한 데이터에 대한 예측 결과를 저장하고 활용
---