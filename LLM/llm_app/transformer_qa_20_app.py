# FastAPI 추론 서비스
# - /llm_app/transformer_qa_20_app.py
# - FastAPI 구동: 터미널에서 구동, uvicorn transformer_qa_20_app:app --reload
# - Postman app
# - API 코드로 테스트: Python, Java...

from fastapi import FastAPI # REST API 서버 구축용 프레임워크
from pydantic import BaseModel # 입력 데이터 검증 및 구조 정의
import torch
from torch.amp import autocast # PyTorch, AMP(Automatic Mixed Precision) 추론 최적화
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from peft import LoraConfig, get_peft_model

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 베이스 모델 로드
MODEL_NAME = "monologg/koelectra-base-v3-finetuned-korquad"
# 토크나이저
# - KorQuAD 1.0/2.0: 위키 문서 기반, 질문-답변 쌍 포함, 내부적으로 ElectraTokenizerFast
# - Fast 토크나이저라서 offset_mapping 사용 가능
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME) # Classification 베이스 모델 로드, DistilBERT 기반 이진 분류 모델 (긍정/부정)

# Feature Extraction
for name, param in model.named_parameters():
    if 'lora' not in name: # LoRA 모듈이 아닌 경우
        param.requires_grad = False # 모델 본체 동결

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['query', 'value'], # attention 모듈에 LoRA 적용
    lora_dropout=0.1,
    bias='none',
    task_type='QUESTION_ANS'
)
model = get_peft_model(model, lora_config)

# 저장된 가중치 로드
# torch.load() 파일에서 파라미터(가중치) 딕셔너리를 불러옴
# model.load_state_dict() 불러온 파리미터를 모델 구조에 맞게 적용
model.load_state_dict(torch.load('../llm_models/20_transformer_qa/best_model.pt'))
model.to(device)

# Fast API 생성
app = FastAPI()

# 요청 데이터 구조 정의
class Query(BaseModel):
    question: str
    context: str

@app.post('/qa')
def qa_service(request: Query):
    # 토크나이저
    inputs = tokenizer(
        request.question,
        request.context,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt',
        return_offsets_mapping=True
    )

    # 모델 입력 준비(offset_mapping 구조 제외, 핵심 구조만 모델에 전달)
    model_inputs = {
        'input_ids': inputs['input_ids'].to(device),
        'attention_mask': inputs['attention_mask'].to(device),
        'token_type_ids': inputs['token_type_ids'].to(device)
    }

    # 추론
    model.eval() # 검증/추론 모드 전환
    with torch.no_grad():
        outputs = model(**model_inputs)
    
    # logits 저장, 모델이 출력한 시작/끝 위치 logits, 후처리 함수는 이 logits을 이용해 가장 가능성이 높은 답변 스팬을 선택한다
    start_idx = outputs.start_logits.argmax()
    end_idx = outputs.end_logits.argmax()

    # offset_mapping으로 답변 복원
    offsets = inputs['offset_mapping'][0].tolist()
    start_char, end_char = offsets[start_idx][0], offsets[end_idx][1]
    answer = request.context[start_char:end_char]

    # 결과 반환
    return {
        'question': request.question,
        'answer': answer
    }