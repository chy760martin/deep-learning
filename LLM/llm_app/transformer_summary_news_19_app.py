# FastAPI 추론 서비스
# - /llm_app/transformer_summary_news_19_app.py
# - FastAPI 구동: 터미널에서 구동, uvicorn transformer_summary_news_19_app:app --reload
# - 윈도우 파워쉘: Invoke-RestMethod -Uri "http://127.0.0.1:8000/summarize" -Method Post -ContentType "application/json" -Body '{"text":"I really love this movie, it was fantastic!"}'
# - Postman app
# - API 코드로 테스트: Python, Java...

from fastapi import FastAPI # REST API 서버 구축용 프레임워크
from pydantic import BaseModel # 입력 데이터 검증 및 구조 정의
import torch
from torch.amp import autocast # PyTorch, AMP(Automatic Mixed Precision) 추론 최적화
from transformers import BartTokenizer
from transformers import BartForConditionalGeneration
from peft import LoraConfig, get_peft_model

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 베이스 모델 로드
MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME) # tokenizer 베이스 모델 로드
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME) # Classification 베이스 모델 로드, DistilBERT 기반 이진 분류 모델 (긍정/부정)

# Feature Extraction
for name, param in model.named_parameters():
    if 'lora' not in name: # LoRA 모듈이 아닌 경우
        param.requires_grad = False # 모델 본체 동결

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias='none',
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

# 저장된 가중치 로드
# torch.load() 파일에서 파라미터(가중치) 딕셔너리를 불러옴
# model.load_state_dict() 불러온 파리미터를 모델 구조에 맞게 적용
model.load_state_dict(torch.load('../llm_models/19_transformer_summary_news/best_model.pt'))
model.to(device)
model.eval() # 검증/추론 모드 전환

# Fast API 생성
app = FastAPI()

# 요청 데이터 구조 정의
class SummarizeRequest(BaseModel):
    text: str
    max_length: int=128
    num_beams: int=4

@app.post('/summarize')
def summarize(request: SummarizeRequest):
    inputs = tokenizer(
        request.text,
        max_length=1024,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)

    summary_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=request.max_length,
        num_beams=request.num_beams,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {'summary': summary}