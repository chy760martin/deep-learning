# FastAPI 추론 서비스

from fastapi import FastAPI # REST API 서버 구축용 프레임워크
from pydantic import BaseModel # 입력 데이터 검증 및 구조 정의
import torch
from torch.amp import autocast # PyTorch, AMP(Automatic Mixed Precision) 추론 최적화
from transformers import DistilBertTokenizer # Hugging Face의 DistilBERT 토크나이저
from transformers import DistilBertForSequenceClassification # Hugging Face의 DistilBERT 토크나이저
from peft import LoraConfig, get_peft_model

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 베이스 모델 로드
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME) # tokenizer 베이스 모델 로드
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME) # Classification 베이스 모델 로드, DistilBERT 기반 이진 분류 모델 (긍정/부정)

# Feature Extraction + LoRA Fine-tuning 조합, 분류 레이어와 LoRA 모듈만 학습
for param in model.distilbert.parameters(): # 본체 동결(feature extractor)
    param.requires_grad = False

# Attention의 q_lin, v_lin 레이어에서 LoRA가 768차원 → 8차원 축소 → 768차원 복원 과정을 거쳐 업데이트를 추가하는 구조
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q_lin', 'v_lin'], # DistilBERT attention
    bias='none',
    task_type='SEQ_CLS'
)
model = get_peft_model(model, lora_config)

# 저장된 가중치 로드
# torch.load() 파일에서 파라미터(가중치) 딕셔너리를 불러옴
# model.load_state_dict() 불러온 파리미터를 모델 구조에 맞게 적용
model.load_state_dict(torch.load('../llm_models/18_transformer_classifier_sentiment/best_model.pt'))
model.to(device)
model.eval() # 검증/추론 모드 전환

# FastAPI 인스턴스 생성
app = FastAPI()

# 입력 데이터 구조 정의: 
# - API 요청 시 JSON 데이터 구조 정의, 예시: {"text": "I love this movie!"}
class TextInput(BaseModel):
    text: str

# 추론 함수 정의
def predict(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True
    ).to(device)

    # 검증/추론시 미분 연산 하지 않음
    with autocast(device_type='cuda', dtype=torch.float16): # AMP(autocast) 적용 → GPU에서 FP16 연산으로 속도 최적화
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=-1).item()
    return 'Positive' if pred == 1 else 'Negative'

# /predict 엔드포인트에 POST 요청 시 추론 수행
# - 응답: JSON 형태 → {"sentiment": "Positive"}
@app.post('/predict')
def predict_sentiment(input: TextInput):
    result = predict(input.text, tokenizer, model, device)
    return {'sentiment': result}