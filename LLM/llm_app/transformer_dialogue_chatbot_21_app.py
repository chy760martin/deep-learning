# FastAPI 추론 서비스
# - /llm_app/transformer_qa_20_app.py
# - FastAPI 구동: 터미널에서 구동, uvicorn transformer_dialogue_chatbot_21_app:app --reload
# - Postman app
# - API 코드로 테스트: Python, Java...
# body 예시: {
#   "text": "오늘은 관광지에 대해서 물어보려고 하는데 그런 것도 잘 알고 있어?",
#   "max_length": 128,
#   "min_length": 10,
#   "num_beams": 5,
#   "early_stopping": true,
#   "do_sample": true,
#   "top_k": 60,
#   "top_p": 0.9,
#   "temperature": 0.9,
#   "num_return_sequences": 1,
#   "repetition_penalty": 1.5,
#   "no_repeat_ngram_size": 4,
#   "length_penalty": 1.2
# }

from fastapi import FastAPI # REST API 서버 구축용 프레임워크
from pydantic import BaseModel # 입력 데이터 검증 및 구조 정의
import torch
from torch.amp import autocast # PyTorch, AMP(Automatic Mixed Precision) 추론 최적화
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
import re

# GPU 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'PyTorch Version: {torch.__version__}, Device: {device}')

# 같은 구조의 모델 초기화
tokenizer = AutoTokenizer.from_pretrained('google/mt5-small', legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained('google/mt5-small', tie_word_embeddings=False, use_safetensors=True) # MT5 모델 불러온다

# 모델 본체 동결 처리: Feature Extraction
for param in model.parameters():
    param.requires_grad=False
        
# LoRA 적용: LoRA 모듈만 학습되도록 설정(경량 파인튜닝)
lora_config = LoraConfig(
    r=8, # 작은 rank(r=8)로 효율적인 파인튜닝 가능
    lora_alpha=32, # LoRA scaling factor
    # target_modules=['q', 'v'], # Attention 모듈의 Query/Value 부분에 LoRA 레이어 추가
    target_modules=['q', 'v'], # Attention 모듈의 Query/Value 부분에 LoRA 레이어 추가
    lora_dropout=0.1, # 드롭아웃
    bias='none',
    task_type='SEQ_2_SEQ_LM' # 대화 응답 생성은 Seq2Seq LM
)
# LoRA 모델 생성
model = get_peft_model(model, lora_config)

# 저장된 state_dict 불러오기
model.load_state_dict(torch.load('../llm_models/21_transformer_dialogue_chatbot/best_model.pt'))

# 추론/검증 모드 적용
model.eval()

# 모델 전체를 GPU/CPU 디바이스 메모리로 이동
model = model.to(device)

# Fast API 서버 구성
app = FastAPI()

# Body(JSON)로 보내려면 BaseModel을 사용하여 요청도 JSON 형식으로 보내야한다
# - 예시) {'text': '입력 문장'} 을 body에 넣으면 정상 동작된다
class InputText(BaseModel):
    session_id: str
    text: str
    max_length: int=128 # 기본값 설정
    min_length: int=10 # 최소 길이 강제

    num_beams: int=5 # 5~7 정도가 대화형 모델에서는 가장 균형이 좋다
    early_stopping: bool=True
    do_sample: bool=True
    top_p: float=0.9 # 0.9 → 0.6로 줄여서 불필요한 변형을 줄인다, 변형 단어 출력 감소
    top_k: int=60 # 상위 후보 샘플링
    temperature: float=0.9 # 0.7 → 0.5 정도로 낮추면 더 안정적인 답변을 얻을 수 있다
    num_return_sequences: int=1

    repetition_penalty: float=1.5 # 반복 억제
    no_repeat_ngram_size: int=4 # 4-gram 반복 금지
    length_penalty: float=1.2 # 1.2~2.0 정도로 설정하면, 모델이 불필요하게 길게 반복하는 걸 줄인다

histories = {}

@app.post('/predict')
async def predict(input_text: InputText):
    # 여러 사용자가 동시에 접속해도 세션별로 히스토리를 관리
    if input_text.session_id not in histories: # 세션 아이디가 없으면 공백 초기화
        histories[input_text.session_id] = [] # 문자열 대신 리스트로 초기화
    # histories[input_text.session_id] += f'사용자: {input_text.text}\n' # 문자열 사용자 입력을 히스토리에 추가
    histories[input_text.session_id].append({'role': 'user', 'content': input_text.text})

    # 입력을 토크나이저로 변환
    inputs = tokenizer(
        input_text.text,
        return_tensors='pt'
    )

    # 입력 텐서를 모델과 같은 디바이스로 이동
    inputs = { k: v.to(device) for k, v in inputs.items() }

    # 추론 모드이므로 미분 연산 하지 않음
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_length=input_text.max_length,
                min_length=input_text.min_length,
                
                num_beams=input_text.num_beams,
                early_stopping=input_text.early_stopping,
                do_sample=input_text.do_sample,
                top_p=input_text.top_p,
                top_k=input_text.top_k,
                temperature=input_text.temperature,
                num_return_sequences=input_text.num_return_sequences,

                repetition_penalty=input_text.repetition_penalty,
                no_repeat_ngram_size=input_text.no_repeat_ngram_size,
                length_penalty=input_text.length_penalty            
            )

    def clean_output(decoded: str) -> str:
        # 특수 토큰 제거
        decoded = re.sub(r'<(extra_id_\d+|pad|unk)>', '', decoded)
        # 공백 정리
        decoded = decoded.strip()
        # 중복 문장부호 정리
        decoded = re.sub(r'([.!?,])\1+', r'\1', decoded)
        # 쉼표 기준 반복 제거
        tokens = [t.strip() for t in decoded.split(",")]
        unique_tokens = []
        for t in tokens:
            if t and t not in unique_tokens:
                unique_tokens.append(t)
        decoded = ", ".join(unique_tokens)
        # 연속된 동일 단어 제거
        decoded = re.sub(r'\b(\w+)( \1)+\b', r'\1', decoded)
        return decoded
    
    decoded_list = [clean_output(tokenizer.decode(o, skip_special_tokens=True)) for o in outputs]

    # 모델 응답을 히스토리에 추가
    # histories[input_text.session_id] += f'모델: {decoded_list[0]}\n' # 문자열 저장
    histories[input_text.session_id].append({'role': 'model', 'content': decoded_list[0]})
    return {"result": decoded_list, 'history': histories[input_text.session_id]}