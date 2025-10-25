# app_08_transfer_learning_model_cats_dogs.py
# Streamlit 앱 - 고양이와 강아지 분류기
# - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현
# - 사용자가 이미지를 업로드하면 모델이 고양이인지 강아지인지 예측
# - 모델은 이전에 학습된 전이 학습 모델 사용
# - 모델 로딩, 이미지 전처리, 예측, UI 구성 등 포함
# - 모델은 model_utils.py에서 정의된 TransferLearningModel 클래스를 사용
# - 모델 가중치는 ../models/transfer_learning_model_cats_dogs.pth에 저장
# - Streamlit의 캐시 기능을 사용하여 모델 로딩 최적화
# - 이미지 업로드, 전처리, 예측 결과 표시 기능 포함
# - UI는 제목, 설명, 파일 업로더, 이미지 표시, 예측 결과 표시로 구성
# - 사용자는 jpg, jpeg, png 형식의 이미지 업로드 가능
# - 예측 결과는 "고양이" 또는 "강아지"로 표시
# - 앱은 GPU가 사용 가능하면 GPU를 사용하여 모델 추론 수행
# - 이미지 전처리는 torchvision.transforms를 사용하여 크기 조정, 텐서 변환, 정규화 포함
# - 예측은 torch.no_grad() 컨텍스트에서 수행하여 메모리 사용 최적화
# - 모델은 eval() 모드로 설정하여 드롭아웃 등 비활성화
# - 업로드된 이미지는 PIL 라이브러리를 사용하여 열고 RGB로 변환
# - 업로드된 이미지는 Streamlit의 st.image()로 표시
# - 예측된 클래스는 0(고양이) 또는 1(강아지)로 반환
# - 예측 결과는 st.success()로 강조 표시
# - 앱은 사용자가 쉽게 접근하고 사용할 수 있도록 설계
# - 모델은 torchvision의 vit_b_16 사전학습 모델을 기반으로 함

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from model_utils import TransferLearningModel

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로딩
@st.cache_resource # - @st.cache_resource: 모델을 한 번만 로딩하고 캐시하여 앱 속도 향상
def load_model():
    base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT) # - vit_b_16: Vision Transformer 사전학습 모델
    model = TransferLearningModel(base_model, feature_extractor=True, num_classes=2).to(device) # - feature_extractor=True: 특징 추출기로 사용, num_classes=2: 고양이와 강아지 두 클래스
    model.load_state_dict(torch.load('../models/transfer_learning_model_cats_dogs.pth', map_location=device)) # - 학습된 모델 가중치 로드
    model.eval() # - 추론 모드로 전환
    return model

model = load_model()

# 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 이미지 크기를 224×224로 조정 (ViT 입력 크기)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device) # - unsqueeze(0): 배치 차원 추가 → [1, 3, 224, 224], 배치 차원 추가 및 디바이스로 이동

# 예측 함수
def predict(image_tensor):
    with torch.no_grad(): # - 추론 시에는 기울기 계산 불필요
        outputs = model(image_tensor) # - 모델에 이미지 텐서 입력
        _, predicted = torch.max(outputs, 1) # - 가장 높은 점수를 가진 클래스 선택
    return predicted.item() # - 예측된 클래스 인덱스 반환 (0: 고양이, 1: 강아지)

# Streamlit UI
st.title("고양이 vs 강아지 분류기") # - 앱 제목
st.write("이미지를 업로드하면 AI가 고양이인지 강아지인지 예측해줍니다!") # - 앱 설명

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png" ]) # - 파일 업로더
if uploaded_file is not None: # - 파일이 업로드되었을 때
    image = Image.open(uploaded_file).convert("RGB") # - 이미지를 RGB로 변환
    # st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.image(image, caption="업로드된 이미지", use_container_width=True) # - use_container_width=True: 컨테이너 너비에 맞게 이미지 크기 조정

    image_tensor = preprocess_image(image) # - 이미지 전처리
    prediction = predict(image_tensor) # - 예측 수행

    label = "고양이" if prediction == 0 else "강아지" # - 예측 결과에 따른 라벨 설정
    st.success(f"예측 결과: **{label}** 입니다!")