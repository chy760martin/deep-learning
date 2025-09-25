# Fashion MNIST MLP 모델을 Streamlit 으로 웹서비스화

import streamlit as st
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import os

# 모델 클래스 정의(MLPDeepLearningModel)
class MLPDeepLearningModel(nn.Module):
    # model 정의 - 아키텍처를 구성하는 다양한 계층(layer)을 정의
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.flatten(x) # 입력층
        x = self.fc1(x) # 은닉층
        x = self.relu(x) # 활성화함수 ReLU(비선형함수)
        x = self.dropout(x) # overfitting 방지
        x = self.fc2(x) # 출력층
        return x

# 라벨 맵 정의
labels_map = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

# 경로 구성
base_dir = os.path.dirname(__file__)  # 현재 파일 기준 디렉토리
model_path = os.path.join(base_dir, '..', 'models', 'model_mlp_fashion_mnist.ckpt')

# 모델 로딩
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPDeepLearningModel().to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Grayscale(), # Fashion MNIST는 흑백
    transforms.Resize( (28, 28) ), # 사이즈 조정
    transforms.ToTensor() # 이미지 0~255 값을 가지는데, 0~1사이 값으로 변환
])

# Streamlit UI 생성
st.title('Fashion MNIST 이미지 분류기')
uploaded_file = st.file_uploader('이미지를 업로드하세요 (28 X 28 흑백)', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    # st.image(image, caption='업로드된 이미지', use_column_width=True)
    st.image(image, caption='업로드된 이미지', width='stretch')

    # 전처리 및 추론
    img_tensor = transform(image).unsqueeze(0) # 차원 추가
    pil_image = TF.to_pil_image(img_tensor.squeeze())
    st.image(pil_image, caption='전처리된 이미지', width='stretch')
    with torch.no_grad():
        output = model(img_tensor) # 모델 추론
        _, pred = torch.max(output, dim=1) # 모델 추론값 추출
        label = labels_map[pred.item()]
    
    st.success(f'예측 결과: **{label}**')

    # 예측 확률 시각화 추가 위치
    probs = torch.nn.functional.softmax(output, dim=1)
    st.subheader("📊 클래스별 예측 확률")
    for i, p in enumerate(probs[0]):
        st.write(f"{labels_map[i]}: {p.item():.2%}")