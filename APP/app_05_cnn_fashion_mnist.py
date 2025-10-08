# Fashion MNIST CNN 모델을 Streamlit 으로 웹서비스화
# 사용자가 이미지를 업로드하면 예측 결과가 바로 표시되고, 클래스별 예측 확률이 막대그래프로 나타남
# 사용법 : ./APP> streamlit run app_05_cnn_fashion_mnist.py

import streamlit as st
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import os

# 모델 클래스 정의(MLPDeepLearningModel)
# 딥러닝 CNN 아키텍처 : Feature Extractor(Conv>Pool>Conv>Pool) -> Fully-Connected(Flatten) -> Classification(Linear>Dropout->Linear)
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)        
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)        
        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout25 = nn.Dropout(p=0.25)
        self.dropout50 = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm2d(32) # BatchNorm2d를 추가하면 일반화 성능 향상 가능
        self.bn2 = nn.BatchNorm2d(64)
        self.act = nn.LeakyReLU(0.01) # LeakyReLU는 음수 입력에 대해 아주 작은 기울기(예: 0.01)를 유지, 이렇게 하면 뉴런이 완전히 죽지 않고, 미세하게라도 학습에 참여할 수 있다.

    
    def forward(self, x):
        # 패딩이 적용되었기 때문에 컨볼루션층을 통과한 데이터는 크기는 변하지 않고 필터 개수와 동일하게 출력채널 개수만 변함, 맥스풀링층을 통과한 데이터는 1/2로 바뀌지만 채널 개수는 변하지 않음
        # conv1, data shape = (H, W, C) = (28, 28, 1)
        x = self.bn1(self.conv1(x)) # (28, 28, 1)
        # x = torch.relu(x) # (28, 28, 32)
        x = self.act(x)
        x = self.pooling(x) # (28, 28, 32)
        x = self.dropout25(x) # (14, 14, 32)
        # conv2
        x = self.bn2(self.conv2(x)) # (14, 14, 32)
        # x = torch.relu(x) # (14, 14, 64)
        x = self.act(x)
        x = self.pooling(x) # (14, 14, 64)
        x = self.dropout25(x) # (7, 7, 64)
        # (높이,너비,채널)3차원 텐서이므로 완전연결층(Fully-Connected)과 연결을 위해 view() 명령어를 이용해서 3차원 텐서를 1차원 vector로 만들어 주는 역할
        # 여기서 -1은 PyTorch에게 "배치 크기는 자동으로 계산해줘"라는 의미, 7 * 7 * 64는 Conv 레이어를 통과한 후의 feature map 크기를 하드코딩한 것이다.
        # x = x.view(-1, 7 * 7 * 64) # 이 방식은 입력 이미지가 항상 28×28이고, Conv 구조가 고정되어 있을 때만 잘 작동, 단점: 만약 이미지 크기나 Conv 구조가 바뀌면 오류 발생 가능성이 높습니다.
        
        # x.size(0)은 현재 배치 크기를 의미, -1은 나머지 모든 차원을 자동으로 펼쳐서 1차원 벡터로 만들어준다., 장점: 유연하고, 재사용 가능한 모델 구조를 만들 수 있다.
        # 예시) x.shape = [32, 64, 7, 7]
        # x.view(-1, 3136)         # 하드코딩 방식
        # x.view(x.size(0), -1)    # 자동 계산 방식 → 32 × 3136
        x = x.view(x.size(0), -1)

        # Linear
        x = self.fc1(x)
        # x = torch.relu(x)
        x = self.act(x)
        x = self.dropout50(x)
        
        x = self.fc2(x)

        return x

# 라벨 맵 정의
labels_map = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

# 경로 구성
base_dir = os.path.dirname(__file__)  # 현재 파일 기준 디렉토리
model_path = os.path.join(base_dir, '..', 'models', 'model_cnn_fashion_mnist.ckpt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(DEVICE)
# model.load_state_dict(torch.load(model_path, map_location=DEVICE))

# 모델 로드 : 중단된 지점부터 이어서 학습
checkpoint = torch.load(model_path, map_location=DEVICE)# GPU/CPU 환경에 따라 로딩 시 map_location=DEVICE 옵션을 추가하여 안전하게 한다.
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Grayscale(), # RGB 이미지를 흑백으로 변환하는 과정(Fashion MNIST Dataset 흑백)
    transforms.Resize( (28, 28) ), # 사이즈 조정
    transforms.ToTensor(), # 이미지 0~255 값을 가지는데, 0~1사이 값으로 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화 추가
])

# Streamlit UI 생성
st.title('Fashion MNIST 이미지 분류기')
uploaded_file = st.file_uploader('이미지를 업로드하세요 (28 X 28 흑백)', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    # st.image(image, caption='업로드된 이미지', use_column_width=True)
    st.image(image, caption='업로드된 이미지', width='stretch')

    # 전처리 및 추론
    img_tensor = transform(image).unsqueeze(0).to(DEVICE) # 배치 차원 추가
    # pil_image = TF.to_pil_image(img_tensor.squeeze())
    pil_image = TF.to_pil_image(img_tensor.squeeze(0)) # 배치 차원만 제거
    st.image(pil_image, caption='전처리된 이미지', width='stretch')
    with torch.no_grad():
        output = model(img_tensor).to(DEVICE) # 모델 추론
        _, pred = torch.max(output, dim=1) # 모델 추론값 추출
        label = labels_map[pred.item()]
    
    st.success(f'예측 결과: **{label}**')

    # 예측 확률 시각화
    # probs = torch.nn.functional.softmax(output, dim=1)
    probs = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
    st.write(probs)
    st.subheader("클래스별 예측 확률")   
    
    # 예측 확률 막대그래프
    fig, ax = plt.subplots()
    ax.bar(labels_map.values(), probs)
    st.pyplot(fig)

    # for i, p in enumerate(probs[0]):
    for i, p in enumerate(probs):
        st.write(f"{labels_map[i]}: {p.item():.2%}")
