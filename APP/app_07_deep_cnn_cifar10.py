# CIFAR10 Dataset, Deep CNN 모델을 Streamlit 으로 웹서비스화
# 사용자가 이미지를 업로드하면 예측 결과가 바로 표시되고, 클래스별 예측 확률이 막대그래프로 나타남
# 사용법 : ./APP> streamlit run app_07_deep_cnn_cifar10.py

import streamlit as st
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import os

# 딥러닝 Deep CNN 아키텍처 
# - Feature Extractor(Conv>Pool>Conv>Pool) -> Fully-Connected(Flatten) -> Classification(Linear>Dropout->Linear)
# <Feature Extractor>
# conv1>relu>conv2>relu>pooling>dropout -> conv3>relu>conv4>relu>pooling>dropout -> conv5>relu>pooling>dropout -> conv6>relu>pooling>dropout -> conv7>relu>pooling>dropout
# <Fully-Connected>
# Flatten 
# <Classification>
# fc1>relu>Dropout -> fc2

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # conv1, conv2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # conv3, conv4
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # conv5
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # conv6
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # conv7
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
                
        # pooling
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # fc1, fc2
        self.fc1 = nn.Linear(1 * 1 * 256, 128)
        self.fc2 = nn.Linear(128, 10)

        # dropout
        self.dropout25 = nn.Dropout(p=0.25)
        self.dropout50 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        # 패딩이 적용되었기 때문에 컨볼루션층을 통과한 데이터는 크기는 변하지 않고 필터 개수와 동일하게 출력채널 개수만 변함
        # 맥스풀링층을 통과한 데이터는 1/2로 바뀌지만 채널 개수는 변하지 않음

        # conv1, conv2 data shape = (H, W, C) = (32, 32, 3)
        x = self.conv1(x) # (32, 32, 3)
        x = torch.relu(x) # (32, 32, 32)
        x = self.conv2(x) # (32, 32, 32)
        x = torch.relu(x) # (32, 32, 32)
        x = self.pooling(x) # (32, 32, 32)
        x = self.dropout25(x) # (16, 16, 32)

        # conv3, conv4
        x = self.conv3(x) # (16, 16, 32)
        x = torch.relu(x) # (16, 16, 64)
        x = self.conv4(x) # (16, 16, 64)
        x = torch.relu(x) # (16, 16, 64)
        x = self.pooling(x) # (16, 16, 64)
        x = self.dropout25(x) # (8, 8, 64)

        # conv5
        x = self.conv5(x) # (8, 8, 64)
        x = torch.relu(x) # (8, 8, 128)
        x = self.pooling(x) # (8, 8, 128)
        x = self.dropout25(x) # (4, 4, 128)

        # conv6
        x = self.conv6(x) # (4, 4, 128)
        x = torch.relu(x) # (4, 4, 128)
        x = self.pooling(x) # (4, 4, 128)
        x = self.dropout25(x) # (2, 2, 128)

        # conv7
        x = self.conv7(x) # (2, 2, 128)
        x = torch.relu(x) # (2, 2, 256)
        x = self.pooling(x) # (2, 2, 256)
        x = self.dropout25(x) # (1, 1, 256)

        # (높이,너비,채널)3차원 텐서이므로 완전연결층(Fully-Connected)과 연결을 위해 view() 명령어를 이용해서 3차원 텐서를 1차원 vector로 만들어 주는 역할
        # view(-1, 벡터크기) 이용해서 배치차원은 유지되고 3차원 텐서는 1차원 벡터로 변환됨
        x = x.view(-1, 1 * 1 * 256)

        # Linear
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout50(x)
        
        x = self.fc2(x)

        return x

# 라벨 맵 정의 
labels_map = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

# 경로 구성
base_dir = os.path.dirname(__file__)  # 현재 파일 기준 디렉토리
model_path = os.path.join(base_dir, '..', 'models', 'model_deep_cnn_cifar10.ckpt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(DEVICE)
# model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)

# 모델 로드 : 중단된 지점부터 이어서 학습
# checkpoint = torch.load(model_path, map_location=DEVICE)# GPU/CPU 환경에 따라 로딩 시 map_location=DEVICE 옵션을 추가하여 안전하게 한다.
# model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    # transforms.Grayscale(), # RGB 이미지를 흑백으로 변환하는 과정(Fashion MNIST Dataset 흑백)
    transforms.Resize( (32, 32) ), # 사이즈 조정
    transforms.ToTensor(), # 이미지 0~255 값을 가지는데, 0~1사이 값으로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화 추가
])

# Streamlit UI 생성
st.title('CIFAR-10 이미지 분류기\n(with Deep CNN Model)')
uploaded_file = st.file_uploader('이미지를 업로드하세요 (32x32 RGB)', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    # st.image(image, caption='업로드된 이미지', use_column_width=True)
    st.image(image, caption='업로드된 이미지', width='stretch')

    # 전처리 및 추론
    img_tensor = transform(image).unsqueeze(0).to(DEVICE) # 배치 차원 추가
    # pil_image = TF.to_pil_image(img_tensor.squeeze())
    img_for_display = img_tensor.squeeze(0).cpu() * 0.5 + 0.5
    img_for_display = TF.to_pil_image(img_for_display) # 배치 차원만 제거
    st.image(img_for_display, caption='전처리된 이미지', width='stretch')
    with torch.no_grad():
        output = model(img_tensor).to(DEVICE) # 모델 추론
        _, pred = torch.max(output, dim=1) # 모델 추론값 추출
        label = labels_map[pred.item()]
    
    st.success(f'예측 결과: **{label}**')

    # 예측 확률 시각화
    # probs = torch.nn.functional.softmax(output, dim=1)
    probs = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
    # st.write(probs)
    st.subheader("클래스별 예측 확률")   
    
    # 예측 확률 막대그래프
    fig, ax = plt.subplots()
    ax.bar(labels_map.values(), probs)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # for i, p in enumerate(probs[0]):
    for i, p in enumerate(probs):
        st.write(f"{labels_map[i]}: {p.item():.2%}")
