# MNIST 숫자 분류기 웹앱 데모
# 사용법 : ./APP> streamlit run app_02_mlp_model.py
# 1. 사용자 입력 방식
# - 테스트셋에서 무작위 이미지 선택
# - 사용자가 직접 이미지 업로드 (추후 확장 가능)
# 2. 모델 추론
# - 학습된 MLP 모델 로딩 (torch.load)
# - 이미지 전처리 후 예측 수행
# 3. 결과 시각화
# - 예측 결과 출력 (정답 vs 예측)
# - Confusion Matrix 및 오차 분석
# - 틀린 예측 샘플 시각화

import streamlit as st
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 모델 클래스 정의
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

# 경로 구성
base_dir = os.path.dirname(__file__)  # 현재 파일 기준 디렉토리
model_path = os.path.join(base_dir, '..', 'models', 'model_mlp_mnist.ckpt')

# 모델 로딩
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPDeepLearningModel().to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()

# MNIST 데이터셋 로딩
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='../data/MNIST_data', train=False, transform=transform, download=True)

# Streamlit UI
st.title('MNIST 숫자 분류기 웹앱')
st.write('테스트셋에서 무작위로 이미지를 선택해 예측 결과를 확인하세요.')

# 무작위 이미지 선택
sample_idx = st.slider('이미지 인덱스 선택', 0, len(test_dataset)-1, 0)
image, label = test_dataset[sample_idx]

# 모델 추론
with torch.no_grad(): # 미분 연산하지 않음
    input_tensor = image.view(-1, 28 * 28).to(DEVICE)
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()
    
# 시각화
st.image(image.squeeze().numpy(), caption=f'실제 라벨: {label}, 예측: {pred}', width=150, channels='GRAY')

# Confusion Matrix 버튼
if st.button('전체 Confusion Matrix 보기'):
    all_preds = []
    all_labels = []
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for imgs, labels in test_loader:
            x = imgs.view(-1, 28 * 28).to(DEVICE) # 학습데이터
            y = labels.to(DEVICE) # 정답데이터

            outputs = model(x) # 모델 예측
            _, preds = torch.max(outputs, 1) # 모델 예측 값 추출

            all_preds.extend(preds.cpu().numpy()) # GPU 텐서를 CPU로 옮기고 넘파이 배열로 변환
            all_labels.extend(y.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.text('Classification Report:')
    st.text(classification_report(all_labels, all_preds, digits=4))