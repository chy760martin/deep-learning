# MLP 이진 뷴류기 데모
# 사용법 : ./APP> streamlit run app_mlp_model.py
# 사용자가 숫자를 입력하면 예측 결과가 바로 표시되고, 입력값에 해당하는 위치에 빨간 선이 그려진 예측 곡선이 함께 나타남
import streamlit as st
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os

# 모델 클래스 정의
class MLPModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.net(x)
        return x

# 경로 구성
base_dir = os.path.dirname(__file__)  # 현재 파일 기준 디렉토리
model_path = os.path.join(base_dir, '..', 'models', 'mlp_model.ckpt')

# 모델 로딩
model = MLPModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Streamlit UI
st.title('MLP 이진 분류기 데모')
st.write('숫자를 입력하면 0 또는 1로 분류해드립니다.')

# 사용자 입력
user_input = st.number_input('숫자 입력', value=10.0)

# 예측
input_tensor = torch.tensor( [user_input], dtype=torch.float32 )
with torch.no_grad(): # 미분 연산하지 않음
    pred = model(input_tensor)
    label = 'Class 1' if pred.item() > 0.5 else 'Class 0'

# 결과 출력
st.write(f'예측 확률: `{pred.item():.4f}`')
st.write(f'분류 결과: **{label}**')

# 실시간 예측 그래프
st.subheader('입력값에 따른 예측 확률')

x_vals = torch.linspace(0, 25, steps=100).view(-1, 1) # torch.linspace 함수는 지정된 구간을 균등하게 나눈 값들을 가지는 텐서를 생성하는 PyTorch 함수
with torch.no_grad():
    y_vals = model(x_vals).squeeze().numpy() # numpy.squeeze() 함수는 NumPy에서 사용되는 함수로, 배열(array)에서 크기가 1인 차원을 제거하는데 사용

fig, ax = plt.subplots()
ax.plot(x_vals.numpy(), y_vals, label='Prediction', color='blue')
ax.axvline(user_input, color='red', linestyle='--', label='Input')
ax.set_xlabel('Input')
ax.set_ylabel('Prediction')
ax.set_title('MLP Model Prediction Graph')
ax.grid()
ax.legend()

st.pyplot(fig)