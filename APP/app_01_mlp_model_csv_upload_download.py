# MLP 이진 뷴류기 데모
# 사용법 : ./APP> streamlit run app_01_mlp_model_csv_upload_download.py
# 사용자가 숫자를 입력하면 예측 결과가 바로 표시되고, 입력값에 해당하는 위치에 빨간 선이 그려진 예측 곡선이 함께 나타남
# CSV 업로드를 통한 배치 예측 및 시각화
# 예측 결과를 CSV 파일로 다운로드할 수 있도록 기능, 이렇게 하면 사용자가 업로드한 데이터에 대한 예측 결과를 저장하고 활용

import streamlit as st
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
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
st.write("숫자를 입력하거나 CSV 파일을 업로드하면 분류 결과를 보여드립니다.")


# 단일 입력 예측, 사용자 입력
st.subheader("숫자 입력 예측")
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


# CSV 업로드 예측
st.subheader("CSV 업로드로 배치 예측")
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요 (컬럼명: input)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'input' not in df.columns:
        st.error("❌ CSV 파일에 'input'이라는 컬럼이 있어야 합니다.")
    else:
        inputs = torch.tensor(df['input'].values, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            preds = model(inputs)
            logicals = (preds > 0.5).float()

        df['probability'] = preds.numpy()
        df['prediction'] = logicals.numpy().astype(int)

        st.write("📊 예측 결과:")
        st.dataframe(df)

        # 시각화
        fig2, ax2 = plt.subplots()
        ax2.plot(df['input'], df['probability'], 'bo', label='Prediction')
        ax2.set_xlabel("Input")
        ax2.set_ylabel("Prediction")
        ax2.set_title("Batch Prediction Result")
        ax2.grid(True)
        st.pyplot(fig2)


# CSV 다운로드 버튼 추가
st.subheader("예측 결과 다운로드")

if uploaded_file is not None and 'input' in df.columns:
    # 결과를 CSV 형식으로 변환
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')

    # 다운로드 버튼
    st.download_button(
        label="예측 결과 CSV 다운로드",
        data=csv_bytes,
        file_name="prediction_result.csv",
        mime="text/csv"
    )
