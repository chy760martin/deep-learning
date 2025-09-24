# MNIST 숫자 분류기 웹앱 데모
# 사용법 : ./APP> streamlit run app_02_mlp_model.py
# 1. 사용자 입력 방식
# - 테스트셋에서 무작위 이미지 선택
# - 사용자가 직접 이미지 업로드
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
from PIL import Image, ImageOps, ImageEnhance
from streamlit_drawable_canvas import st_canvas
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

# GPU 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 경로 구성
base_dir = os.path.dirname(__file__)  # 현재 파일 기준 디렉토리
model_path = os.path.join(base_dir, '..', 'models', 'model_mlp_mnist.ckpt')

# 모델 로딩
model = MLPDeepLearningModel().to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()

# MNIST 테스트 데이터셋 로딩
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='../data/MNIST_data', train=False, transform=transform, download=True)

# Streamlit UI
st.title('MNIST 숫자 분류기 웹앱')
st.write("테스트셋에서 무작위 이미지를 선택하거나, 직접 손글씨 이미지를 업로드해보세요!")

# 탭 구성
# tab1, tab2, tab3 = st.tabs(['테스트셋 예측', '이미지 업로드', '직접 그리기'])
mode = st.selectbox("모드 선택", ["테스트셋 예측", "이미지 업로드", "직접 그리기"])


# 테스트셋 예측 탭
if mode == "테스트셋 예측":
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

# 이미지 업로드 탭
elif mode == "이미지 업로드":
    uploaded_file = st.file_uploader('손글씨 숫자 이미지 업로드 (PNG/JPG)', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L') # 흑백 변환
        image = ImageOps.invert(image) # MNIST 스타일 반전
        image = image.resize( (28, 28) ) # MNIST 크기로 조정

        st.image(image, caption='업로드 이미지', width=150)

        transform = transforms.Compose([
            transforms.ToTensor(), # 0~255 이미지 값을 0~1 사이값을 변환
            transforms.Normalize((0.1307,), (0.3081,)) # 정규화
        ])
        input_tensor = transform(image).view(-1, 28 * 28).to(DEVICE)

        with torch.no_grad(): # 미분 연산하지 않음
            output = model(input_tensor) # 모델 예측
            pred = torch.argmax(output, dim=1).item() # 모델 예측값 추출
        
        st.success(f'모델 예측 결과: {pred}')

# 직접 그리기 탭
elif mode == "직접 그리기":
    st.write("🖌️ 아래 캔버스에 숫자를 직접 그려보세요")

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_test",
        update_streamlit=True
    )

    if canvas_result.image_data is not None:
        image_data = canvas_result.image_data

        def center_image(img):
            img = np.array(img)
            coords = np.column_stack(np.where(img > 0))
            if coords.size == 0:
                return Image.fromarray(img)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            cropped = img[y_min:y_max+1, x_min:x_max+1]
            canvas = np.zeros((28, 28), dtype=np.uint8)
            h, w = cropped.shape
            y_offset = (28 - h) // 2
            x_offset = (28 - w) // 2
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
            return Image.fromarray(canvas)

        if np.std(image_data) < 1:
            st.warning("이미지가 너무 희미하거나 비어 있습니다. 숫자를 굵게 그려주세요.")
        else:
            image = Image.fromarray((image_data[:, :, :3]).astype('uint8')).convert('L')
            image = ImageOps.invert(image)
            image = image.resize((28, 28))
            image = center_image(image)

            st.image(image, caption='그린 숫자', width=150)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            input_tensor = transform(image).view(-1, 28 * 28)

            # 모델 추론 (모델은 미리 로딩되어 있어야 함)
            with torch.no_grad():
                # output은 모델의 최종 출력 텐서, 보통 shape은 [1, 10](숫자 0~9에 대한 로짓 값)
                output = model(input_tensor)
                # pred = torch.argmax(output, dim=1).item()
                # softmax()를 통해 각 클래스에 대한 확률로 변환, squeeze()는 배치 차원을 제거해 1D 배열로 만든다, numpy()는 넘파이 데이터로 만든다.
                # .squeeze()는 불필요한 차원을 제거 [1, 10] -> [10]으로 바꿔서 1D 배열로 만든다.
                probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().numpy()
                pred = int(np.argmax(probabilities))

            st.success(f'모델 예측 결과: {pred}')
            st.markdown(f"### ✏️ 모델이 인식한 숫자: `{pred}`")

            # 예측 확률 막대그래프
            fig, ax = plt.subplots()
            ax.bar(range(10), probabilities, color='skyblue')
            ax.set_xticks(range(10))
            ax.set_xlabel('Num Class')
            ax.set_ylabel('Prediction')
            ax.set_title('Model Prediction')

            st.pyplot(fig)