# app_19_deep_learning_model_hybrid_mnist.py
# Streamlit 앱 - Hybrid 모델 MNIST 손글씨 숫자 이미지 분류
# '0~9' 숫자 이미지 분류를 위한 CNN + RNN 하이브리드 모델 사용

import streamlit as st
import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from model_utils import HybridModel_CNN_RNN
import os, json, cv2 # 얼굴 검출용
from streamlit_drawable_canvas import st_canvas
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import pandas as pd

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로딩
@st.cache_resource # - @st.cache_resource: 모델을 한 번만 로딩하고 캐시하여 앱 속도 향상
def load_model():
    num_classes = 10 # MNIST는 10개 숫자 클래스 (0~9)
    model = HybridModel_CNN_RNN(in_channels=1, rnn_hidden=128, num_classes=num_classes, dropout_rate=0.3).to(DEVICE) # 모델 객체 생성

    model_path = os.path.join("models", "model_hybrid_mnist.pt") # - 학습된 모델 가중치 로드
    if not os.path.exists(model_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        st.stop()

    model.load_state_dict(torch.load(model_path, map_location=DEVICE)) # - 학습된 모델 가중치 로드
    model.eval() # - 추론 모드로 전환
    return model

model = load_model() # 모델 로드

# MNIST 테스트 데이터셋 로딩
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='../data/MNIST_data', train=False, transform=transform, download=True)

# 라벨 맵 로딩 - labels_map.json 파일에서 클래스 인덱스와 강아지 이름 매핑
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 파일의 디렉토리 경로
labels_path = os.path.join(BASE_DIR, 'labels_map.json') # 예: 클래스 인덱스 → 마스크 작용/미작용 이름

# if not os.path.exists(labels_path): # labels_map.json 파일이 없으면 에러 메시지 출력
#     st.error(f"labels_map.json 파일을 찾을 수 없습니다: {labels_path}")
#     st.stop() # - 파일이 없으면 앱 중지

# with open(labels_path, 'r') as f: # - labels_map.json 파일 열기
#     labels_map = {int(k):v for k, v in json.load(f).items()} # - JSON 파일에서 딕셔너리로 로드, 키를 int로 변환

# Streamlit UI
st.title('CNN MNIST 숫자 분류기 웹앱')
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
        # input_tensor = image.view(-1, 28 * 28).to(DEVICE) # MLP 모델 형태
        # input_tensor = image.view(-1, 1, 28, 28).to(DEVICE) # CNN 모델 형태
        input_tensor = image.unsqueeze(0).to(DEVICE) # - image.unsqueeze(0)는 차원 생성
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
                x = imgs.to(DEVICE) # 이미 [batch, 1, 28, 28] 형태를 갖추고 있음
                y = labels.to(DEVICE) # 정답데이터

                outputs = model(x) # 모델 예측
                _, preds = torch.max(outputs, 1) # 모델 예측 값 추출

                all_preds.extend(preds.cpu().numpy()) # GPU 텐서를 CPU로 옮기고 넘파이 배열로 변환
                all_labels.extend(y.cpu().numpy())
        
        # confusion matrix 계산 (normalize='true'로 비율 기반)
        cm = confusion_matrix(all_labels, all_preds, normalize='true')

        # 시각화
        st.text('Confusion Matrix:')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Normalized Confusion Matrix')
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
        # input_tensor = transform(image).view(-1, 28 * 28).to(DEVICE)
        # input_tensor = transform(image).view(-1, 1, 28, 28).to(DEVICE) # CNN 모델 형태
        input_tensor = transform(image).unsqueeze(0).to(DEVICE) # [1, 1, 28, 28] image.unsqueeze(0)는 차원 생성

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
            # input_tensor = transform(image).view(-1, 28 * 28)
            # input_tensor = transform(image).view(-1, 1, 28, 28).to(DEVICE) # CNN 모델 형태
            input_tensor = transform(image).unsqueeze(0).to(DEVICE) # [1, 1, 28, 28]

            # 모델 추론 (모델은 미리 로딩되어 있어야 함)
            with torch.no_grad():
                # output은 모델의 최종 출력 텐서, 보통 shape은 [1, 10](숫자 0~9에 대한 로짓 값)
                output = model(input_tensor)
                # pred = torch.argmax(output, dim=1).item()
                # softmax()를 통해 각 클래스에 대한 확률로 변환, squeeze()는 배치 차원을 제거해 1D 배열로 만든다, numpy()는 넘파이 데이터로 만든다.
                # .squeeze()는 불필요한 차원을 제거 [1, 10] -> [10]으로 바꿔서 1D 배열로 만든다.
                probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
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

st.markdown("---")
st.caption("Powered by CNN + RNN Hybrid Learning")