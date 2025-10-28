# app_20_deep_learning_model_hybrid_emnist.py
# Streamlit 앱 - Hybrid 모델 EMNIST 손글씨 숫자 + 알파벳 이미지 분류
# '0~9' 숫자 + 알파벳 이미지 분류를 위한 CNN + RNN 하이브리드 모델 사용

# app_emnist_handwriting_recognizer.py

import streamlit as st
import torch
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import json, os
from model_utils import HybridModel_CNN_RNN
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows용 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로딩
@st.cache_resource
def load_model():
    model = HybridModel_CNN_RNN(in_channels=1, rnn_hidden=128, num_classes=47).to(DEVICE)
    model.load_state_dict(torch.load("./models/model_hybrid_emnist.pt", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# 라벨 맵 로딩
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 파일의 디렉토리 경로
labels_path = os.path.join(BASE_DIR, 'labels_map.json') # 예: 클래스 인덱스 → 마스크 작용/미작용 이름
with open(labels_path, "r") as f:
    labels_map = {int(k): v for k, v in json.load(f).items()}

# Streamlit UI
st.title("✍️ 손글씨 문자 인식기 (EMNIST 기반)")
st.write("아래 캔버스에 숫자나 알파벳을 직접 그려보세요. 모델이 실시간으로 인식합니다.")

# 캔버스 설정
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

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

# 이미지 전처리 함수
def preprocess_image(image_data):
    image = Image.fromarray((image_data[:, :, :3]).astype('uint8')).convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = center_image(image)     # 중심 정렬
    image = ImageEnhance.Contrast(image).enhance(2.0)  # 대비 강화

    # 중심 정렬
    img_np = np.array(image)
    coords = np.column_stack(np.where(img_np > 0))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = img_np[y_min:y_max+1, x_min:x_max+1]
    canvas = np.zeros((28, 28), dtype=np.uint8)
    h, w = cropped.shape
    y_offset = (28 - h) // 2
    x_offset = (28 - w) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    return Image.fromarray(canvas)

# 예측 수행
if canvas_result.image_data is not None:
    image = preprocess_image(canvas_result.image_data)
    if image is None:
        st.warning("이미지가 비어 있거나 너무 흐립니다. 다시 그려주세요.")
    else:
        st.image(image, caption="입력 이미지", width=150)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
            pred = int(np.argmax(probabilities))
            pred_label = labels_map.get(pred, "Unknown")

        st.success(f"🧠 모델 예측 결과: `{pred_label}` (클래스 인덱스: {pred})")

        # 🔢 Top-3 예측 결과 표시
        topk = torch.topk(torch.tensor(probabilities), k=3)
        top_indices = topk.indices.numpy()
        top_probs = topk.values.numpy()

        st.markdown("### 🔍 Top-3 예측 결과")
        for i in range(3):
            label = labels_map.get(int(top_indices[i]), "Unknown")
            prob = top_probs[i]
            st.write(f"Top-{i+1}: `{label}` ({prob:.2%})")

        # 📋 예측 결과 복사/다운로드
        st.markdown("### 📥 예측 결과 복사 및 저장")
        st.text_input("예측된 문자", value=pred_label, disabled=True)
        st.download_button("결과 저장하기", data=pred_label, file_name="prediction.txt", mime="text/plain")

        # 확률 시각화
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=[labels_map[i] for i in range(len(probabilities))], y=probabilities, ax=ax, palette="Blues_d")
        ax.set_title("클래스별 예측 확률")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Class")
        plt.xticks(rotation=90)
        st.pyplot(fig)

uploaded_file = st.file_uploader("손글씨 이미지 업로드 (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = preprocess_image(Image.open(uploaded_file))
    st.image(image, caption="업로드 이미지", width=150)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        pred_label = labels_map.get(pred, "Unknown")

    st.success(f"🧠 모델 예측 결과: `{pred_label}` (클래스 인덱스: {pred})")
    st.code(pred_label, language="text")  # 복사 가능한 텍스트 출력
    st.download_button("결과 복사/저장", pred_label, file_name="prediction.txt")