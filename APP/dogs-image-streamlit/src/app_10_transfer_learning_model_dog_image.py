# app_10_transfer_learning_model_dog_image.py
# Streamlit 앱 - 강아지 종 분류기

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from model_utils import TransferLearningModel  # 현용님이 만든 클래스
import os, json
import pandas as pd

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로딩
@st.cache_resource # - @st.cache_resource: 모델을 한 번만 로딩하고 캐시하여 앱 속도 향상
def load_model():
    base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT) # - vit_b_16: Vision Transformer 사전학습 모델
    model = TransferLearningModel(base_model, feature_extractor=True, num_classes=4).to(device) # - feature_extractor=True: 특징 추출기로 사용, num_classes=2: 강아지 4 클래스

    model_path = os.path.join("models", "model_transfer_learning_dog_image.ckpt") # - 학습된 모델 가중치 로드
    if not os.path.exists(model_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        st.stop()

    model.load_state_dict(torch.load(model_path, map_location=device)) # - 학습된 모델 가중치 로드
    model.eval() # - 추론 모드로 전환
    return model

model = load_model()

# 이미지 전처리 함수
def detect_and_preprocess(image):
    import cv2
    from PIL import Image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return preprocess_image(image)  # fallback: 전체 이미지 사용

    x, y, w, h = faces[0]
    face_img = img_cv[y:y+h, x:x+w]
    face_pil = Image.fromarray(face_img)
    return preprocess_image(face_pil)

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
        probabilities = torch.nn.functional.softmax(outputs, dim=1) # - 소프트맥스 함수로 확률 계산
        predicted = torch.argmax(probabilities, dim=1).item() # - 가장 높은 점수를 가진 클래스 선택
        probabilities = probabilities.cpu().numpy()[0] # - 확률을 CPU로 옮기고 numpy 배열로 변환
    return predicted, probabilities # - 예측된 클래스 인덱스 반환 (0~3), 각 클래스에 대한 확률 배열 반환

# 라벨 맵 로딩 - labels_map.json 파일에서 클래스 인덱스와 강아지 이름 매핑
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 파일의 디렉토리 경로
labels_path = os.path.join(BASE_DIR, 'labels_map.json') # 예: 클래스 인덱스 → 강아지 이름

if not os.path.exists(labels_path): # labels_map.json 파일이 없으면 에러 메시지 출력
    st.error(f"labels_map.json 파일을 찾을 수 없습니다: {labels_path}")
    st.stop() # - 파일이 없으면 앱 중지

with open(labels_path, 'r') as f: # - labels_map.json 파일 열기
    labels_map = {int(k):v for k, v in json.load(f).items()} # - JSON 파일에서 딕셔너리로 로드, 키를 int로 변환

# Streamlit UI
st.title("강아지 종 분류기") # - 앱 제목
st.write("이미지를 업로드하면 AI가 무슨 강아지인지 예측해줍니다!") # - 앱 설명

# 단일 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png" ]) # - 파일 업로더
if uploaded_file is not None: # - 파일이 업로드되었을 때
    image = Image.open(uploaded_file).convert("RGB") # - 이미지를 RGB로 변환
    # st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.image(image, caption="업로드된 이미지", use_container_width=True) # - use_container_width=True: 컨테이너 너비에 맞게 이미지 크기 조정

    try:
        image_tensor = preprocess_image(image) # - 이미지 전처리
        # image_tensor = detect_and_preprocess(image) # - 이미지 전처리
        prediction, probabilities = predict(image_tensor) # - 예측 수행
        label = labels_map[prediction]

        emoji_map = {
            "Chihuahua": "🐕",
            "jindo_dog": "🦮",
            "shepherd": "🐶",
            "yorkshire_terrier": "🐩"
        }
        st.success(f'예측 결과: **{label}** {emoji_map.get(label, "")}')

        st.subheader("예측 확률")
        st.bar_chart( {labels_map[i]: prob for i, prob in enumerate(probabilities)} )
    except Exception as e:
        st.error(f"예측 처리 중 오류가 발생했습니다: {e}")

# 웹캠 입력
camera_image = st.camera_input("📷 웹캠으로 사진 촬영")

if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="촬영된 이미지", use_container_width=True)

    try:
        image_tensor = preprocess_image(image) # - 이미지 전처리
        # image_tensor = detect_and_preprocess(image) # - 이미지 전처리
        prediction, probabilities = predict(image_tensor)
        label = labels_map[prediction]

        emoji_map = {
            "Chihuahua": "🐕",
            "jindo_dog": "🦮",
            "shepherd": "🐶",
            "yorkshire_terrier": "🐩"
        }
        st.success(f'예측 결과: **{label}** {emoji_map.get(label, "")}')

        st.subheader("예측 확률")
        st.bar_chart({labels_map[i]: prob for i, prob in enumerate(probabilities)})
    except Exception as e:
        st.error(f"웹캠 예측 처리 중 오류가 발생했습니다: {e}")

# 다중 이미지 업로드 (옵션)
uploaded_files = st.file_uploader("이미지를 여러 장 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []  # 예측 결과 저장 리스트
    st.write(f"{len(uploaded_files)}개의 이미지가 업로드되었습니다.")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        try:
            image_tensor = preprocess_image(image) # - 이미지 전처리
            # image_tensor = detect_and_preprocess(image) # - 이미지 전처리
            prediction, probabilities = predict(image_tensor)
            label = labels_map[prediction]

            emoji_map = {
                "Chihuahua": "🐕",
                "jindo_dog": "🦮",
                "shepherd": "🐶",
                "yorkshire_terrier": "🐩"
            }
            st.success(f'예측 결과: **{label}** {emoji_map.get(label, "")}')
            # 결과 저장
            results.append({
                "파일명": uploaded_file.name,
                "예측 클래스": label,
                "확률": f"{probabilities[prediction]:.4f}"
            })
            st.bar_chart({labels_map[i]: prob for i, prob in enumerate(probabilities)})
        except Exception as e:
            st.error(f"예측 처리 중 오류가 발생했습니다: {e}")
    
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        st.subheader("📄 예측 결과 요약")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 결과 CSV 다운로드",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Powered by Vision Transformer (ViT) + Transfer Learning")