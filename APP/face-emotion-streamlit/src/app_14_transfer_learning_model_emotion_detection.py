# app_14_transfer_learning_model_emotion_detection.py
# Streamlit 앱 - Kaggle Face Emotion Detection Classification
# {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}

import streamlit as st
import torch
from torchvision import models, transforms
import cv2 # 얼굴 검출용
from PIL import Image # 이미지 포멧용
import numpy as np
from model_utils import TransferLearningModel  # 현용님이 만든 클래스
import os, json
import pandas as pd

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로딩
@st.cache_resource # - @st.cache_resource: 모델을 한 번만 로딩하고 캐시하여 앱 속도 향상
def load_model():
    # base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT) # - vit_b_16: Vision Transformer 사전학습 모델
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # ResNet 모델을 사전학습 가중치로 불러옴.
    num_classes=7
    model = TransferLearningModel(base_model, feature_extractor=True, num_classes=num_classes).to(device) # - feature_extractor=True: 특징 추출기로 사용, num_classes=2: 마스크 착용 여부 2 클래스

    model_path = os.path.join("models", "model_transfer_learning_emotion_detection.pt") # - 학습된 모델 가중치 로드
    if not os.path.exists(model_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        st.stop()

    model.load_state_dict(torch.load(model_path, map_location=device)) # - 학습된 모델 가중치 로드
    model.eval() # - 추론 모드로 전환
    return model

model = load_model()
# 이미지 전처리 함수 - OpenCV를 이용해 이미지에서 얼굴을 자동으로 검출하고, 그 얼굴 영역만 잘라서 전처리하는 함수
# 1) 이미지에서 얼굴을 찾는다 (OpenCV Haar Cascade), 2) 얼굴이 있으면 crop해서 전처리, 3) 얼굴이 없으면 전체 이미지를 전처리
def detect_and_preprocess(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # OpenCV에서 제공하는 Haar Cascade 얼굴 검출기를 불러옴.
    img_cv = np.array(image) # 넘파이 변환하여 OpenCV가 처리할 수 있도록 함.
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY) # 얼굴 검출은 흑백 이미지에서 더 빠르고 정확하게 작동하므로, RGB 이미지를 그레이스케일로 변환
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) # 얼굴 검출, 1.1(이미지 크기를 10%씩 줄여가며 탐색), 4(최소 4개의 이웃 사각형이 있어야 얼굴로 판단)

    if len(faces) == 0:
        return preprocess_image(image) # 얼굴이 하나도 검출되지 않으면, 전체 이미지를 그대로 전처리해서 반환(fallback: 전체 이미지 사용)

    x, y, w, h = faces[0] # 첫 번째로 검출된 얼굴의 좌표를 가져옵니다. (여러 얼굴이 있을 경우 첫 번째만 사용)
    face_img = img_cv[y:y+h, x:x+w] # 얼굴 영역만 잘라냅니다 (crop)
    face_pil = Image.fromarray(face_img) # NumPy 배열을 다시 PIL 이미지로 변환, preprocess_image()는 PIL 이미지를 입력으로 받기 때문임.
    return preprocess_image(face_pil)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 이미지 크기를 224×224로 조정 (ViT 입력 크기)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device) # - unsqueeze(0): 배치 차원 추가 → [1, 3, 224, 224], 배치 차원 추가 및 디바이스로 이동

# Grad-CAM 함수 - 모델, 입력 이미지 텐서, 예측 클래스 인덱스를 받아 heatmap을 생성
# 빨간색 영역 - 모델이 가장 주목한 부분입니다. 이 영역이 예측에 가장 큰 영향을 미쳤으며, 모델이 이 부분을 통해 클래스를 결정했다고 볼 수 있습니다.
# 노란색과 녹색 영역 - 이 부분도 예측에 기여했지만, 빨간색 영역보다는 덜 중요하게 작용했습니다.
# 파란색 영역 - 모델이 상대적으로 덜 주목하거나 거의 주목하지 않은 부분입니다.
# ResNet 모델, 전처리된 입력 이미지(shape:[1,3,224,224]), Grad-CAM을 적용할 대상 레이어(보통 마지막 convolution block), 예측된 클래스 인덱스(0~7)
def grad_cam(model, image_tensor, target_layer, predicted_class):
    gradients = []
    activations = []
    
    def forward_hook(module, input, output): # 순전파 시 활성화값(feature map)을 저장
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output): # 역전파 시 gradient를 저장
        gradients.append(grad_output[0])
    
    # Hook 등록, 지정한 target_layer에 hook을 연결하여 forward/backward 시점에 데이터를 수집
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # 순전파 및 역전파
    model.zero_grad() # 미분 초기화
    output = model(image_tensor) # 모델 예측

    # output[0] 배치에서 첫번째 이미지 예측 결과, predicted_class: 예측된 클래스 인덱스 (예: 3 → happy)
    # 따라서 output[0, predicted_class]는: 
    # 모델이 해당 이미지에 대해 예측한 클래스의 점수만 하나 선택하는 것입니다.
    # 예시: output[0, 3] = 3.1 → 모델이 "happy" 클래스에 대해 3.1이라는 점수를 줬다는 뜻
    class_score = output[0, predicted_class]
    class_score.backward() # 미분 연산, 예측된 클래스 하나만 선택하여 역전파 한다.

    # Grad-CAM 계산
    grads = gradients[0].cpu().detach().numpy()[0] # 텐서 -> 넘파이 변경
    acts = activations[0].cpu().detach().numpy()[0]
    weights = np.mean(grads, axis=(1, 2)) # 각 채널의 중요도 (gradient의 평균값)
    # 최종적으로 각 위치의 중요도를 누적할 공간
    cam = np.zeros(acts.shape[1:], dtype=np.float32) # acts.shape: [C, H, W] → 채널 수, 높이, 너비, acts.shape[1:]: [H, W] → 각 채널의 공간 크기

    for i, w in enumerate(weights): # 각 채널의 feature map에 해당 채널의 중요도(weight)를 곱해서 누적
        cam += w * acts[i]
    
    cam = np.maximum(cam, 0) # 음수 제거(음수는 중요하지 않다고 간주한다.)
    cam = cv2.resize(cam, (224, 224)) # heatmap을 원본 이미지 크기와 맞춤
    # 0~1 범위로 정규화
    cam = cam - np.min(cam) # 가장 낮은 값 제거
    cam = cam / np.max(cam) # 가장 높은 값으로 스케일링

    return cam


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
labels_path = os.path.join(BASE_DIR, 'labels_map.json') # 예: 클래스 인덱스 → 마스크 작용/미작용 이름

if not os.path.exists(labels_path): # labels_map.json 파일이 없으면 에러 메시지 출력
    st.error(f"labels_map.json 파일을 찾을 수 없습니다: {labels_path}")
    st.stop() # - 파일이 없으면 앱 중지

with open(labels_path, 'r') as f: # - labels_map.json 파일 열기
    labels_map = {int(k):v for k, v in json.load(f).items()} # - JSON 파일에서 딕셔너리로 로드, 키를 int로 변환

# Streamlit UI
st.title("얼굴 감정 분류기") # - 앱 제목
st.write("이미지를 업로드하면 얼굴 감정를 예측해줍니다!") # - 앱 설명

# 단일 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png" ]) # - 파일 업로더
if uploaded_file is not None: # - 파일이 업로드되었을 때
    image = Image.open(uploaded_file).convert("RGB") # - 이미지를 RGB로 변환
    # st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.image(image, caption="업로드된 이미지", width='stretch') # - use_container_width=True: 컨테이너 너비에 맞게 이미지 크기 조정

    try:
        # image_tensor = preprocess_image(image) # - 이미지 전처리
        image_tensor = detect_and_preprocess(image) # - 이미지 전처리
        prediction, probabilities = predict(image_tensor) # - 예측 수행
        label = labels_map[prediction]

        st.success(f'예측 결과: **{label}**')
        st.subheader("예측 확률")
        st.bar_chart( {labels_map[i]: prob for i, prob in enumerate(probabilities)} )

        # Grad-CAM 시각화 추가
        target_layer = model.model.layer4[-1] # TransferLearningModel 클래스 내부의 ResNet18 마지막 Residual Block 지정
        # 모델과 에측된 클래스에 대해 Grad-CAM heatmap을 생성
        cam = grad_cam(model=model.model, image_tensor=image_tensor, target_layer=target_layer, predicted_class=prediction)

        # 원본 이미지와 heatmap overlay
        img_np = np.array(image.resize( (224, 224) )) / 255.0
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET) # COLORMAP_JET은 빨강(높은 중요도), 파랑(낮은 중요도)으로 표현됨
        heatmap = np.float32(heatmap) / 255
        overlay = heatmap + img_np
        overlay = overlay / np.max(overlay)
        
        st.subheader('Grad-CAM 시각화')
        st.image(img_np, caption='원본 이미지', use_container_width=True)
        st.image(overlay, caption='Grad-CAM Overlay', use_container_width=True)

    except Exception as e:
        st.error(f"예측 처리 중 오류가 발생했습니다: {e}")

# 웹캠 입력
camera_image = st.camera_input("📷 웹캠으로 사진 촬영")

if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="촬영된 이미지", width='stretch')

    try:
        # image_tensor = preprocess_image(image)
        image_tensor = detect_and_preprocess(image) # - 이미지 전처리
        prediction, probabilities = predict(image_tensor)
        label = labels_map[prediction]

        st.success(f'예측 결과: **{label}**')

        st.subheader("예측 확률")
        st.bar_chart({labels_map[i]: prob for i, prob in enumerate(probabilities)})

        # Grad-CAM 시각화 추가
        target_layer = model.model.layer4[-1] # ResNet18의 마지막 Residual Block
        cam = grad_cam(model=model.model, image_tensor=image_tensor, target_layer=target_layer, predicted_class=prediction)

        # 원본 이미지와 heatmap overlay
        img_np = np.array(image.resize( (224, 224) )) / 255.0
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        overlay = heatmap + img_np
        overlay = overlay / np.max(overlay)

        st.subheader('Grad-CAM 시각화')
        st.image(img_np, caption='원본 이미지', use_container_width=True)
        st.image(overlay, caption='Grad-CAM Overlay', use_container_width=True)

    except Exception as e:
        st.error(f"웹캠 예측 처리 중 오류가 발생했습니다: {e}")

# 다중 이미지 업로드 (옵션)
uploaded_files = st.file_uploader("이미지를 여러 장 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []  # 예측 결과 저장 리스트
    st.write(f"{len(uploaded_files)}개의 이미지가 업로드되었습니다.")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, width='stretch')

        try:
            # image_tensor = preprocess_image(image)
            image_tensor = detect_and_preprocess(image) # - 이미지 전처리
            prediction, probabilities = predict(image_tensor)
            label = labels_map[prediction]
            
            st.success(f'예측 결과: **{label}**')
            # 결과 저장
            results.append({
                "파일명": uploaded_file.name,
                "예측 클래스": label,
                "확률": f"{probabilities[prediction]:.4f}"
            })
            st.bar_chart({labels_map[i]: prob for i, prob in enumerate(probabilities)})

            # Grad-CAM 시각화 추가
            target_layer = model.model.layer4[-1] # ResNet18의 마지막 Residual Block
            cam = grad_cam(model=model.model, image_tensor=image_tensor, target_layer=target_layer, predicted_class=prediction)

            # 원본 이미지와 heatmap overlay
            img_np = np.array(image.resize( (224, 224) )) / 255.0
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            overlay = heatmap + img_np
            overlay = overlay / np.max(overlay)

            st.subheader('Grad-CAM 시각화')
            st.image(img_np, caption='원본 이미지', use_container_width=True)
            st.image(overlay, caption='Grad-CAM Overlay', use_container_width=True)

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
st.caption("Powered by ResNet + Transfer Learning")