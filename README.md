<h2> My tech stack </h2>
<div align=center>
        <img src="https://img.shields.io/badge/springboot-6DB33F?style=for-the-badge&logo=springboot&logoColor=white">
        <img src="https://img.shields.io/badge/Spring-6DB33F?style=for-the-badge&logo=Spring&logoColor=white">
        <img src="https://img.shields.io/badge/java-007396?style=for-the-badge&logo=OpenJDK&logoColor=white">
        <img src="https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E">
        <img src="https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jQuery&logoColor=white"/>
        <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">
        <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
        <img src="https://img.shields.io/badge/ORACLE-F80000?style=for-the-badge&logo=oracle&logoColor=white"/>
        <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=MySQL&logoColor=white">
        <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black"/>
        <img src="https://img.shields.io/badge/Apache Tomcat-F8DC75?style=for-the-badge&logo=apachetomcat&logoColor=black"/>
        <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white">
        <img src="https://img.shields.io/badge/Anaconda-44A833?style=for-the-badge&logo=Anaconda&logoColor=white"/>
    <br>
</div>

<br>
<h2> Deep Learning </h2>

---
### 16. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : Kaggle Breast Ultrasound Detection - 유방암 예측 이미지 데이터셋 사용
> Model - 16_transfer_learning_kaggle_breast_ultrasound_detection.ipynb
> Streamlit 웹앱 기본 구조 
> - breast-detection-streamlit/src/app_16_transfer_learning_model_breast_ultrasound_detection.py
> - breast-detection-streamlit/src/model_utils.py
> - breast-detection-streamlit/src/labels_map.json
> - breast-detection-streamlit/models/model_transfer_learning_covid19_detection.pt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - 유방암 예측 이미지 분류 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 유방암 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 15. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : Kaggle Kaggle COVID19 Detection - COVID19 감염 예측 이미지 데이터셋 사용
> Model - 15_transfer_learning_kaggle_covid19_detection.ipynb
> Streamlit 웹앱 기본 구조 
> - covid19-detection-streamlit/src/app_15_transfer_learning_model_covid19_detection.py
> - covid19-detection-streamlit/src/model_utils.py
> - covid19-detection-streamlit/src/labels_map.json
> - covid19-detection-streamlit/models/model_transfer_learning_covid19_detection.pt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - COVID19 감염 예측 이미지 분류 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 COVID19 감염 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 14. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : Kaggle Face Emotion Detection Classification - Kaggle 얼굴에 감정 이미지 분류 데이터셋 사용
> Model - 14_transfer_learning_kaggle_emotion_detection.ipynb
> Streamlit 웹앱 기본 구조 
> - face-emotion-streamlit/src/app_14_transfer_learning_model_emotion_detection.py
> - face-emotion-streamlit/src/model_utils.py
> - face-emotion-streamlit/src/labels_map.json
> - face-emotion-streamlit/models/model_transfer_learning_emotion_detection.pt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - Kaggle 얼굴 감정 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 얼굴 감정 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 13. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : Kaggle Face Mask Detection Classification - Kaggle 얼굴에 마스크(face mask detection) 이미지 분류 데이터셋 사용
> Model - 13_transfer_learning_kaggle_face_mask_detection.ipynb
> Streamlit 웹앱 기본 구조 
> - face-mask-streamlit/src/app_13_transfer_learning_model_face_mask_detection.py
> - face-mask-streamlit/src/model_utils.py
> - face-mask-streamlit/src/labels_map.json
> - face-mask-streamlit/models/model_transfer_learning_face_mask_detection.ckpt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - Kaggle 얼굴에 마스크(face mask detection) 착용 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 얼굴에 마스크 착용 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 12. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : Kaggle brain tumor Image Classification (MRI) - Kaggle 뇌종양(Brain Tumor) 이미지 분류 데이터셋 사용
> Model - 12_transfer_learning_kaggle_brain_tumor_mri.ipynb
> Streamlit 웹앱 기본 구조 
> - brain-tumor-streamlit/src/app_12_transfer_learning_model_brain_tumor.py
> - brain-tumor-streamlit/src/model_utils.py
> - brain-tumor-streamlit/src/labels_map.json
> - brain-tumor-streamlit/models/model_transfer_learning_brain_tumor_mri.ckpt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - Kaggle 뇌종양(Brain Tumor) 이미지 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 뇌종양(brain tumor) 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 10. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : 강아지 감정 Dataset 사용
> Model - 11_transfer_learning_vit_dog_emotion_gpu.ipynb
> Streamlit 웹앱 기본 구조 
> - dogs-image-streamlit/src/app_11_transfer_learning_model_dog_emotion.py
> - dogs-image-streamlit/src/model_utils.py
> - dogs-image-streamlit/src/labels_map.json
> - dogs-image-streamlit/models/model_transfer_learning_dog_emotion.ckpt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - 강아지 감정 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 강아지 감정 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 9. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : 강아지 종 Dataset 사용
> Model - 10_transfer_learning_vit_custom_image_gpu.ipynb
> Streamlit 웹앱 기본 구조 
> - dogs-image-streamlit/src/app_10_transfer_learning_model_dog_image.py
> - dogs-image-streamlit/src/model_utils.py
> - dogs-image-streamlit/src/labels_map.json
> - dogs-image-streamlit/models/model_transfer_learning_dog_image.ckpt
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - 강아지 종 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 무슨 강아지인지 예측(단일 이미지 업로드, 웹캠 이미지, 멀티 이미지 업로드)
---
### 8. Transfer Learning(전이학습) - Pre-Trained Model(사전학습모델) : 고양이와 강아지 Dataset 사용
> Model - 09_transfer_learning_cats_dogs_gpu.ipynb
> Streamlit 웹앱 기본 구조 
> - cats-dogs-streamlit/src/app_08_transfer_learning_model_cats_dogs.py
> - cats-dogs-streamlit/src/model_utils.py
> - cats-dogs-streamlit/models/transfer_learning_model_cats_dogs.pth
1. Transfer Learning 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. Streamlit 앱 - 고양이와 강아지 분류기 데모 - streamlit 라이브러리를 사용하여 웹 앱 형태로 구현, 사용자가 이미지를 업로드하면 모델이 고양이인지 강아지인지 예측
---
### 7. Deep CNN(Convolution Neural Network) : CIFAR10 Dataset 사용
> Model - 07_deep_cnn_cifar10_gpu.ipynb
> Streamlit 웹앱 - app_07_deep_cnn_cifar10.py
1. Deep CNN 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. 웹에서 CIFAR10 숫자 분류기 웹앱 데모 - 사용자 입력 방식(사용자가 직접 이미지 업로드), 모델 추론(학습된 CNN 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력))
---
### 6. CNN(Convolution Neural Network) : CIFAR10 Dataset 사용
> Model - 06_cnn_cifar10_gpu.ipynb
> Streamlit 웹앱 - app_06_cnn_cifar10.py
1. CNN 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 학습률 개선, StepLR(일정 에폭마다 학습률을 감소), EarlyStopping(일정 에폭 동안 성능 향상이 없을 경우 학습을 조기 중단, 과적합 방지, 학습 시간 절약) 
6. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
7. 모델 저장 및 불러오기
8. 테스트 및 시각화
9. 웹에서 CIFAR10 숫자 분류기 웹앱 데모 - 사용자 입력 방식(사용자가 직접 이미지 업로드), 모델 추론(학습된 CNN 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력))
---
### 5. CNN(Convolution Neural Network) : Fashion MNIST Dataset 사용
> Model - 05_cnn_fashion_mnist_gpu.ipynb
> Streamlit 웹앱 - app_05_cnn_fashion_mnist.py
1. CNN 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 학습률 개선, StepLR(일정 에폭마다 학습률을 감소), EarlyStopping(일정 에폭 동안 성능 향상이 없을 경우 학습을 조기 중단, 과적합 방지, 학습 시간 절약) 
6. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
7. 모델 저장 및 불러오기
8. 테스트 및 시각화
9. 웹에서 MNIST 숫자 분류기 웹앱 데모 - 사용자 입력 방식(사용자가 직접 이미지 업로드), 모델 추론 학습된 CNN 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력 (정답 vs 예측))
---
### 4. CNN(Convolution Neural Network) : 필기체손글씨 MNIST Dataset 사용
> Model - 04_cnn_mnist_gpu.ipynb
> Streamlit 웹앱 - app_04_cnn_mnist.py
1. CNN 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 학습률 개선, StepLR(일정 에폭마다 학습률을 감소), EarlyStopping(일정 에폭 동안 성능 향상이 없을 경우 학습을 조기 중단, 과적합 방지, 학습 시간 절약) 
6. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
7. 모델 저장 및 불러오기
8. 테스트 및 시각화
9. 웹에서 MNIST 숫자 분류기 웹앱 데모 - 사용자 입력 방식(테스트셋에서 무작위 이미지 선택, 사용자가 직접 이미지 업로드, 사용자가 직접 그리기), 모델 추론(학습된 CNN 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력 (정답 vs 예측), Confusion Matrix 및 오차 분석, 틀린 예측 샘플 시각화))
---
### 3. MLP(Multi-Layer Perceptron) DeepLearning : Fashion MNIST Dataset 사용
> Model - 03_mlp_fashion_mnist_gpu.ipynb
> Streamlit 웹앱 - app_03_mlp_fashion_mnist.py
1. MLP 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가
5. 학습률 개선, StepLR(일정 에폭마다 학습률을 감소), EarlyStopping(일정 에폭 동안 성능 향상이 없을 경우 학습을 조기 중단, 과적합 방지, 학습 시간 절약) 
6. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
7. 모델 저장 및 불러오기
8. 테스트 및 시각화
9. 웹에서 Fashion MNIST 숫자 분류기 웹앱 데모 - 사용자 입력 방식(테스트셋에서 무작위 이미지 선택), 모델 추론(학습된 MLP 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력 (정답 vs 예측), Confusion Matrix 및 오차 분석, 틀린 예측 샘플 시각화))
---
### 2. MLP(Multi-Layer Perceptron) DeepLearning : 필기체손글씨 MNIST Dataset 사용
> Model - 02_mlp_mnist_gpu.ipynb
> Streamlit 웹앱 - app_02_mlp_mnist_model.py, app_02_mlp_mnist_model_image_upload.py
1. MLP 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 학습 및 평가 train, evaluate, test 함수 분리로 유지보수 용이, 정확도 및 손실 계산 방식 추가 
5. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
6. 모델 저장 및 불러오기
7. 테스트 및 시각화
8. 웹에서 MNIST 숫자 분류기 웹앱 데모 - 사용자 입력 방식(테스트셋에서 무작위 이미지 선택, 사용자가 직접 이미지 업로드, 사용자가 직접 그리기), 모델 추론(학습된 MLP 모델 로딩 (torch.load), 이미지 전처리 후 예측 수행, 결과 시각화(예측 결과 출력 (정답 vs 예측), Confusion Matrix 및 오차 분석, 틀린 예측 샘플 시각화))
---
### 1. MLP(Multi-Layer Perceptron) DeepLearning : basic dataset 사용
> Model - 01_mlp.ipynb
> Streamlit 웹앱 - app_01_mlp_model.py, app_01_mlp_model_csv_upload.py, app_01_mlp_model_csv_upload_download.py
1. MLP 모델을 기반으로 성능 평가 및 시각화 강화
2. Dataset 및 DataLoader를 활용한 데이터 처리
3. 하이퍼파라미터 튜닝(학습률, 은닉층 크기 등)
4. 정확도 계산(accuracy_score), 혼돈 행렬 계산(confusion_matrix), Confusion Matrix 시각화, 정밀도, 재현율, F1-score 등 출력(classification_report)
5. 모델 저장 및 불러오기
6. 테스트 및 시각화
7. 웹에서 MLP 이진 뷴류기 데모 - 사용자가 숫자를 입력하면 예측 결과가 바로 표시되고, 입력값에 해당하는 위치에 빨간 선이 그려진 예측 곡선이 함께 나타남, CSV 업로드를 통한 배치 예측 및 시각화, 예측 결과를 CSV 파일로 다운로드할 수 있도록 기능, 이렇게 하면 사용자가 업로드한 데이터에 대한 예측 결과를 저장하고 활용
---


project
    - Kaggle Pima Indians Diabetes Dataset - Kaggle 당뇨병 발병 예측
    - Kaggle Titanic Dataset - Kaggle 타이타닉 생존자 예측
    - Kaggle brain tumor Image Classification (MRI) - Kaggle 뇌종양(Brain Tumor) 이미지 분류, pre-trained model(mobilenet_v2) 적용
    - GTSRB (German Traffic Sign Recognition Benchmark) - 표지판(Traffic sign) 이미지 분류, CNN 모델 아키텍처 적용
    - Kaggle Surface Crack Detection - 콘크리트 표면 결함 (Surface Crack Detection) 이미지 분류, CNN 모델 아키텍처 적용
    - Kaggle COVID 19 Radiography - COVID 19 감염 예측, COVID(코로나)/Viral Pneumonia(바이러스성 폐럼)/Lung Opacity(폐 음영 - 폐렴,폐암,간질성 폐 질환 등 다양한 원인)/Normal(정상) 이미지 분류, pre-trained model(mobilenet_v2) 적용
    - Kaggle Breast Ultrasound Image Classification - Kaggle Breast Ultrasound (유방초음파), normal(정상), benign(양성), malignant(악성) 예측 이미지 분류, pre-trained model(mobilenet_v2) 적용
    - Kaggle Dog Breed Image Classification Dataset - Kaggle Dog Breed Image 은 개 품종 예측 이미지 분류
LLM
    - Transformers - 트랜스포머 아키텍처(임베딩, 어텐션, 정규화, 피드 포워드, 인코더, 디코더)
    - Transformers - 트랜스포머 허깅페이스(라이브러리)
    - Transformers - 트랜스포머 허깅페이스 허브에 모델 업로드, 한국어 기사(연합뉴스 데이터셋) 제목을 바탕으로 기사의 카테고리를 분류하는 텍스트 분류 및 추론, pre-trained model(klue/roberta-base) 적용
    - Transformers - 트랜스포머 허깅페이스 허브에 모델 업로드, MRPC(Microsoft Research Paraphrase Corpus) 데이터셋 바탕으로 레이블(1:유사,0:비유사) 영어 문장 유사도 카테고리를 분류하는 텍스트 분류 및 추론, pre-trained model(bert-base-uncased) 적용
    - Transformers - 기초적인 문장 데이터셋 적용하여 Small Language Model(SLLM) 구축 및 텍스트 생성
    - Transformers - 트랜스포머 허깅페이스 데이터셋(WikiText-2)을 적용하여 영어 Small Language Model(SLLM) 구축 및 텍스트 생성
    - Transformers - 트랜스포머 허깅페이스 데이터셋(ShareGPT-KO)을 적용하여 한글 Small Language Model(SLLM) 구축 및 텍스트 생성
    - Transformers - 트랜스포머 데이터셋(ChatbotData)을 적용하여 한글 Small Language Model(SLLM) 구축 및 텍스트 생성